from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tempfile import gettempdir
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer
from bank_langchain_agent import get_chroma_client, reset_chroma_client_cache


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"


DATE_CANDIDATES = (
    "date",
    "transaction date",
    "txn date",
    "posted date",
    "value date",
)

DESCRIPTION_CANDIDATES = (
    "description",
    "narration",
    "remarks",
    "particulars",
    "details",
    "transaction details",
)

REFERENCE_CANDIDATES = (
    "reference",
    "ref no",
    "ref",
    "transaction id",
    "utr",
    "cheque no",
)

DEBIT_CANDIDATES = (
    "debit",
    "withdrawal",
    "withdrawals",
    "withdrawal amt.",
    "withdrawal amt",
    "dr",
)
CREDIT_CANDIDATES = (
    "credit",
    "deposit",
    "deposits",
    "deposit amt.",
    "deposit amt",
    "cr",
)
AMOUNT_CANDIDATES = ("amount", "transaction amount")
BALANCE_CANDIDATES = ("balance", "available balance", "closing balance")

DATE_FORMATS = (
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d/%m/%y",
    "%d-%m-%y",
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%m/%d/%y",
    "%d %b %Y",
    "%d %B %Y",
)

CSV_ENCODINGS = (
    "utf-8-sig",
    "utf-16",
    "utf-16-le",
    "utf-16-be",
    "cp1252",
    "latin-1",
)


@dataclass
class ParsedTransaction:
    transaction_id: str
    document: str
    metadata: dict[str, Any]


def normalize_header(header: str) -> str:
    return " ".join(header.strip().lower().replace("_", " ").split())


def first_matching_column(
    fieldnames: list[str],
    candidates: tuple[str, ...],
    explicit_name: str | None = None,
) -> str | None:
    normalized_to_original = {normalize_header(name): name for name in fieldnames}
    if explicit_name:
        return normalized_to_original.get(normalize_header(explicit_name), explicit_name)

    for candidate in candidates:
        if candidate in normalized_to_original:
            return normalized_to_original[candidate]

    for normalized_name, original_name in normalized_to_original.items():
        for candidate in candidates:
            if candidate in normalized_name or normalized_name in candidate:
                return original_name
    return None


def clean_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def is_separator_value(value: str) -> bool:
    cleaned = clean_value(value)
    return bool(cleaned) and set(cleaned) == {"*"}


def parse_decimal(raw_value: str) -> Decimal | None:
    value = clean_value(raw_value)
    if not value:
        return None

    negative = False
    if value.startswith("(") and value.endswith(")"):
        negative = True
        value = value[1:-1]

    value = value.replace(",", "").replace("₹", "").replace("$", "")

    if value.endswith("-"):
        negative = True
        value = value[:-1]

    try:
        amount = Decimal(value)
    except InvalidOperation:
        return None

    return -amount if negative else amount


def parse_date(raw_value: str) -> str:
    value = clean_value(raw_value)
    if not value:
        return ""

    for date_format in DATE_FORMATS:
        try:
            return datetime.strptime(value, date_format).date().isoformat()
        except ValueError:
            continue

    return value


def stringify_decimal(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return format(value, "f")


def build_document(
    date_value: str,
    description: str,
    amount: Decimal | None,
    transaction_type: str,
    balance: Decimal | None,
    reference: str,
    row: dict[str, Any],
) -> str:
    parts = [
        f"Date: {date_value or 'unknown'}",
        f"Description: {description or 'unknown'}",
        f"Type: {transaction_type}",
    ]

    if amount is not None:
        parts.append(f"Amount: {format(amount, 'f')}")
    if balance is not None:
        parts.append(f"Balance: {format(balance, 'f')}")
    if reference:
        parts.append(f"Reference: {reference}")

    extras = {
        key: clean_value(value)
        for key, value in row.items()
        if clean_value(value)
    }
    if extras:
        parts.append(f"Raw Row: {json.dumps(extras, ensure_ascii=True, sort_keys=True)}")

    return " | ".join(parts)


def infer_amount_and_type(
    row: dict[str, Any],
    debit_column: str | None,
    credit_column: str | None,
    amount_column: str | None,
) -> tuple[Decimal | None, str]:
    debit = parse_decimal(row.get(debit_column, "")) if debit_column else None
    credit = parse_decimal(row.get(credit_column, "")) if credit_column else None
    amount = parse_decimal(row.get(amount_column, "")) if amount_column else None

    if debit is not None and debit != 0:
        return -abs(debit), "debit"
    if credit is not None and credit != 0:
        return abs(credit), "credit"
    if amount is not None:
        return amount, "credit" if amount >= 0 else "debit"
    return None, "unknown"


def transaction_hash(source_file: Path, row_number: int, document: str) -> str:
    seed = f"{source_file}:{row_number}:{document}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def read_csv_rows(csv_path: Path) -> list[list[str]]:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            with csv_path.open("r", encoding=encoding, newline="") as handle:
                sample = handle.read(4096)
                handle.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                except csv.Error:
                    dialect = csv.excel
                return list(csv.reader(handle, dialect))
        except (UnicodeDecodeError, UnicodeError) as error:
            last_error = error

    if last_error is not None:
        raise last_error
    raise ValueError("Unable to read CSV file.")


def locate_header_row(csv_path: Path) -> tuple[list[str], list[list[str]]]:
    rows = read_csv_rows(csv_path)

    for index, row in enumerate(rows):
        normalized_row = [normalize_header(cell) for cell in row]
        if "date" in normalized_row and (
            "narration" in normalized_row or "description" in normalized_row
        ):
            return row, rows[index + 1 :]

        if any("date" in cell for cell in normalized_row) and (
            any(
                keyword in cell
                for cell in normalized_row
                for keyword in ("narration", "description", "details", "remarks", "particulars")
            )
            or any(
                keyword in cell
                for cell in normalized_row
                for keyword in ("withdraw", "deposit", "debit", "credit", "amount", "balance")
            )
        ):
            return row, rows[index + 1 :]

    if not rows:
        raise ValueError("CSV file is empty.")

    return rows[0], rows[1:]


def parse_transactions(
    csv_path: Path,
    date_column: str | None,
    description_column: str | None,
    debit_column: str | None,
    credit_column: str | None,
    amount_column: str | None,
    balance_column: str | None,
    reference_column: str | None,
) -> list[ParsedTransaction]:
    fieldnames, data_rows = locate_header_row(csv_path)
    if not fieldnames:
        raise ValueError("CSV file has no header row.")

    date_column = first_matching_column(fieldnames, DATE_CANDIDATES, date_column)
    description_column = first_matching_column(
        fieldnames, DESCRIPTION_CANDIDATES, description_column
    )
    debit_column = first_matching_column(fieldnames, DEBIT_CANDIDATES, debit_column)
    credit_column = first_matching_column(fieldnames, CREDIT_CANDIDATES, credit_column)
    amount_column = first_matching_column(fieldnames, AMOUNT_CANDIDATES, amount_column)
    balance_column = first_matching_column(fieldnames, BALANCE_CANDIDATES, balance_column)
    reference_column = first_matching_column(
        fieldnames, REFERENCE_CANDIDATES, reference_column
    )

    if not date_column or not description_column:
        excluded_columns = {
            column
            for column in (
                date_column,
                debit_column,
                credit_column,
                amount_column,
                balance_column,
                reference_column,
            )
            if column
        }
        if not description_column:
            for fieldname in fieldnames:
                normalized_fieldname = normalize_header(fieldname)
                if fieldname in excluded_columns:
                    continue
                if any(token in normalized_fieldname for token in ("desc", "detail", "remark", "particular", "narr")):
                    description_column = fieldname
                    break
            if not description_column:
                for fieldname in fieldnames:
                    if fieldname not in excluded_columns:
                        description_column = fieldname
                        break

        if not date_column:
            for fieldname in fieldnames:
                if "date" in normalize_header(fieldname):
                    date_column = fieldname
                    break

        if not date_column or not description_column:
            raise ValueError(
                "Unable to identify the date/description columns. "
                "Use --date-column and --description-column to specify them."
            )

    parsed_transactions: list[ParsedTransaction] = []
    for index, raw_row in enumerate(data_rows, start=1):
        row = {
            fieldname: raw_row[position] if position < len(raw_row) else ""
            for position, fieldname in enumerate(fieldnames)
        }
        if not any(clean_value(value) for value in row.values()):
            continue

        date_value = parse_date(row.get(date_column, ""))
        description = clean_value(row.get(description_column, ""))
        if not date_value or not description:
            continue
        if is_separator_value(date_value) or is_separator_value(description):
            continue

        amount, transaction_type = infer_amount_and_type(
            row, debit_column, credit_column, amount_column
        )
        balance = parse_decimal(row.get(balance_column, "")) if balance_column else None
        reference = clean_value(row.get(reference_column, "")) if reference_column else ""

        document = build_document(
            date_value=date_value,
            description=description,
            amount=amount,
            transaction_type=transaction_type,
            balance=balance,
            reference=reference,
            row=row,
        )
        transaction_id = transaction_hash(csv_path, index, document)
        metadata = {
            "row_number": index,
            "date": date_value,
            "description": description,
            "amount": stringify_decimal(amount),
            "transaction_type": transaction_type,
            "balance": stringify_decimal(balance),
            "reference": reference or None,
            "source_file": str(csv_path),
        }
        parsed_transactions.append(
            ParsedTransaction(
                transaction_id=transaction_id,
                document=document,
                metadata=metadata,
            )
        )

    return parsed_transactions


def embed_documents(
    embedding_model: SentenceTransformer,
    documents: list[str],
    batch_size: int,
) -> list[list[float]]:
    all_embeddings: list[list[float]] = []
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        all_embeddings.extend(
            embedding_model.encode(
                batch, batch_size=len(batch), show_progress_bar=False
            ).tolist()
        )
    return all_embeddings


def upsert_transactions(
    transactions: list[ParsedTransaction],
    persist_directory: Path,
    collection_name: str,
    embedding_model: str,
    batch_size: int,
) -> None:
    if not transactions:
        print("No transactions found in the CSV.")
        return

    embedding_client = SentenceTransformer(embedding_model)
    persist_directory.mkdir(parents=True, exist_ok=True)
    chroma_client = get_chroma_client(str(persist_directory))
    try:
        collection = chroma_client.get_or_create_collection(name=collection_name)
    except Exception:
        resolved_directory = persist_directory.resolve()
        generated_locations = (
            (DATA_DIR / "chroma_bank_transactions").resolve(),
            Path(gettempdir()).resolve(),
        )
        can_reset_generated_store = (
            resolved_directory == generated_locations[0]
            or generated_locations[1] in resolved_directory.parents
        )
        if not can_reset_generated_store:
            raise
        shutil.rmtree(persist_directory, ignore_errors=True)
        persist_directory.mkdir(parents=True, exist_ok=True)
        reset_chroma_client_cache()
        chroma_client = get_chroma_client(str(persist_directory))
        collection = chroma_client.get_or_create_collection(name=collection_name)

    documents = [transaction.document for transaction in transactions]
    embeddings = embed_documents(
        embedding_model=embedding_client,
        documents=documents,
        batch_size=batch_size,
    )

    collection.upsert(
        ids=[transaction.transaction_id for transaction in transactions],
        documents=documents,
        metadatas=[transaction.metadata for transaction in transactions],
        embeddings=embeddings,
    )

    print(
        f"Stored {len(transactions)} transactions in collection "
        f"'{collection_name}' at '{persist_directory}'."
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest a bank statement CSV into ChromaDB using a local open-source embedding model."
    )
    parser.add_argument("csv_path", type=Path, help="Path to the bank statement CSV.")
    parser.add_argument(
        "--persist-directory",
        type=Path,
        default=DATA_DIR / "chroma_bank_transactions",
        help="Directory where the ChromaDB files should be stored.",
    )
    parser.add_argument(
        "--collection-name",
        default="bank_transactions",
        help="ChromaDB collection name.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence Transformers model name or local path to use for embeddings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of rows to embed per request.",
    )
    parser.add_argument("--date-column", help="Explicit date column name.")
    parser.add_argument("--description-column", help="Explicit description column name.")
    parser.add_argument("--debit-column", help="Explicit debit column name.")
    parser.add_argument("--credit-column", help="Explicit credit column name.")
    parser.add_argument("--amount-column", help="Explicit amount column name.")
    parser.add_argument("--balance-column", help="Explicit balance column name.")
    parser.add_argument("--reference-column", help="Explicit reference column name.")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    transactions = parse_transactions(
        csv_path=args.csv_path,
        date_column=args.date_column,
        description_column=args.description_column,
        debit_column=args.debit_column,
        credit_column=args.credit_column,
        amount_column=args.amount_column,
        balance_column=args.balance_column,
        reference_column=args.reference_column,
    )

    upsert_transactions(
        transactions=transactions,
        persist_directory=args.persist_directory,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
