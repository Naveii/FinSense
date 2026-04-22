from __future__ import annotations

import asyncio
import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

import chromadb
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool, tool
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"


def default_chroma_dir() -> Path:
    configured_dir = os.getenv("CHROMA_PERSIST_DIR")
    if configured_dir:
        return Path(configured_dir)

    # Streamlit Cloud checks the repo out under /mount/src. Keep generated
    # Chroma files in runtime storage so stale or partially-created DB files in
    # the source checkout cannot break collection creation on reboot.
    if str(PROJECT_ROOT).startswith("/mount/src"):
        runtime_root = Path(os.getenv("TMPDIR", "/tmp")) / "bank_statement_insights"
        return runtime_root / "chroma_bank_transactions"

    return DATA_DIR / "chroma_bank_transactions"


DEFAULT_CHROMA_DIR = default_chroma_dir()
DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION_NAME", "bank_transactions")
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
DEFAULT_AGENT_MODEL = os.getenv("AGENT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

CATEGORY_OPTIONS = [
    "income",
    "emi_loan",
    "rent_housing",
    "utilities",
    "groceries",
    "food_dining",
    "shopping",
    "travel_transport",
    "entertainment",
    "healthcare",
    "education",
    "cash_transfer",
    "savings_investment",
    "other",
]

DISCRETIONARY_CATEGORIES = {
    "food_dining",
    "shopping",
    "travel_transport",
    "entertainment",
    "cash_transfer",
}

INCOME_KEYWORDS = ("salary", "payroll", "income", "bonus", "reimbursement", "interest")
EMI_KEYWORDS = ("emi", "loan", "nach", "ecs", "autopay", "finance")
PERSONAL_FINANCE_KEYWORDS = (
    "bank",
    "statement",
    "transaction",
    "transactions",
    "spend",
    "spent",
    "expense",
    "expenses",
    "savings",
    "saving",
    "budget",
    "income",
    "salary",
    "credit",
    "debit",
    "loan",
    "emi",
    "merchant",
    "upi",
    "balance",
    "cashflow",
    "cash flow",
    "finance",
    "financial",
    "payment",
    "payments",
)

STOPWORDS = {
    "show",
    "find",
    "list",
    "all",
    "my",
    "me",
    "for",
    "from",
    "with",
    "that",
    "than",
    "under",
    "over",
    "between",
    "during",
    "within",
    "only",
    "just",
    "what",
    "which",
    "when",
    "where",
    "much",
    "many",
    "spent",
    "spend",
    "payments",
    "payment",
    "transactions",
    "transaction",
    "debits",
    "debit",
    "credits",
    "credit",
    "large",
    "largest",
    "highest",
    "biggest",
    "total",
    "sum",
    "merchant",
    "category",
    "categories",
    "group",
    "score",
    "health",
    "financial",
    "finance",
    "rate",
    "ratio",
    "income",
    "loan",
    "loans",
    "emi",
    "expense",
    "expenses",
    "savings",
    "statement",
    "bank",
    "upi",
}

MONTH_LOOKUP = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

RETRIEVAL_HINTS = (
    "show",
    "find",
    "list",
    "which",
    "when",
    "where",
    "largest",
    "highest",
    "biggest",
    "total",
    "sum",
    "debit",
    "credit",
    "upi",
    "transaction",
    "transactions",
    "payment",
    "payments",
)

SPENDING_HINTS = (
    "category",
    "categories",
    "group",
    "grouping",
    "merchant type",
    "merchant types",
    "merchant",
    "breakdown",
    "split",
    "spending pattern",
    "spending patterns",
)

HEALTH_HINTS = (
    "health score",
    "financial health",
    "savings rate",
    "emi",
    "emi ratio",
    "income ratio",
    "discretionary",
    "net savings",
    "cash flow",
    "cashflow",
)


def parse_amount(value: Any) -> Decimal:
    if value in (None, ""):
        return Decimal("0")
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return Decimal("0")


def format_currency(value: Any) -> str:
    amount = parse_amount(value)
    if amount < 0:
        return f"-INR {abs(amount):,.2f}"
    return f"INR {amount:,.2f}"


def format_month_range(dates: list[str]) -> str:
    parsed_dates = []
    for value in dates:
        try:
            parsed_dates.append(datetime.strptime(value, "%Y-%m-%d"))
        except ValueError:
            continue
    if not parsed_dates:
        return "unknown period"
    parsed_dates.sort()
    start = parsed_dates[0]
    end = parsed_dates[-1]
    if start.year == end.year and start.month == end.month:
        return start.strftime("%b %Y")
    return f"{start.strftime('%b %Y')} to {end.strftime('%b %Y')}"


def merchant_hint(description: str) -> str:
    if not description:
        return "matched merchants"
    parts = description.split("-")
    if len(parts) > 1:
        return parts[1].strip().title()
    return description[:40].title()


def merchant_name_from_description(description: str) -> str:
    if not description:
        return "matched merchants"
    normalized = description.replace("/", "-").replace("@", "-")
    pieces = [piece.strip() for piece in normalized.split("-") if piece.strip()]
    for piece in pieces:
        if piece.lower() in {"upi", "imps", "neft", "card", "pos", "ach"}:
            continue
        if any(char.isalpha() for char in piece):
            return piece.title()
    return description[:40].title()


def health_score_band(score: Decimal) -> str:
    if score >= Decimal("80"):
        return "strong"
    if score >= Decimal("60"):
        return "stable"
    if score >= Decimal("40"):
        return "mixed"
    return "stretched"


def parse_iso_date(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def infer_reference_year(transactions: list[dict[str, Any]]) -> int | None:
    years = []
    for transaction in transactions:
        parsed = parse_iso_date(str(transaction.get("metadata", {}).get("date", "")))
        if parsed:
            years.append(parsed.year)
    return max(years) if years else None


def extract_month_range(question: str, reference_year: int | None) -> tuple[str | None, str | None]:
    normalized = question.lower()
    found: list[int] = []
    for token, month_number in MONTH_LOOKUP.items():
        if token in normalized:
            if month_number not in found:
                found.append(month_number)

    if not found or reference_year is None:
        return None, None

    start_month = min(found)
    end_month = max(found)
    start = datetime(reference_year, start_month, 1)
    if end_month == 12:
        end = datetime(reference_year, 12, 31)
    else:
        end = datetime(reference_year, end_month + 1, 1) - timedelta(days=1)
    return start.date().isoformat(), end.date().isoformat()


def extract_amount_thresholds(question: str) -> tuple[Decimal | None, Decimal | None]:
    normalized = question.lower().replace(",", "")
    above_match = re.search(r"(?:above|over|greater than|more than)\s+(\d+(?:\.\d+)?)", normalized)
    below_match = re.search(r"(?:below|under|less than)\s+(\d+(?:\.\d+)?)", normalized)
    min_amount = Decimal(above_match.group(1)) if above_match else None
    max_amount = Decimal(below_match.group(1)) if below_match else None
    return min_amount, max_amount


def extract_merchant_terms(question: str) -> list[str]:
    preferred_match = re.search(
        r"(?:on|to|from|at)\s+([a-z0-9_@.-]+(?:\s+[a-z0-9_@.-]+)?)",
        question.lower(),
    )
    if preferred_match:
        phrase = re.sub(r"[^a-z0-9\s]", " ", preferred_match.group(1))
        preferred_terms = [token for token in phrase.split() if len(token) > 2]
        preferred_terms = [
            token
            for token in preferred_terms
            if token not in STOPWORDS and token not in MONTH_LOOKUP
        ]
        if preferred_terms:
            return preferred_terms

    normalized = re.sub(r"[^a-z0-9\s]", " ", question.lower())
    tokens = [token for token in normalized.split() if len(token) > 2]
    return [
        token
        for token in tokens
        if token not in STOPWORDS and token not in MONTH_LOOKUP
    ]


def build_query_filters(question: str, transactions: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = question.lower()
    min_amount, max_amount = extract_amount_thresholds(normalized)
    transaction_type = None
    if "debit" in normalized or "debits" in normalized:
        transaction_type = "debit"
    elif "credit" in normalized or "credits" in normalized:
        transaction_type = "credit"

    start_date, end_date = extract_month_range(normalized, infer_reference_year(transactions))
    merchant_terms = extract_merchant_terms(normalized)

    return {
        "transaction_type": transaction_type,
        "min_amount": str(min_amount) if min_amount is not None else None,
        "max_amount": str(max_amount) if max_amount is not None else None,
        "start_date": start_date,
        "end_date": end_date,
        "merchant_terms": merchant_terms,
    }


def record_matches_filters(record: dict[str, Any], filters: dict[str, Any]) -> bool:
    metadata = record.get("metadata", {})
    description = str(metadata.get("description", "") or "").lower()
    amount = abs(parse_amount(metadata.get("amount")))
    transaction_type = str(metadata.get("transaction_type", "") or "").lower()
    date_value = str(metadata.get("date", "") or "")

    if filters.get("transaction_type") and transaction_type != filters["transaction_type"]:
        return False

    min_amount = filters.get("min_amount")
    if min_amount is not None and amount < parse_amount(min_amount):
        return False

    max_amount = filters.get("max_amount")
    if max_amount is not None and amount > parse_amount(max_amount):
        return False

    start_date = filters.get("start_date")
    if start_date and date_value < start_date:
        return False

    end_date = filters.get("end_date")
    if end_date and date_value > end_date:
        return False

    merchant_terms = filters.get("merchant_terms") or []
    if merchant_terms and not any(term in description for term in merchant_terms):
        return False

    return True


def sort_records_for_question(records: list[dict[str, Any]], question: str) -> list[dict[str, Any]]:
    normalized = question.lower()
    if any(token in normalized for token in ("largest", "highest", "biggest", "max")):
        return sorted(
            records,
            key=lambda record: abs(parse_amount(record.get("metadata", {}).get("amount"))),
            reverse=True,
        )
    if any(token in normalized for token in ("latest", "recent", "newest")):
        return sorted(records, key=lambda record: str(record.get("metadata", {}).get("date", "")), reverse=True)
    return records


class AsyncCompatibleChatHuggingFace(ChatHuggingFace):
    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        return await asyncio.to_thread(self.invoke, input, config=config, **kwargs)

    async def agenerate_prompt(
        self, prompts: Any, stop: Any = None, callbacks: Any = None, **kwargs: Any
    ) -> Any:
        return await asyncio.to_thread(
            self.generate_prompt, prompts, stop=stop, callbacks=callbacks, **kwargs
        )

    async def agenerate(
        self, messages: Any, stop: Any = None, callbacks: Any = None, **kwargs: Any
    ) -> Any:
        return await asyncio.to_thread(
            self.generate, messages, stop=stop, callbacks=callbacks, **kwargs
        )


def build_local_chat_model(model_id: str) -> ChatHuggingFace:
    llm = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 128, "do_sample": False, "return_full_text": False},
    )
    return AsyncCompatibleChatHuggingFace(llm=llm, model_id=model_id)


class FinanceScopeGuardrail:
    def check(self, question: str) -> tuple[bool, str]:
        normalized = question.strip().lower()
        if any(keyword in normalized for keyword in PERSONAL_FINANCE_KEYWORDS):
            return True, ""

        return (
            False,
            "I can only help with personal finance questions tied to your statements, spending, savings, merchants, income, loans, or financial health.",
        )


class TransactionStore:
    def __init__(
        self,
        persist_directory: Path,
        collection_name: str,
        embedding_model_name: str,
    ) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._embedding_model: SentenceTransformer | None = None
        self._all_transactions_cache: list[dict[str, Any]] | None = None
        self.chroma_client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)

    def _get_embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def semantic_search(self, query: str, top_k: int) -> dict[str, Any]:
        query_embedding = self._get_embedding_model().encode(
            [query], show_progress_bar=False
        )[0].tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max(top_k * 3, top_k),
            include=["documents", "metadatas", "distances"],
        )

        matches = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for index, transaction_id in enumerate(ids):
            matches.append(
                {
                    "transaction_id": transaction_id,
                    "distance": distances[index] if index < len(distances) else None,
                    "metadata": metadatas[index] if index < len(metadatas) else {},
                    "document": documents[index] if index < len(documents) else "",
                }
            )

        matches = self._clean_records(matches)[:top_k]
        return {"query": query, "top_k": top_k, "matches": matches}

    def all_transactions(self) -> list[dict[str, Any]]:
        if self._all_transactions_cache is not None:
            return self._all_transactions_cache

        total = self.collection.count()
        results = self.collection.get(
            limit=total,
            include=["documents", "metadatas"],
        )
        transactions = []
        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        for index, transaction_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else {}
            transactions.append(
                {
                    "transaction_id": transaction_id,
                    "document": documents[index] if index < len(documents) else "",
                    "metadata": metadata,
                }
            )
        self._all_transactions_cache = self._clean_records(transactions)
        return self._all_transactions_cache

    def _clean_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cleaned = []
        seen: set[tuple[str, str, str, str]] = set()

        for record in records:
            metadata = record.get("metadata", {})
            date_value = str(metadata.get("date", "") or "")
            description = str(metadata.get("description", "") or "")
            amount = parse_amount(metadata.get("amount"))
            transaction_type = str(metadata.get("transaction_type", "") or "")

            if amount == 0:
                continue
            if len(date_value) != 10 or date_value.count("-") != 2:
                continue
            dedupe_key = (date_value, description, str(amount), transaction_type)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            cleaned.append(record)

        return cleaned


class MerchantClassifier:
    def __init__(self, llm_loader: Callable[[], ChatHuggingFace] | None = None) -> None:
        self._llm_loader = llm_loader
        self._chain: Any | None = None
        self.cache: dict[str, str] = {}

    def _get_chain(self) -> Any | None:
        if self._chain is not None:
            return self._chain
        if self._llm_loader is None:
            return None

        self._chain = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Classify the bank transaction into exactly one merchant category. "
                        "Return only the category label from this list: "
                        + ", ".join(CATEGORY_OPTIONS)
                        + ".",
                    ),
                    (
                        "human",
                        "Transaction description: {description}\n"
                        "Amount: {amount}\n"
                        "Transaction type: {transaction_type}\n"
                        "Category:",
                    ),
                ]
            )
            | self._llm_loader()
            | StrOutputParser()
        )
        return self._chain

    def classify(self, description: str, amount: str, transaction_type: str) -> str:
        cache_key = f"{description}|{amount}|{transaction_type}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        normalized = self._fallback(description, amount, transaction_type)
        if normalized == "other":
            chain = self._get_chain()
            if chain is not None:
                raw_output = chain.invoke(
                    {
                        "description": description or "unknown",
                        "amount": amount or "0",
                        "transaction_type": transaction_type or "unknown",
                    }
                )
                candidate = raw_output.strip().splitlines()[0].strip().lower().replace(" ", "_")
                if candidate in CATEGORY_OPTIONS:
                    normalized = candidate

        self.cache[cache_key] = normalized
        return normalized

    def _fallback(self, description: str, amount: str, transaction_type: str) -> str:
        text = f"{description} {transaction_type}".lower()
        amount_value = parse_amount(amount)
        if any(keyword in text for keyword in INCOME_KEYWORDS) or (
            transaction_type == "credit" and amount_value >= Decimal("10000")
        ):
            return "income"
        if any(keyword in text for keyword in EMI_KEYWORDS):
            return "emi_loan"
        if "rent" in text:
            return "rent_housing"
        if any(keyword in text for keyword in ("electric", "water", "gas", "broadband", "recharge")):
            return "utilities"
        if any(keyword in text for keyword in ("swiggy", "zomato", "restaurant", "cafe", "food")):
            return "food_dining"
        if any(keyword in text for keyword in ("amazon", "flipkart", "myntra", "store")):
            return "shopping"
        if any(keyword in text for keyword in ("uber", "ola", "metro", "flight", "travel")):
            return "travel_transport"
        if any(keyword in text for keyword in ("movie", "netflix", "spotify", "prime")):
            return "entertainment"
        if any(keyword in text for keyword in ("hospital", "clinic", "pharma")):
            return "healthcare"
        if any(keyword in text for keyword in ("school", "college", "course", "fees")):
            return "education"
        if "upi-" in text or "imps" in text or "neft" in text:
            return "cash_transfer"
        return "other"


class FinancialTools:
    def __init__(self, store: TransactionStore, classifier: MerchantClassifier) -> None:
        self.store = store
        self.classifier = classifier

    def retrieval_tool(self) -> BaseTool:
        @tool("rag_retrieval_tool")
        def rag_retrieval_tool(query: str, top_k: int = 12) -> str:
            """Semantic retrieval over embedded transactions. Use for lookup, search, and evidence gathering."""
            semantic_matches = self.store.semantic_search(query=query, top_k=max(top_k * 3, 24)).get(
                "matches", []
            )
            all_transactions = self.store.all_transactions()
            filters = build_query_filters(query, all_transactions)

            filtered_transactions = [
                transaction
                for transaction in all_transactions
                if record_matches_filters(transaction, filters)
            ]

            if filters.get("merchant_terms") or filters.get("min_amount") or filters.get("max_amount") or filters.get("start_date") or filters.get("transaction_type"):
                matched_ids = {transaction["transaction_id"] for transaction in filtered_transactions}
                records = [
                    match for match in semantic_matches if match["transaction_id"] in matched_ids
                ]
                if not records:
                    records = filtered_transactions
            else:
                records = semantic_matches

            records = sort_records_for_question(records, query)[:top_k]
            total_amount = sum(
                (abs(parse_amount(record.get("metadata", {}).get("amount"))) for record in records),
                Decimal("0"),
            )
            result = {
                "query": query,
                "top_k": top_k,
                "matches": records,
                "applied_filters": filters,
                "aggregate": {
                    "count": len(records),
                    "total_amount": str(total_amount),
                },
            }
            return json.dumps(result, indent=2)

        return rag_retrieval_tool

    def spending_category_tool(self) -> BaseTool:
        @tool("spending_category_analyser")
        def spending_category_analyser(query: str = "") -> str:
            """Group transactions by merchant type and summarize spending using an LLM classifier."""
            transactions = self.store.all_transactions()
            if query:
                filters = build_query_filters(query, transactions)
                matched_ids = {
                    match["transaction_id"]
                    for match in self.store.semantic_search(query=query, top_k=min(25, len(transactions))).get("matches", [])
                }
                transactions = [
                    transaction
                    for transaction in transactions
                    if transaction["transaction_id"] in matched_ids or record_matches_filters(transaction, filters)
                ]

            category_totals: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
            grouped_transactions: dict[str, list[dict[str, Any]]] = defaultdict(list)

            for transaction in transactions:
                metadata = transaction["metadata"]
                amount = parse_amount(metadata.get("amount"))
                category = self.classifier.classify(
                    description=str(metadata.get("description", "")),
                    amount=str(metadata.get("amount", "0")),
                    transaction_type=str(metadata.get("transaction_type", "unknown")),
                )
                grouped_transactions[category].append(
                    {
                        "transaction_id": transaction["transaction_id"],
                        "date": metadata.get("date"),
                        "description": metadata.get("description"),
                        "amount": str(amount),
                        "transaction_type": metadata.get("transaction_type"),
                    }
                )
                if amount < 0:
                    category_totals[category] += abs(amount)

            summary = {
                "query_filter": query or None,
                "categories": [
                    {
                        "category": category,
                        "spend_total": str(category_totals[category]),
                        "transaction_count": len(items),
                        "transactions": items,
                    }
                    for category, items in sorted(
                        grouped_transactions.items(),
                        key=lambda item: category_totals[item[0]],
                        reverse=True,
                    )
                ],
            }
            return json.dumps(summary, indent=2)

        return spending_category_analyser

    def financial_health_tool(self) -> BaseTool:
        @tool("financial_health_score_tool")
        def financial_health_score_tool(query: str = "") -> str:
            """Compute financial health metrics: savings rate, EMI-to-income ratio, discretionary spend percentage."""
            transactions = self.store.all_transactions()
            total_income = Decimal("0")
            total_expenses = Decimal("0")
            emi_spend = Decimal("0")
            discretionary_spend = Decimal("0")
            categorized = []

            for transaction in transactions:
                metadata = transaction["metadata"]
                amount = parse_amount(metadata.get("amount"))
                description = str(metadata.get("description", ""))
                transaction_type = str(metadata.get("transaction_type", "unknown"))
                category = self.classifier.classify(
                    description=description,
                    amount=str(metadata.get("amount", "0")),
                    transaction_type=transaction_type,
                )
                categorized.append(
                    {
                        "transaction_id": transaction["transaction_id"],
                        "date": metadata.get("date"),
                        "description": description,
                        "amount": str(amount),
                        "category": category,
                    }
                )

                if amount > 0:
                    if category == "income" or any(keyword in description.lower() for keyword in INCOME_KEYWORDS):
                        total_income += amount
                elif amount < 0:
                    expense = abs(amount)
                    total_expenses += expense
                    if category == "emi_loan" or any(keyword in description.lower() for keyword in EMI_KEYWORDS):
                        emi_spend += expense
                    if category in DISCRETIONARY_CATEGORIES:
                        discretionary_spend += expense

            if total_income == 0:
                total_income = sum(
                    (
                        parse_amount(transaction["metadata"].get("amount"))
                        for transaction in transactions
                        if parse_amount(transaction["metadata"].get("amount")) > 0
                    ),
                    Decimal("0"),
                )
                income_assumption = "No clear salary/income credits found; used total credits as income proxy."
            else:
                income_assumption = "Income based on credits classified as salary/income."

            net_savings = total_income - total_expenses
            savings_rate = (net_savings / total_income * 100) if total_income else Decimal("0")
            emi_to_income = (emi_spend / total_income * 100) if total_income else Decimal("0")
            discretionary_pct = (
                discretionary_spend / total_income * 100 if total_income else Decimal("0")
            )
            score = max(
                Decimal("0"),
                min(
                    Decimal("100"),
                    Decimal("100")
                    + savings_rate * Decimal("0.5")
                    - emi_to_income * Decimal("0.7")
                    - discretionary_pct * Decimal("0.3"),
                ),
            )

            result = {
                "income_assumption": income_assumption,
                "metrics": {
                    "total_income": str(total_income),
                    "total_expenses": str(total_expenses),
                    "net_savings": str(net_savings),
                    "savings_rate_pct": str(savings_rate.quantize(Decimal("0.01"))),
                    "emi_to_income_ratio_pct": str(emi_to_income.quantize(Decimal("0.01"))),
                    "discretionary_spend_pct": str(discretionary_pct.quantize(Decimal("0.01"))),
                    "financial_health_score": str(score.quantize(Decimal("0.01"))),
                },
                "notes": [
                    "Savings rate = (income - expenses) / income.",
                    "EMI ratio uses transactions classified as emi_loan or matching loan keywords.",
                    "Discretionary spend includes food_dining, shopping, travel_transport, entertainment, and cash_transfer.",
                ],
                "categorized_transactions": categorized,
            }
            return json.dumps(result, indent=2)

        return financial_health_score_tool


def build_citations(selected_tool: str, tool_output: dict[str, Any]) -> list[str]:
    citations: list[str] = []

    if selected_tool == "rag_retrieval_tool":
        matches = tool_output.get("matches", [])
        if matches:
            filters = tool_output.get("applied_filters", {})
            dates = [
                match.get("metadata", {}).get("date", "")
                for match in matches
                if match.get("metadata")
            ]
            descriptions = [
                match.get("metadata", {}).get("description", "")
                for match in matches
                if match.get("metadata")
            ]
            citations.append(
                f"Based on {len(matches)} retrieved transactions related to {merchant_name_from_description(descriptions[0]) if descriptions else 'matched merchants'} across {format_month_range(dates)}."
            )
            if filters:
                filter_parts = []
                if filters.get("transaction_type"):
                    filter_parts.append(filters["transaction_type"])
                if filters.get("min_amount"):
                    filter_parts.append(f"above INR {filters['min_amount']}")
                if filters.get("max_amount"):
                    filter_parts.append(f"below INR {filters['max_amount']}")
                if filters.get("start_date") and filters.get("end_date"):
                    filter_parts.append(f"between {filters['start_date']} and {filters['end_date']}")
                if filters.get("merchant_terms"):
                    filter_parts.append("matching " + ", ".join(filters["merchant_terms"]))
                if filter_parts:
                    citations.append("Applied filters: " + "; ".join(filter_parts) + ".")
            for match in matches[:3]:
                metadata = match.get("metadata", {})
                citations.append(
                    f"{metadata.get('date', 'unknown date')}: {metadata.get('description', 'transaction')} for {format_currency(metadata.get('amount'))}."
                )

    elif selected_tool == "spending_category_analyser":
        categories = tool_output.get("categories", [])
        if categories:
            top_category = categories[0]
            transactions = top_category.get("transactions", [])
            citations.append(
                f"Based on {len(transactions)} transactions in category '{top_category.get('category', 'unknown')}' across {format_month_range([transaction.get('date', '') for transaction in transactions])}."
            )
            citations.append(
                f"Observed spend in that category: {format_currency(top_category.get('spend_total', '0'))}."
            )

    elif selected_tool == "financial_health_score_tool":
        metrics = tool_output.get("metrics", {})
        categorized_transactions = tool_output.get("categorized_transactions", [])
        citations.append(
            f"Based on {len(categorized_transactions)} classified transactions across {format_month_range([transaction.get('date', '') for transaction in categorized_transactions])}."
        )
        citations.append(
            "Income proxy: "
            + tool_output.get("income_assumption", "No assumption text available.")
        )
        citations.append(
            f"Savings rate {metrics.get('savings_rate_pct', '0')}%, EMI-to-income {metrics.get('emi_to_income_ratio_pct', '0')}%, discretionary spend {metrics.get('discretionary_spend_pct', '0')}%."
        )

    return citations


def extract_supporting_contexts(selected_tool: str, tool_output: dict[str, Any]) -> list[str]:
    if selected_tool == "rag_retrieval_tool":
        return [
            match.get("document", "")
            for match in tool_output.get("matches", [])
            if match.get("document")
        ]
    if selected_tool == "spending_category_analyser":
        contexts = []
        for category in tool_output.get("categories", []):
            for transaction in category.get("transactions", []):
                contexts.append(
                    f"Date: {transaction.get('date')} | Description: {transaction.get('description')} | Amount: {transaction.get('amount')} | Type: {transaction.get('transaction_type')} | Category: {category.get('category')}"
                )
        return contexts
    if selected_tool == "financial_health_score_tool":
        return [
            f"Date: {transaction.get('date')} | Description: {transaction.get('description')} | Amount: {transaction.get('amount')} | Category: {transaction.get('category')}"
            for transaction in tool_output.get("categorized_transactions", [])
        ]
    return []


def build_agent_answer(question: str, selected_tool: str, tool_output: dict[str, Any]) -> str:
    question_lower = question.lower()

    if selected_tool == "scope_guardrail":
        return tool_output.get("message", "I can only answer personal finance questions.")

    if selected_tool == "rag_retrieval_tool":
        matches = tool_output.get("matches", [])
        if not matches:
            return "I could not find matching transactions for that question."
        aggregate = tool_output.get("aggregate", {})
        applied_filters = tool_output.get("applied_filters", {})

        debit_matches = [
            match
            for match in matches
            if match.get("metadata", {}).get("transaction_type") == "debit"
        ]
        credit_matches = [
            match
            for match in matches
            if match.get("metadata", {}).get("transaction_type") == "credit"
        ]

        if any(token in question_lower for token in ("largest", "highest", "biggest", "max")):
            candidate_pool = debit_matches if "debit" in question_lower else credit_matches if "credit" in question_lower else matches
            best = max(
                candidate_pool,
                key=lambda match: abs(parse_amount(match.get("metadata", {}).get("amount"))),
            )
            metadata = best.get("metadata", {})
            return (
                f"The largest matching transaction is {metadata.get('description', 'a transaction')} on "
                f"{metadata.get('date', 'an unknown date')} for {format_currency(metadata.get('amount'))}."
            )

        if any(token in question_lower for token in ("how many", "count", "number of")):
            candidate_pool = debit_matches if "debit" in question_lower else credit_matches if "credit" in question_lower else matches
            return f"I found {len(candidate_pool)} matching transactions."

        if "total" in question_lower or "sum" in question_lower:
            candidate_pool = debit_matches if "debit" in question_lower else credit_matches if "credit" in question_lower else matches
            total = sum(
                (abs(parse_amount(match.get("metadata", {}).get("amount"))) for match in candidate_pool),
                Decimal("0"),
            )
            merchant = merchant_name_from_description(
                str(candidate_pool[0].get("metadata", {}).get("description", "")) if candidate_pool else ""
            )
            return f"The total across {len(candidate_pool)} matching transactions for {merchant} is {format_currency(total)}."

        if any(token in question_lower for token in ("spent on", "payments to", "paid to")):
            total = parse_amount(aggregate.get("total_amount", "0"))
            merchant = merchant_name_from_description(
                str(matches[0].get("metadata", {}).get("description", ""))
            )
            return (
                f"You spent {format_currency(total)} across {len(matches)} matching transactions for {merchant}."
            )

        top_match = matches[0]
        metadata = top_match.get("metadata", {})
        amount_values = [
            abs(parse_amount(match.get("metadata", {}).get("amount")))
            for match in matches
        ]
        total = sum(amount_values, Decimal("0"))
        filter_summary = []
        if applied_filters.get("start_date") and applied_filters.get("end_date"):
            filter_summary.append(
                f"between {applied_filters['start_date']} and {applied_filters['end_date']}"
            )
        if applied_filters.get("min_amount"):
            filter_summary.append(f"above INR {applied_filters['min_amount']}")
        return (
            f"I found {len(matches)} relevant transactions totaling {format_currency(total)}. "
            + (f"These results are filtered {' and '.join(filter_summary)}. " if filter_summary else "")
            + (
            f"The strongest match is {metadata.get('description', 'a transaction')} on "
            f"{metadata.get('date', 'an unknown date')} for {format_currency(metadata.get('amount'))}."
            )
        )

    if selected_tool == "spending_category_analyser":
        categories = tool_output.get("categories", [])
        if not categories:
            return "I could not group the transactions into merchant categories."
        top_category = categories[0]
        lead_description = ""
        if top_category.get("transactions"):
            lead_description = str(top_category["transactions"][0].get("description", ""))
        merchant = merchant_name_from_description(lead_description)
        return (
            f"Your heaviest spending bucket here is {top_category.get('category', 'unknown').replace('_', ' ')}, "
            f"with {top_category.get('transaction_count', 0)} transactions adding up to "
            f"{format_currency(top_category.get('spend_total', '0'))}. "
            f"The leading merchant signal in that bucket is {merchant}."
        )

    if selected_tool == "financial_health_score_tool":
        metrics = tool_output.get("metrics", {})
        score = parse_amount(metrics.get("financial_health_score", "0"))
        band = health_score_band(score)
        return (
            f"Your financial health score is {metrics.get('financial_health_score', '0')}, which looks {band}. "
            f"Savings rate is {metrics.get('savings_rate_pct', '0')}%, "
            f"EMI-to-income ratio is {metrics.get('emi_to_income_ratio_pct', '0')}%, "
            f"and discretionary spend is {metrics.get('discretionary_spend_pct', '0')}%."
        )

    return "I processed the question but could not summarize the result cleanly."


class LangChainFinanceAgent:
    def __init__(
        self,
        tools: list[BaseTool],
        llm_loader: Callable[[], ChatHuggingFace] | None = None,
    ) -> None:
        self.tools = {tool.name: tool for tool in tools}
        self.guardrail = FinanceScopeGuardrail()
        self._llm_loader = llm_loader
        self._router_chain: Any | None = None

    def _get_router_chain(self) -> Any | None:
        if self._router_chain is not None:
            return self._router_chain
        if self._llm_loader is None:
            return None

        self._router_chain = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Route the user question to exactly one tool. "
                        "Return only one tool name from this list: "
                        "rag_retrieval_tool, spending_category_analyser, financial_health_score_tool.\n"
                        "Use rag_retrieval_tool for search/lookups/evidence.\n"
                        "Use spending_category_analyser for grouping spend by merchant or category.\n"
                        "Use financial_health_score_tool for savings rate, EMI ratio, discretionary spend, or financial health.",
                    ),
                    ("human", "{question}"),
                ]
            )
            | self._llm_loader()
            | StrOutputParser()
        )
        return self._router_chain

    def _route_by_rules(self, question: str) -> str | None:
        normalized = question.strip().lower()
        if any(hint in normalized for hint in HEALTH_HINTS):
            return "financial_health_score_tool"
        if any(hint in normalized for hint in SPENDING_HINTS):
            return "spending_category_analyser"
        if any(hint in normalized for hint in RETRIEVAL_HINTS):
            return "rag_retrieval_tool"
        return None

    def invoke(self, question: str) -> dict[str, Any]:
        in_scope, guardrail_message = self.guardrail.check(question)
        if not in_scope:
            tool_output = {"message": guardrail_message}
            return {
                "question": question,
                "selected_tool": "scope_guardrail",
                "tool_output": tool_output,
                "answer_text": build_agent_answer(question, "scope_guardrail", tool_output),
                "citations": [],
                "contexts": [],
            }

        routed_tool = self._route_by_rules(question)
        if routed_tool is None:
            router_chain = self._get_router_chain()
            if router_chain is not None:
                routed_tool = router_chain.invoke({"question": question}).strip().splitlines()[0].strip()
                routed_tool = routed_tool.lower().replace("`", "").replace('"', "").replace("'", "")
        if routed_tool not in self.tools:
            routed_tool = "rag_retrieval_tool"

        tool_result = self.tools[routed_tool].invoke({"query": question})
        try:
            parsed_output = json.loads(tool_result)
        except json.JSONDecodeError:
            parsed_output = {"raw_output": tool_result}

        return {
            "question": question,
            "selected_tool": routed_tool,
            "tool_output": parsed_output,
            "answer_text": build_agent_answer(question, routed_tool, parsed_output),
            "citations": build_citations(routed_tool, parsed_output),
            "contexts": extract_supporting_contexts(routed_tool, parsed_output),
        }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local LangChain agent for bank transaction retrieval and financial analysis."
    )
    parser.add_argument("question", help="Natural-language question for the agent.")
    parser.add_argument(
        "--persist-directory",
        type=Path,
        default=DEFAULT_CHROMA_DIR,
        help="Directory where the ChromaDB files are stored.",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION,
        help="ChromaDB collection name.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Sentence Transformers model used for retrieval.",
    )
    parser.add_argument(
        "--agent-model",
        default=DEFAULT_AGENT_MODEL,
        help="Local Hugging Face instruction model used for routing and category classification.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    llm_cache: dict[str, Any] = {}

    def llm_loader() -> ChatHuggingFace:
        if "model" not in llm_cache:
            llm_cache["model"] = build_local_chat_model(args.agent_model)
        return llm_cache["model"]

    store = TransactionStore(
        persist_directory=args.persist_directory,
        collection_name=args.collection_name,
        embedding_model_name=args.embedding_model,
    )
    financial_tools = FinancialTools(store=store, classifier=MerchantClassifier(llm_loader))
    tools = [
        financial_tools.retrieval_tool(),
        financial_tools.spending_category_tool(),
        financial_tools.financial_health_tool(),
    ]
    agent = LangChainFinanceAgent(tools=tools, llm_loader=llm_loader)
    response = agent.invoke(args.question)
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()

