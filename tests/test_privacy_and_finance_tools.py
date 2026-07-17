from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bank_langchain_agent import FinancialTools, MerchantClassifier
from bank_statement_to_chroma import parse_transactions, redact_reference, redact_sensitive_text


class FakeStore:
    def __init__(self, transactions: list[dict]) -> None:
        self.transactions = transactions

    def all_transactions(self) -> list[dict]:
        return self.transactions

    def semantic_search(self, query: str, top_k: int) -> dict:
        return {"matches": self.transactions[:1]}


def transaction(transaction_id: str, description: str, amount: str, kind: str) -> dict:
    return {
        "transaction_id": transaction_id,
        "document": description,
        "metadata": {
            "date": "2026-03-01",
            "description": description,
            "amount": amount,
            "transaction_type": kind,
        },
    }


class PrivacyTests(unittest.TestCase):
    def test_redacts_personal_identifiers_before_indexing(self) -> None:
        self.assertNotIn("9876543210", redact_sensitive_text("UPI-alex@upi-9876543210"))
        self.assertIn("REDACTED_UPI", redact_sensitive_text("UPI-alex@upi"))
        self.assertTrue(redact_reference("UTR123456789").startswith("TRANSFER_REF_"))

    def test_parser_does_not_store_raw_row_or_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "statement.csv"
            csv_path.write_text(
                "Date,Narration,Debit,Reference,Account Number\n"
                "01/03/2026,UPI-MERCHANT-alex@upi-9876543210,250.00,UTR123456789,123456789012\n",
                encoding="utf-8",
            )
            rows = parse_transactions(csv_path, None, None, None, None, None, None, None)

        self.assertEqual(len(rows), 1)
        indexed = rows[0]
        self.assertNotIn("Raw Row", indexed.document)
        self.assertNotIn("123456789012", indexed.document)
        self.assertNotIn("source_file", indexed.metadata)
        self.assertTrue(indexed.metadata["reference"].startswith("TRANSFER_REF_"))


class FinanceToolTests(unittest.TestCase):
    def test_structured_total_uses_all_matching_transactions(self) -> None:
        store = FakeStore(
            [
                transaction("1", "Coffee Corner", "-100", "debit"),
                transaction("2", "Coffee Corner", "-250", "debit"),
                transaction("3", "Coffee Corner", "-75", "debit"),
            ]
        )
        tool = FinancialTools(store, MerchantClassifier()).retrieval_tool()
        result = json.loads(tool.invoke({"query": "total spent on Coffee"}))

        self.assertEqual(result["aggregate"]["count"], 3)
        self.assertEqual(result["aggregate"]["total_amount"], "425")

    def test_spending_groups_all_debits_not_semantic_top_k(self) -> None:
        rows = [transaction(str(index), "Food cafe", "-10", "debit") for index in range(30)]
        tool = FinancialTools(FakeStore(rows), MerchantClassifier()).spending_category_tool()
        result = json.loads(tool.invoke({"query": "group my spending"}))

        self.assertEqual(sum(category["transaction_count"] for category in result["categories"]), 30)
        self.assertEqual(result["transaction_scope"], "debit transactions only")


if __name__ == "__main__":
    unittest.main()
