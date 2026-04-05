from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import pandas as pd
import streamlit as st

from bank_langchain_agent import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_COLLECTION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_CHROMA_DIR,
    FinancialTools,
    LangChainFinanceAgent,
    MerchantClassifier,
    TransactionStore,
    build_local_chat_model,
)
from bank_statement_to_chroma import parse_transactions, upsert_transactions


st.set_page_config(
    page_title="Bank Statement Insights",
    page_icon=":material/account_balance:",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_STATEMENT_PATH = PROJECT_ROOT / "sample_data" / "sample_bank_statement.csv"


def format_currency(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "INR 0"
    sign = "-" if number < 0 else ""
    return f"{sign}INR {abs(number):,.2f}"


def format_month_range(dates: list[str]) -> str:
    parsed = []
    for value in dates:
        try:
            parsed.append(datetime.strptime(value, "%Y-%m-%d"))
        except ValueError:
            continue
    if not parsed:
        return "unknown period"
    parsed.sort()
    start = parsed[0]
    end = parsed[-1]
    if start.year == end.year and start.month == end.month:
        return start.strftime("%b %Y")
    return f"{start.strftime('%b %Y')} to {end.strftime('%b %Y')}"


def merchant_hint(description: str) -> str:
    if not description:
        return "matched merchants"
    parts = description.split("-")
    if len(parts) > 1:
        for part in parts[1:]:
            cleaned = part.strip()
            if cleaned and "@" not in cleaned and cleaned.upper() != cleaned[: len(cleaned)]:
                return cleaned.title()
        return parts[1].strip().title()
    return description[:40].title()


def build_citations(selected_tool: str, tool_output: dict[str, Any]) -> list[str]:
    citations: list[str] = []

    if selected_tool == "rag_retrieval_tool":
        matches = tool_output.get("matches", [])
        if matches:
            descriptions = [
                match.get("metadata", {}).get("description", "")
                for match in matches
                if match.get("metadata")
            ]
            merchant = merchant_hint(descriptions[0]) if descriptions else "matched merchants"
            dates = [
                match.get("metadata", {}).get("date", "")
                for match in matches
                if match.get("metadata")
            ]
            citations.append(
                f"Based on {len(matches)} retrieved transactions related to {merchant} across {format_month_range(dates)}."
            )
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
            dates = [transaction.get("date", "") for transaction in transactions]
            citations.append(
                f"Based on {len(transactions)} transactions in category '{top_category.get('category', 'unknown')}' across {format_month_range(dates)}."
            )
            citations.append(
                f"Observed spend in that category: {format_currency(top_category.get('spend_total'))}."
            )

    elif selected_tool == "financial_health_score_tool":
        metrics = tool_output.get("metrics", {})
        categorized_transactions = tool_output.get("categorized_transactions", [])
        dates = [transaction.get("date", "") for transaction in categorized_transactions]
        citations.append(
            f"Based on {len(categorized_transactions)} classified transactions across {format_month_range(dates)}."
        )
        citations.append(
            "Income proxy: "
            + tool_output.get("income_assumption", "No assumption text available.")
        )
        citations.append(
            f"Savings rate {metrics.get('savings_rate_pct', '0')}%, EMI-to-income {metrics.get('emi_to_income_ratio_pct', '0')}%, discretionary spend {metrics.get('discretionary_spend_pct', '0')}%."
        )

    return citations


def generate_chat_answer(selected_tool: str, tool_output: dict[str, Any]) -> str:
    if selected_tool == "rag_retrieval_tool":
        matches = tool_output.get("matches", [])
        if not matches:
            return "I could not find matching transactions for that question."
        top_match = matches[0]
        metadata = top_match.get("metadata", {})
        return (
            f"I found {len(matches)} relevant transactions. "
            f"The strongest match is {metadata.get('description', 'a transaction')} on "
            f"{metadata.get('date', 'an unknown date')} for {format_currency(metadata.get('amount'))}."
        )

    if selected_tool == "spending_category_analyser":
        categories = tool_output.get("categories", [])
        if not categories:
            return "I could not group the transactions into merchant categories."
        top_category = categories[0]
        return (
            f"Your heaviest category in this result set is `{top_category.get('category', 'unknown')}`, "
            f"with {top_category.get('transaction_count', 0)} transactions and "
            f"{format_currency(top_category.get('spend_total', '0'))} in spend."
        )

    if selected_tool == "financial_health_score_tool":
        metrics = tool_output.get("metrics", {})
        return (
            f"Your financial health score is {metrics.get('financial_health_score', '0')}. "
            f"Savings rate is {metrics.get('savings_rate_pct', '0')}%, "
            f"EMI-to-income ratio is {metrics.get('emi_to_income_ratio_pct', '0')}%, "
            f"and discretionary spend is {metrics.get('discretionary_spend_pct', '0')}%."
        )

    return "I processed the question but could not summarize the result cleanly."


def tool_output_to_dataframe(tool_output: dict[str, Any]) -> pd.DataFrame:
    if "matches" in tool_output:
        rows = []
        for match in tool_output.get("matches", []):
            metadata = match.get("metadata", {})
            rows.append(
                {
                    "Date": metadata.get("date"),
                    "Description": metadata.get("description"),
                    "Amount": metadata.get("amount"),
                    "Type": metadata.get("transaction_type"),
                    "Distance": match.get("distance"),
                }
            )
        return pd.DataFrame(rows)

    if "categories" in tool_output:
        rows = []
        for category in tool_output.get("categories", []):
            rows.append(
                {
                    "Category": category.get("category"),
                    "Spend Total": category.get("spend_total"),
                    "Transaction Count": category.get("transaction_count"),
                }
            )
        return pd.DataFrame(rows)

    if "metrics" in tool_output:
        return pd.DataFrame(
            [
                {"Metric": key, "Value": value}
                for key, value in tool_output.get("metrics", {}).items()
            ]
        )

    return pd.DataFrame()


def ensure_default_data_loaded() -> None:
    if not SAMPLE_STATEMENT_PATH.exists():
        return

    try:
        store = TransactionStore(
            persist_directory=DEFAULT_CHROMA_DIR,
            collection_name=DEFAULT_COLLECTION,
            embedding_model_name=DEFAULT_EMBEDDING_MODEL,
        )
        if store.collection.count() > 0:
            return
    except Exception:
        pass

    transactions = parse_transactions(
        csv_path=SAMPLE_STATEMENT_PATH,
        date_column=None,
        description_column=None,
        debit_column=None,
        credit_column=None,
        amount_column=None,
        balance_column=None,
        reference_column=None,
    )
    upsert_transactions(
        transactions=transactions,
        persist_directory=DEFAULT_CHROMA_DIR,
        collection_name=DEFAULT_COLLECTION,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        batch_size=100,
    )


@st.cache_resource(show_spinner=False)
def get_finance_agent() -> tuple[LangChainFinanceAgent, FinancialTools]:
    ensure_default_data_loaded()
    llm = build_local_chat_model(DEFAULT_AGENT_MODEL)
    store = TransactionStore(
        persist_directory=DEFAULT_CHROMA_DIR,
        collection_name=DEFAULT_COLLECTION,
        embedding_model_name=DEFAULT_EMBEDDING_MODEL,
    )
    financial_tools = FinancialTools(store=store, classifier=MerchantClassifier(llm))
    agent = LangChainFinanceAgent(
        tools=[
            financial_tools.retrieval_tool(),
            financial_tools.spending_category_tool(),
            financial_tools.financial_health_tool(),
        ],
        llm=llm,
    )
    return agent, financial_tools


def ingest_uploaded_csv(uploaded_file) -> str:
    with NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = Path(temp_file.name)

    transactions = parse_transactions(
        csv_path=temp_path,
        date_column=None,
        description_column=None,
        debit_column=None,
        credit_column=None,
        amount_column=None,
        balance_column=None,
        reference_column=None,
    )
    upsert_transactions(
        transactions=transactions,
        persist_directory=DEFAULT_CHROMA_DIR,
        collection_name=DEFAULT_COLLECTION,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        batch_size=100,
    )
    get_finance_agent.clear()
    return f"Loaded {len(transactions)} transactions from {uploaded_file.name}."


def render_health_dashboard(financial_tools: FinancialTools) -> None:
    health_json = financial_tools.financial_health_tool().invoke({"query": "dashboard"})
    health_data = json.loads(health_json)
    metrics = health_data.get("metrics", {})

    st.subheader("Health Score")
    score_col, savings_col = st.columns(2)
    with score_col:
        st.metric("Financial Health Score", metrics.get("financial_health_score", "0"))
        st.caption(health_data.get("income_assumption", ""))
    with savings_col:
        st.metric("Net Savings", format_currency(metrics.get("net_savings", "0")))
        st.metric("Savings Rate", f"{metrics.get('savings_rate_pct', '0')}%")

    emi_col, discretionary_col, income_col = st.columns(3)
    with emi_col:
        st.metric("EMI / Income", f"{metrics.get('emi_to_income_ratio_pct', '0')}%")
    with discretionary_col:
        st.metric(
            "Discretionary Spend",
            f"{metrics.get('discretionary_spend_pct', '0')}%",
        )
    with income_col:
        st.metric("Total Income", format_currency(metrics.get("total_income", "0")))

    st.subheader("Metric Breakdown")
    metrics_df = tool_output_to_dataframe(health_data)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)


def render_chat_panel(agent: LangChainFinanceAgent) -> None:
    st.subheader("Chat")
    st.caption("Ask about spending patterns, merchant categories, or financial health.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Ask a question like 'show all large UPI debits' or 'what is my financial health score'.",
                "citations": [],
                "table": pd.DataFrame(),
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                st.markdown("**Citations**")
                for citation in message["citations"]:
                    st.caption(citation)
            table = message.get("table")
            if isinstance(table, pd.DataFrame) and not table.empty:
                st.dataframe(table, use_container_width=True, hide_index=True)

    prompt = st.chat_input("Ask about your transactions")
    if not prompt:
        return

    st.session_state.messages.append(
        {"role": "user", "content": prompt, "citations": [], "table": pd.DataFrame()}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analysing transactions..."):
            response = agent.invoke(prompt)
            citations = response.get(
                "citations",
                build_citations(response["selected_tool"], response["tool_output"]),
            )
            summary = response.get(
                "answer_text",
                generate_chat_answer(response["selected_tool"], response["tool_output"]),
            )
            st.markdown(summary)
            if citations:
                st.markdown("**Citations**")
                for citation in citations:
                    st.caption(citation)
            table = tool_output_to_dataframe(response["tool_output"])
            if not table.empty:
                st.dataframe(table, use_container_width=True, hide_index=True)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": summary,
            "citations": citations,
            "table": table,
        }
    )


def main() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f8f2e7 0%, #f4f7fb 100%);
        }
        div[data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.76);
            border: 1px solid rgba(31, 41, 55, 0.08);
            border-radius: 18px;
            padding: 0.55rem 0.8rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(31, 41, 55, 0.06);
            border-radius: 16px;
            padding: 0.5rem 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Bank Statement Insights")
    st.caption("Upload a CSV, review your financial health, and chat with citation-backed transaction answers.")

    left_col, right_col = st.columns([0.92, 1.08], gap="large")
    agent, financial_tools = get_finance_agent()

    with left_col:
        st.subheader("Upload CSV")
        uploaded_file = st.file_uploader("Bank statement CSV", type=["csv"])
        if uploaded_file is not None:
            if st.button("Process Statement", use_container_width=True):
                with st.spinner("Embedding transactions and updating ChromaDB..."):
                    message = ingest_uploaded_csv(uploaded_file)
                st.success(message)
                agent, financial_tools = get_finance_agent()

        render_health_dashboard(financial_tools)

    with right_col:
        render_chat_panel(agent)


if __name__ == "__main__":
    main()
