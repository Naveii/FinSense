from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp
from typing import Any
from uuid import uuid4

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import chromadb
import pandas as pd
import streamlit as st

from bank_langchain_agent import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_COLLECTION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_CHROMA_DIR,
    LangChainFinanceAgent,
    FinancialTools,
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
LIVE_APP_URL = "https://bankinsightsapppy-hewucidkqvbxdmstv84vyu.streamlit.app/"
EXAMPLE_PROMPTS = [
    "Show all large UPI debits",
    "Group my spending by merchant type",
    "What is my financial health score?",
    "Find my biggest loan or EMI payments",
]


def format_currency(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "INR 0"
    sign = "-" if number < 0 else ""
    return f"{sign}INR {abs(number):,.2f}"


def format_percent(value: Any) -> str:
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "0.00%"


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


def prettify_metric_name(name: str) -> str:
    return name.replace("_", " ").title()


def format_metric_value(name: str, value: Any) -> str:
    metric_name = name.lower()
    if any(token in metric_name for token in ("income", "expenses", "savings")):
        return format_currency(value)
    if "pct" in metric_name or "ratio" in metric_name:
        return f"{value}%"
    return str(value)


def format_support_table(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table

    formatted = table.copy()
    if "Amount" in formatted.columns:
        formatted["Amount"] = formatted["Amount"].apply(format_currency)
    if "Spend Total" in formatted.columns:
        formatted["Spend Total"] = formatted["Spend Total"].apply(format_currency)
    if "Distance" in formatted.columns:
        formatted["Distance"] = formatted["Distance"].apply(
            lambda value: f"{float(value):.3f}" if value not in (None, "") else ""
        )
    if "Transaction Count" in formatted.columns:
        formatted["Transaction Count"] = formatted["Transaction Count"].astype(str)
    return formatted


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
                {"Metric": prettify_metric_name(key), "Value": format_metric_value(key, value)}
                for key, value in tool_output.get("metrics", {}).items()
            ]
        )

    return pd.DataFrame()


def ensure_default_data_loaded() -> None:
    if not SAMPLE_STATEMENT_PATH.exists():
        return
    try:
        default_client = chromadb.PersistentClient(path=str(DEFAULT_CHROMA_DIR))
        default_collection = default_client.get_collection(DEFAULT_COLLECTION)
        if default_collection.count() > 0:
            return
    except Exception:
        pass

    load_sample_dataset(replace_existing=True)


def load_sample_dataset(replace_existing: bool = False) -> str:
    if replace_existing:
        shutil.rmtree(DEFAULT_CHROMA_DIR, ignore_errors=True)

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
    get_finance_agent.clear()
    get_health_dashboard_data.clear()
    return f"Loaded {len(transactions)} sample transactions."


def ensure_session_state_defaults() -> None:
    st.session_state.setdefault("session_storage_dir", None)
    st.session_state.setdefault("session_collection_name", None)
    st.session_state.setdefault("using_session_data", False)


def reset_session_storage() -> None:
    session_storage_dir = st.session_state.get("session_storage_dir")
    if session_storage_dir:
        shutil.rmtree(session_storage_dir, ignore_errors=True)
    st.session_state.session_storage_dir = None
    st.session_state.session_collection_name = None
    st.session_state.using_session_data = False
    get_finance_agent.clear()
    get_health_dashboard_data.clear()


def get_active_storage() -> tuple[Path, str]:
    ensure_session_state_defaults()
    if st.session_state.using_session_data and st.session_state.session_storage_dir:
        return Path(st.session_state.session_storage_dir), st.session_state.session_collection_name
    return DEFAULT_CHROMA_DIR, DEFAULT_COLLECTION


def create_session_storage() -> tuple[Path, str]:
    reset_session_storage()
    session_dir = Path(mkdtemp(prefix="bank_insights_session_"))
    collection_name = f"bank_transactions_{uuid4().hex[:12]}"
    st.session_state.session_storage_dir = str(session_dir)
    st.session_state.session_collection_name = collection_name
    st.session_state.using_session_data = True
    return session_dir, collection_name


@st.cache_resource(show_spinner=False)
def get_finance_agent(
    persist_directory: str,
    collection_name: str,
) -> tuple[LangChainFinanceAgent, FinancialTools]:
    llm_cache: dict[str, Any] = {}

    def llm_loader():
        if "model" not in llm_cache:
            llm_cache["model"] = build_local_chat_model(DEFAULT_AGENT_MODEL)
        return llm_cache["model"]

    store_path = Path(persist_directory)
    try:
        store = TransactionStore(
            persist_directory=store_path,
            collection_name=collection_name,
            embedding_model_name=DEFAULT_EMBEDDING_MODEL,
        )
    except Exception:
        if store_path.resolve() != DEFAULT_CHROMA_DIR.resolve():
            raise
        load_sample_dataset(replace_existing=True)
        store = TransactionStore(
            persist_directory=store_path,
            collection_name=collection_name,
            embedding_model_name=DEFAULT_EMBEDDING_MODEL,
        )
    financial_tools = FinancialTools(store=store, classifier=MerchantClassifier(llm_loader))
    agent = LangChainFinanceAgent(
        tools=[
            financial_tools.retrieval_tool(),
            financial_tools.spending_category_tool(),
            financial_tools.financial_health_tool(),
        ],
        llm_loader=llm_loader,
    )
    return agent, financial_tools


@st.cache_data(show_spinner=False)
def get_health_dashboard_data(
    persist_directory: str,
    collection_name: str,
) -> dict[str, Any]:
    store = TransactionStore(
        persist_directory=Path(persist_directory),
        collection_name=collection_name,
        embedding_model_name=DEFAULT_EMBEDDING_MODEL,
    )
    dashboard_tools = FinancialTools(store=store, classifier=MerchantClassifier())
    health_json = dashboard_tools.financial_health_tool().invoke({"query": "dashboard"})
    return json.loads(health_json)


def ingest_uploaded_csv(uploaded_file) -> str:
    session_directory, session_collection = create_session_storage()
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
        persist_directory=session_directory,
        collection_name=session_collection,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        batch_size=100,
    )
    get_finance_agent.clear()
    get_health_dashboard_data.clear()
    return (
        f"Loaded {len(transactions)} transactions from {uploaded_file.name} into "
        "temporary session storage."
    )


def metric_card_html(
    label: str,
    value: str,
    tone: str = "default",
    subtitle: str = "",
    size: str = "standard",
) -> str:
    tone_class = f"metric-card metric-{tone} metric-size-{size}"
    subtitle_html = f"<div class='metric-subtitle'>{subtitle}</div>" if subtitle else ""
    return (
        f"<div class='{tone_class}'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div>"
        f"{subtitle_html}"
        f"</div>"
    )


def render_health_dashboard(persist_directory: str, collection_name: str) -> None:
    health_data = get_health_dashboard_data(persist_directory, collection_name)
    metrics = health_data.get("metrics", {})
    score = metrics.get("financial_health_score", "0")

    st.markdown("### Financial Health")
    st.markdown(
        """
        <div class="section-kicker">A quick snapshot of income resilience, repayment pressure, and discretionary spend.</div>
        """,
        unsafe_allow_html=True,
    )
    dashboard_html = "".join(
        [
            metric_card_html(
                "Financial Health Score",
                score,
                tone="primary",
                subtitle=health_data.get("income_assumption", ""),
                size="hero",
            ),
            metric_card_html(
                "Net Savings",
                format_currency(metrics.get("net_savings", "0")),
            ),
            metric_card_html(
                "Savings Rate",
                format_percent(metrics.get("savings_rate_pct", "0")),
            ),
            metric_card_html(
                "EMI / Income",
                format_percent(metrics.get("emi_to_income_ratio_pct", "0")),
            ),
            metric_card_html(
                "Discretionary Spend",
                format_percent(metrics.get("discretionary_spend_pct", "0")),
            ),
            metric_card_html(
                "Total Income",
                format_currency(metrics.get("total_income", "0")),
            ),
            metric_card_html(
                "Total Expenses",
                format_currency(metrics.get("total_expenses", "0")),
            ),
        ]
    )
    st.markdown(f"<div class='dashboard-grid'>{dashboard_html}</div>", unsafe_allow_html=True)

    st.subheader("Metric Breakdown")
    metrics_df = tool_output_to_dataframe(health_data)
    with st.expander("See metric breakdown", expanded=False):
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)


def run_prompt(agent: LangChainFinanceAgent, prompt: str) -> None:
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
            st.markdown("#### Summary")
            st.markdown(summary)
            table = format_support_table(tool_output_to_dataframe(response["tool_output"]))
            if citations:
                with st.expander("Evidence and citations", expanded=True):
                    for citation in citations:
                        st.caption(citation)
            if not table.empty:
                with st.expander("Supporting transactions", expanded=True):
                    st.dataframe(table, use_container_width=True, hide_index=True)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": summary,
            "citations": citations,
            "table": table,
        }
    )


def render_chat_panel(agent: LangChainFinanceAgent) -> None:
    st.markdown("### Finance Copilot")
    st.markdown(
        """
        <div class="section-kicker">Ask about merchants, large debits, spending mix, or financial health. Answers include evidence from matching transactions.</div>
        """,
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Upload a statement or use the sample dataset, then ask about large debits, merchant patterns, or your health score.",
                "citations": [],
                "table": pd.DataFrame(),
            }
        ]
    if "queued_prompt" not in st.session_state:
        st.session_state.queued_prompt = None

    st.markdown(
        """
        <div class="prompt-strip">
            <span class="prompt-chip">Citation-backed answers</span>
            <span class="prompt-chip">Merchant grouping</span>
            <span class="prompt-chip">Health score analysis</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='prompt-label'>Suggested questions</div>", unsafe_allow_html=True)
    prompt_columns = st.columns(2, gap="small")
    for index, example_prompt in enumerate(EXAMPLE_PROMPTS):
        with prompt_columns[index % 2]:
            if st.button(example_prompt, key=f"example_prompt_{index}", use_container_width=True):
                st.session_state.queued_prompt = example_prompt

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                with st.expander("Evidence and citations", expanded=False):
                    for citation in message["citations"]:
                        st.caption(citation)
            table = message.get("table")
            if isinstance(table, pd.DataFrame) and not table.empty:
                with st.expander("Supporting transactions", expanded=False):
                    st.dataframe(
                        format_support_table(table),
                        use_container_width=True,
                        hide_index=True,
                    )

    prompt = st.chat_input("Ask about your transactions")
    if not prompt and st.session_state.queued_prompt:
        prompt = st.session_state.queued_prompt
        st.session_state.queued_prompt = None

    if not prompt:
        return

    run_prompt(agent, prompt)


def main() -> None:
    if "status_message" not in st.session_state:
        st.session_state.status_message = ""
    ensure_session_state_defaults()
    ensure_default_data_loaded()

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
        :root {
            --app-bg:
                radial-gradient(circle at top left, rgba(255, 224, 178, 0.35), transparent 28%),
                linear-gradient(180deg, #f7f1e6 0%, #f4f7fb 55%, #edf3fb 100%);
            --surface-bg: rgba(255, 255, 255, 0.78);
            --surface-strong: rgba(255, 255, 255, 0.84);
            --surface-border: rgba(31, 41, 55, 0.08);
            --surface-shadow: 0 18px 44px rgba(15, 23, 42, 0.06);
            --text-strong: #20253a;
            --text-body: #1f2a44;
            --text-muted: #5b6477;
            --text-subtle: #6a7284;
            --input-bg: #ffffff;
            --input-text: #1f2a44;
            --input-muted: #7a8495;
            --button-text: #1f2a44;
            --button-text-muted: #34425d;
            --chat-shell-bg: rgba(255, 255, 255, 0.9);
            --hero-accent: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(255,243,224,0.92));
            --hero-border: rgba(245, 158, 11, 0.12);
            --chip-bg: rgba(255,255,255,0.76);
            --expander-bg: rgba(255,255,255,0.72);
            --button-bg: rgba(255,255,255,0.82);
            --accent: #159a78;
            --accent-soft: rgba(21, 154, 120, 0.12);
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --app-bg:
                    radial-gradient(circle at top left, rgba(20, 184, 166, 0.14), transparent 24%),
                    radial-gradient(circle at bottom right, rgba(59, 130, 246, 0.16), transparent 30%),
                    linear-gradient(180deg, #07111f 0%, #0b1728 52%, #122033 100%);
                --surface-bg: rgba(9, 18, 32, 0.82);
                --surface-strong: rgba(10, 20, 36, 0.9);
                --surface-border: rgba(148, 163, 184, 0.18);
                --surface-shadow: 0 22px 50px rgba(2, 6, 23, 0.4);
                --text-strong: #edf4ff;
                --text-body: #dbe7f5;
                --text-muted: #b3c0d4;
                --text-subtle: #93a4bc;
                --input-bg: rgba(12, 22, 38, 0.96);
                --input-text: #e7eefb;
                --input-muted: #97a8bd;
                --button-text: #e7eefb;
                --button-text-muted: #c8d5e6;
                --chat-shell-bg: rgba(12, 22, 38, 0.88);
                --hero-accent: linear-gradient(145deg, rgba(15,23,42,0.96), rgba(9, 32, 53, 0.94));
                --hero-border: rgba(45, 212, 191, 0.18);
                --chip-bg: rgba(15, 23, 42, 0.72);
                --expander-bg: rgba(10, 18, 32, 0.76);
                --button-bg: rgba(15, 23, 42, 0.8);
                --accent: #5dd6b2;
                --accent-soft: rgba(93, 214, 178, 0.13);
            }
        }
        html, body, .stApp, button, input, textarea, [class*="css"] {
            font-family: "DM Sans", system-ui, sans-serif;
        }
        .stApp {
            background: var(--app-bg);
        }
        header[data-testid="stHeader"] {
            background: transparent !important;
        }
        header[data-testid="stHeader"]::before {
            background: var(--app-bg) !important;
        }
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        .stDeployButton {
            color: var(--text-body) !important;
        }
        .block-container {
            max-width: 1440px;
            padding-top: 1.4rem;
            padding-bottom: 3rem;
        }
        div[data-testid="stChatMessage"] {
            background: var(--surface-bg);
            border: 1px solid var(--surface-border);
            border-radius: 20px;
            padding: 0.78rem 0.95rem;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.07);
        }
        .hero-card, .panel-card {
            background: var(--surface-bg);
            border: 1px solid var(--surface-border);
            border-radius: 22px;
            padding: 1rem 1.15rem;
            box-shadow: var(--surface-shadow);
            backdrop-filter: blur(8px);
        }
        .hero-card {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.75rem;
        }
        .hero-card h1 {
            margin: 0;
            font-size: clamp(2rem, 3.6vw, 2.85rem);
            line-height: 1.02;
            color: var(--text-strong);
            letter-spacing: -0.055em;
        }
        .hero-card p {
            margin: 0.55rem 0 0;
            color: var(--text-muted);
            font-size: 0.98rem;
            max-width: 58rem;
        }
        .hero-actions {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            gap: 0.55rem;
            flex-wrap: wrap;
            min-width: 250px;
        }
        .live-link, .status-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            border: 1px solid var(--surface-border);
            padding: 0.48rem 0.78rem;
            font-size: 0.84rem;
            font-weight: 600;
            text-decoration: none !important;
        }
        .live-link {
            background: var(--accent);
            color: #ffffff !important;
            border-color: transparent;
        }
        .status-pill {
            background: var(--accent-soft);
            color: var(--text-body);
        }
        .section-kicker {
            margin-top: -0.2rem;
            margin-bottom: 0.82rem;
            color: var(--text-muted);
            font-size: 0.93rem;
            line-height: 1.55;
        }
        h1, h2, h3, h4, h5, h6,
        div[data-testid="stMarkdownContainer"] h1,
        div[data-testid="stMarkdownContainer"] h2,
        div[data-testid="stMarkdownContainer"] h3 {
            color: var(--text-strong);
            letter-spacing: -0.02em;
        }
        .metric-card {
            background: var(--surface-strong);
            border: 1px solid var(--surface-border);
            border-radius: 18px;
            padding: 0.95rem 1rem;
            min-height: 136px;
            box-shadow: 0 9px 24px rgba(15, 23, 42, 0.065);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-width: 0;
            overflow: hidden;
        }
        .metric-primary {
            background: var(--hero-accent);
            min-height: 214px;
            border-color: var(--hero-border);
        }
        .metric-label {
            color: var(--text-muted);
            font-size: 0.86rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        .metric-value {
            color: var(--text-body);
            font-size: clamp(1.45rem, 1.9vw, 2.05rem);
            font-weight: 700;
            margin-top: 0.55rem;
            line-height: 1.02;
            letter-spacing: -0.03em;
            overflow-wrap: anywhere;
            word-break: break-word;
        }
        .metric-primary .metric-value {
            font-size: clamp(2.8rem, 4.5vw, 3.9rem);
            margin-top: 0.9rem;
            white-space: nowrap;
            overflow-wrap: normal;
            word-break: normal;
        }
        .metric-subtitle {
            margin-top: 1rem;
            color: var(--text-subtle);
            font-size: 0.96rem;
            line-height: 1.45;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.78rem;
            align-items: stretch;
            margin: 0.7rem 0 0.85rem;
        }
        .metric-size-hero {
            grid-column: 1 / -1;
        }
        @media (max-width: 1100px) {
            .dashboard-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .metric-size-hero {
                min-height: 236px;
            }
        }
        @media (max-width: 720px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
            .metric-value,
            .metric-primary .metric-value {
                white-space: normal;
            }
        }
        .prompt-strip {
            display: flex;
            gap: 0.45rem;
            flex-wrap: wrap;
            margin: 0.2rem 0 0.75rem;
        }
        .prompt-chip {
            background: var(--accent-soft);
            border: 1px solid var(--surface-border);
            border-radius: 999px;
            padding: 0.35rem 0.7rem;
            color: var(--text-body);
            font-size: 0.82rem;
            font-weight: 600;
        }
        .prompt-label {
            color: var(--text-muted);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            margin: 0.3rem 0 0.45rem;
            text-transform: uppercase;
        }
        div[data-testid="stButton"] > button {
            border-radius: 16px;
            border: 1px solid var(--surface-border);
            min-height: 2.75rem;
            background: var(--button-bg);
            color: var(--button-text);
            box-shadow: 0 7px 18px rgba(15, 23, 42, 0.055);
            font-weight: 650;
        }
        div[data-testid="stButton"] > button p,
        div[data-testid="stButton"] > button span,
        div[data-testid="stButton"] > button div[data-testid="stMarkdownContainer"] p {
            color: var(--button-text) !important;
        }
        div[data-testid="stButton"] > button:hover {
            border-color: rgba(21, 154, 120, 0.4);
            color: var(--text-strong);
            transform: translateY(-1px);
        }
        div[data-testid="stButton"] > button[kind="primary"],
        div[data-testid="stButton"] > button[kind="primary"] p,
        div[data-testid="stButton"] > button[kind="primary"] span,
        div[data-testid="stButton"] > button[kind="primary"] div[data-testid="stMarkdownContainer"] p {
            color: #ffffff !important;
        }
        div[data-testid="stFileUploader"] section {
            border-radius: 18px;
            background: var(--surface-bg);
            border: 1px solid var(--surface-border);
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.045);
        }
        div[data-testid="stFileUploaderDropzone"] {
            border-radius: 16px !important;
            border-color: rgba(21, 154, 120, 0.24) !important;
            background: var(--surface-strong) !important;
            color: var(--text-body) !important;
        }
        div[data-testid="stFileUploaderDropzone"] *,
        div[data-testid="stFileUploaderDropzone"] svg {
            color: var(--text-body) !important;
            fill: var(--text-body) !important;
        }
        div[data-testid="stFileUploaderDropzone"] button {
            background: var(--input-bg) !important;
            border: 1px solid var(--surface-border) !important;
            color: var(--button-text) !important;
            box-shadow: none !important;
        }
        div[data-testid="stFileUploaderDropzone"] button *,
        div[data-testid="stFileUploaderDropzone"] button p,
        div[data-testid="stFileUploaderDropzone"] button span {
            color: var(--button-text) !important;
        }
        div[data-testid="stFileUploaderDropzone"] small,
        div[data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] p,
        div[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] span {
            color: var(--text-muted) !important;
        }
        div[data-testid="stAlert"] {
            border-radius: 18px;
            border: 1px solid var(--surface-border);
            background: var(--surface-bg);
        }
        div[data-testid="stExpander"] {
            border-radius: 16px;
        }
        div[data-testid="stExpander"] details {
            background: var(--expander-bg);
            border: 1px solid var(--surface-border);
            border-radius: 16px;
            padding: 0.2rem 0.55rem;
        }
        div[data-testid="stExpander"] summary,
        div[data-testid="stFileUploader"] label,
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] li,
        div[data-testid="stAlertContentInfo"],
        div[data-testid="stAlertContentSuccess"] {
            color: var(--text-body);
        }
        div[data-testid="stCaptionContainer"],
        div[data-testid="stCaptionContainer"] p,
        div[data-testid="stCaptionContainer"] span {
            color: var(--text-muted) !important;
        }
        div[data-testid="stChatInput"] textarea,
        div[data-testid="stTextInput"] input {
            background: var(--input-bg) !important;
            color: var(--input-text) !important;
            border: 1px solid var(--surface-border) !important;
            border-radius: 18px !important;
        }
        div[data-testid="stChatInput"],
        div[data-testid="stChatInput"] > div,
        div[data-testid="stChatInput"] form {
            background: transparent !important;
        }
        div[data-testid="stChatInput"] > div,
        div[data-testid="stChatInput"] form {
            border: none !important;
            box-shadow: none !important;
        }
        div[data-testid="stChatInput"] div[data-baseweb="textarea"],
        div[data-testid="stChatInput"] div[data-baseweb="base-input"] {
            background: var(--chat-shell-bg) !important;
            border: 1px solid var(--surface-border) !important;
            border-radius: 18px !important;
            box-shadow: 0 8px 24px rgba(2, 6, 23, 0.18) !important;
            overflow: hidden !important;
        }
        div[data-testid="stChatInput"] div[data-baseweb="textarea"] > div,
        div[data-testid="stChatInput"] div[data-baseweb="base-input"] > div {
            background: transparent !important;
        }
        div[data-testid="stChatInput"] button {
            background: var(--button-bg) !important;
            border: 1px solid var(--surface-border) !important;
            color: var(--button-text) !important;
            border-radius: 14px !important;
        }
        div[data-testid="stChatInput"] button svg {
            color: var(--button-text) !important;
            fill: var(--button-text) !important;
        }
        div[data-testid="stChatInput"] textarea::placeholder,
        div[data-testid="stTextInput"] input::placeholder {
            color: var(--input-muted) !important;
        }
        div[data-testid="stDataFrame"] [data-testid="stTable"] {
            background: var(--surface-bg);
            border-radius: 14px;
        }
        @media (max-width: 980px) {
            .hero-card {
                align-items: flex-start;
                flex-direction: column;
            }
            .hero-actions {
                justify-content: flex-start;
                min-width: 0;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="hero-card">
            <div>
                <h1>Bank Statement Insights</h1>
                <p>Upload a statement, review financial health, and chat with a citation-backed finance copilot.</p>
            </div>
            <div class="hero-actions">
                <span class="status-pill">Private session storage</span>
                <a class="live-link" href="{LIVE_APP_URL}" target="_blank" rel="noopener noreferrer">Open live app</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([0.98, 1.02], gap="large")
    active_directory, active_collection = get_active_storage()
    agent, _ = get_finance_agent(str(active_directory), active_collection)

    with left_col:
        st.markdown("### Workspace")
        st.markdown(
            """
            <div class="section-kicker">Start with sample data or upload your own CSV. Uploaded statements stay in temporary session storage.</div>
            """,
            unsafe_allow_html=True,
        )
        action_col, reset_col = st.columns(2, gap="small")
        with action_col:
            if st.button("Try Sample Data", use_container_width=True):
                with st.spinner("Loading sample transactions..."):
                    reset_session_storage()
                    st.session_state.status_message = load_sample_dataset()
                active_directory, active_collection = get_active_storage()
                agent, _ = get_finance_agent(str(active_directory), active_collection)
        with reset_col:
            if st.button("Reset Chat", use_container_width=True):
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": "Chat reset. Ask about large debits, merchant patterns, or your financial health.",
                        "citations": [],
                        "table": pd.DataFrame(),
                    }
                ]
                st.session_state.queued_prompt = None
                st.session_state.status_message = "Chat history cleared."

        uploaded_file = st.file_uploader("Bank statement CSV", type=["csv"])
        st.caption(
            "Uploads in this app use temporary per-session storage. "
            "They do not write into the shared demo Chroma collection and are cleared when you switch back to sample data or the session ends."
        )
        if uploaded_file is not None and st.button("Process Statement", use_container_width=True):
            with st.spinner("Parsing statement, generating embeddings, and creating a temporary session index..."):
                st.session_state.status_message = ingest_uploaded_csv(uploaded_file)
            active_directory, active_collection = get_active_storage()
            agent, _ = get_finance_agent(str(active_directory), active_collection)

        if st.session_state.status_message:
            st.success(st.session_state.status_message)
        render_health_dashboard(str(active_directory), active_collection)

    with right_col:
        render_chat_panel(agent)


if __name__ == "__main__":
    main()
