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
    clear_chroma_client,
)
from bank_statement_to_chroma import parse_transactions, upsert_transactions


st.set_page_config(
    page_title="Bank Statement Insights",
    page_icon=":material/account_balance:",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_STATEMENT_PATH = PROJECT_ROOT / "sample_data" / "sample_bank_statement.csv"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
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


def load_sample_dataset(replace_existing: bool = False) -> str:
    if replace_existing:
        shutil.rmtree(DEFAULT_CHROMA_DIR, ignore_errors=True)
        clear_chroma_client(str(DEFAULT_CHROMA_DIR))

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
    st.session_state.setdefault("data_loaded", False)


def reset_session_storage() -> None:
    session_storage_dir = st.session_state.get("session_storage_dir")
    if session_storage_dir:
        clear_chroma_client(str(session_storage_dir))
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
    if uploaded_file.size > MAX_UPLOAD_BYTES:
        raise ValueError("The statement is larger than 10 MB. Please upload a smaller CSV export.")

    session_directory, session_collection = create_session_storage()
    temp_path: Path | None = None
    try:
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
        if not transactions:
            raise ValueError("No valid transaction rows were found. Check the CSV column headers and date format.")
        upsert_transactions(
            transactions=transactions,
            persist_directory=session_directory,
            collection_name=session_collection,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            batch_size=100,
        )
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

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
        f"<article class='{tone_class}'>"
        f"<div class='metric-card-top'><span class='metric-icon'></span><div class='metric-label'>{label}</div></div>"
        f"<div class='metric-value'>{value}</div>"
        f"{subtitle_html}"
        f"</article>"
    )


def render_health_dashboard(persist_directory: str, collection_name: str) -> None:
    health_data = get_health_dashboard_data(persist_directory, collection_name)
    metrics = health_data.get("metrics", {})
    score = metrics.get("financial_health_score", "0")

    st.markdown(
        """<div class="health-heading">
            <div>
                <h3>Financial Health</h3>
                <p>A quick snapshot of income resilience, repayment pressure, and discretionary spend.</p>
            </div>
            <span class="health-freshness"><i></i>Updated just now</span>
        </div>""",
        unsafe_allow_html=True,
    )
    dashboard_html = "".join(
        [
            metric_card_html(
                "Financial Health Score",
                score,
                tone="primary",
                subtitle=(
                    f"{health_data.get('income_assumption', '')}<span class='score-band'>Estimated score</span>"
                ),
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

    metrics_df = tool_output_to_dataframe(health_data)
    with st.expander("Metric Breakdown", expanded=False):
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


def render_chat_panel(agent: LangChainFinanceAgent | None, data_loaded: bool) -> None:
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

    if not data_loaded:
        st.info("Load the sample dataset or a CSV statement to unlock transaction questions and citations.")
        with st.form("finance_chat_composer", clear_on_submit=True, border=False):
            text_column, submit_column = st.columns([8, 1], gap="small")
            with text_column:
                st.text_input(
                    "Ask about your transactions",
                    placeholder="Ask about your transactions...",
                    label_visibility="collapsed",
                    disabled=True,
                )
            with submit_column:
                st.form_submit_button("Send", use_container_width=True, disabled=True)
        return

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
    for index, example_prompt in enumerate(EXAMPLE_PROMPTS):
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

    with st.form("finance_chat_composer", clear_on_submit=True, border=False):
        text_column, submit_column = st.columns([8, 1], gap="small")
        with text_column:
            prompt = st.text_input(
                "Ask about your transactions",
                placeholder="Ask about your transactions...",
                label_visibility="collapsed",
            )
        with submit_column:
            st.form_submit_button("Send", use_container_width=True)

    if not prompt and st.session_state.queued_prompt:
        prompt = st.session_state.queued_prompt
        st.session_state.queued_prompt = None

    if not prompt:
        return

    if agent is not None:
        run_prompt(agent, prompt)


def render_navigation() -> None:
    """Render a compact orientation rail without taking actions away from the workspace."""
    with st.sidebar:
        st.markdown(
            """
            <div class="rail-brand"><span class="rail-mark">BS</span><span>Bank Insights</span></div>
            <div class="rail-label">Navigation</div>
            """,
            unsafe_allow_html=True,
        )
        st.radio(
            "Navigation",
            ["Overview", "Workspace", "Finance Copilot", "Health", "Transactions"],
            key="navigation_section",
            label_visibility="collapsed",
        )
        st.markdown(
            """
            <div class="rail-bottom">
                <div class="rail-pro">Private by design</div>
                <p>Statements are indexed only for the active session.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    if "status_message" not in st.session_state:
        st.session_state.status_message = ""
    ensure_session_state_defaults()
    render_navigation()

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
        :root {
            color-scheme: light;
            --app-bg:
                radial-gradient(circle at 85% 0%, rgba(99, 91, 255, 0.09), transparent 28%),
                linear-gradient(180deg, #fbfbff 0%, #f5f6fc 100%);
            --surface-bg: rgba(255, 255, 255, 0.94);
            --surface-strong: #ffffff;
            --surface-border: #e7e9f3;
            --surface-shadow: 0 12px 30px rgba(28, 35, 70, 0.06);
            --text-strong: #11182b;
            --text-body: #202941;
            --text-muted: #687089;
            --text-subtle: #8990a4;
            --input-bg: #ffffff;
            --input-text: #202941;
            --input-muted: #8790a5;
            --button-text: #272f48;
            --button-text-muted: #59627b;
            --chat-shell-bg: #ffffff;
            --chat-wrapper-bg: #f8f9fe;
            --hero-accent: linear-gradient(135deg, #ffffff 0%, #fbfbff 100%);
            --hero-border: #e7e9f3;
            --chip-bg: #f0efff;
            --expander-bg: #fafbff;
            --button-bg: #ffffff;
            --accent: #635bff;
            --accent-soft: rgba(99, 91, 255, 0.11);
            --success: #1ca76c;
            --rail-bg: rgba(248, 249, 255, 0.88);
        }
        @media (prefers-color-scheme: dark) {
            :root {
                color-scheme: dark;
                --app-bg:
                    radial-gradient(circle at 0% 0%, rgba(42, 163, 255, 0.12), transparent 30%),
                    radial-gradient(circle at 90% 0%, rgba(122, 92, 255, 0.13), transparent 25%),
                    linear-gradient(180deg, #071426 0%, #081a30 100%);
                --surface-bg: rgba(10, 27, 49, 0.9);
                --surface-strong: #0d2039;
                --surface-border: rgba(163, 184, 216, 0.16);
                --surface-shadow: 0 18px 40px rgba(1, 8, 21, 0.34);
                --text-strong: #f4f7ff;
                --text-body: #d7e1f1;
                --text-muted: #a3b3c9;
                --text-subtle: #8294ad;
                --input-bg: #081a30;
                --input-text: #edf3ff;
                --input-muted: #93a5bd;
                --button-text: #e8effc;
                --button-text-muted: #bfcee2;
                --chat-shell-bg: #0b2039;
                --chat-wrapper-bg: #0c223d;
                --hero-accent: linear-gradient(135deg, rgba(11, 29, 53, 0.97), rgba(10, 24, 45, 0.95));
                --hero-border: rgba(136, 160, 198, 0.18);
                --chip-bg: rgba(99, 91, 255, 0.18);
                --expander-bg: rgba(8, 25, 46, 0.92);
                --button-bg: rgba(13, 32, 57, 0.94);
                --accent: #7c6cff;
                --accent-soft: rgba(124, 108, 255, 0.18);
                --success: #4fda9b;
                --rail-bg: rgba(6, 20, 38, 0.8);
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
        @media (min-width: 981px) {
            header[data-testid="stHeader"] {
                height: 0 !important;
                visibility: hidden !important;
            }
            [data-testid="stToolbar"],
            [data-testid="stDecoration"],
            [data-testid="stStatusWidget"],
            .stDeployButton {
                display: none !important;
            }
        }
        .block-container {
            max-width: 1400px;
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
        }
        section[data-testid="stSidebar"] {
            background: var(--rail-bg);
            border-right: 1px solid var(--surface-border);
            width: 13.75rem !important;
            min-width: 13.75rem !important;
        }
        section[data-testid="stSidebar"] > div:first-child {
            width: 13.75rem !important;
            padding-top: 1rem;
        }
        .rail-brand {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            color: var(--text-strong);
            font-size: 0.98rem;
            font-weight: 700;
            padding: 0.35rem 0.45rem 1.45rem;
        }
        .rail-mark {
            display: inline-grid;
            place-items: center;
            width: 2.15rem;
            height: 2.15rem;
            border-radius: 0.7rem;
            background: linear-gradient(135deg, var(--accent), #a157f5);
            color: white;
            font-size: 0.7rem;
            box-shadow: 0 8px 20px rgba(99, 91, 255, 0.28);
        }
        .rail-label {
            color: var(--text-subtle);
            font-size: 0.68rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-weight: 700;
            padding: 0 0.45rem 0.5rem;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] {
            gap: 0.3rem;
        }
        section[data-testid="stSidebar"] label[data-baseweb="radio"] {
            margin: 0 !important;
            padding: 0.58rem 0.62rem;
            border-radius: 0.65rem;
            color: var(--text-muted) !important;
            transition: background 150ms ease, color 150ms ease;
        }
        section[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child {
            display: none !important;
        }
        section[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) {
            background: var(--accent-soft);
            color: var(--accent) !important;
            border: 1px solid rgba(99, 91, 255, 0.24);
        }
        .rail-bottom {
            position: fixed;
            bottom: 1.2rem;
            width: 12.55rem;
            margin: 0 0.35rem;
            padding: 0.8rem;
            border-radius: 0.8rem;
            border: 1px solid var(--surface-border);
            background: var(--surface-bg);
        }
        .rail-pro {
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 700;
        }
        .rail-bottom p {
            margin: 0.3rem 0 0;
            color: var(--text-muted);
            font-size: 0.72rem;
            line-height: 1.45;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--surface-bg);
            border-color: var(--surface-border);
            border-radius: 1rem;
            box-shadow: var(--surface-shadow);
        }
        div[data-testid="stVerticalBlockBorderWrapper"] > div {
            padding: 1.05rem 1.1rem;
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
            margin-bottom: 0.9rem;
            min-height: 6.6rem;
        }
        .hero-card h1 {
            margin: 0;
            font-size: clamp(1.85rem, 3.2vw, 2.55rem);
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
            font-size: 0.88rem;
            line-height: 1.55;
        }
        .health-heading {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.85rem;
        }
        .health-heading h3 {
            margin: 0;
        }
        .health-heading p {
            margin: 0.38rem 0 0;
            color: var(--text-muted);
            font-size: 0.88rem;
            line-height: 1.55;
        }
        .health-freshness {
            display: inline-flex;
            align-items: center;
            gap: 0.38rem;
            color: var(--text-muted);
            font-size: 0.72rem;
            white-space: nowrap;
            padding-top: 0.3rem;
        }
        .health-freshness i {
            width: 0.42rem;
            height: 0.42rem;
            border-radius: 50%;
            background: var(--success);
            box-shadow: 0 0 0 3px rgba(28, 167, 108, 0.1);
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
            border-radius: 0.9rem;
            padding: 0.9rem 0.95rem;
            min-height: 112px;
            box-shadow: none;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-width: 0;
            overflow: hidden;
        }
        .metric-primary {
            background: var(--hero-accent);
            min-height: 250px;
            border-color: var(--hero-border);
        }
        .metric-card-top {
            display: flex;
            align-items: center;
            gap: 0.55rem;
        }
        .metric-icon {
            width: 1.65rem;
            height: 1.65rem;
            border-radius: 0.52rem;
            background: var(--accent-soft);
            border: 1px solid rgba(99, 91, 255, 0.18);
            color: var(--accent);
            position: relative;
            flex: 0 0 auto;
        }
        .metric-icon::after {
            content: "";
            position: absolute;
            width: 0.55rem;
            height: 0.36rem;
            border: 1.5px solid currentColor;
            border-radius: 0.12rem;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
        }
        .metric-primary .metric-icon {
            background: rgba(28, 167, 108, 0.12);
            border-color: rgba(28, 167, 108, 0.18);
        }
        .metric-primary .metric-icon::after {
            color: var(--success);
        }
        .metric-card:nth-child(2) .metric-icon { color: #477bff; background: rgba(71, 123, 255, 0.13); border-color: rgba(71, 123, 255, 0.18); }
        .metric-card:nth-child(3) .metric-icon { color: #7a5cff; background: rgba(122, 92, 255, 0.13); border-color: rgba(122, 92, 255, 0.18); }
        .metric-card:nth-child(4) .metric-icon { color: #ec9a36; background: rgba(236, 154, 54, 0.13); border-color: rgba(236, 154, 54, 0.18); }
        .metric-card:nth-child(5) .metric-icon { color: #e85479; background: rgba(232, 84, 121, 0.13); border-color: rgba(232, 84, 121, 0.18); }
        .metric-card:nth-child(6) .metric-icon { color: #28ad75; background: rgba(40, 173, 117, 0.13); border-color: rgba(40, 173, 117, 0.18); }
        .metric-card:nth-child(7) .metric-icon { color: #db5477; background: rgba(219, 84, 119, 0.13); border-color: rgba(219, 84, 119, 0.18); }
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
            font-size: clamp(3rem, 5.1vw, 4.7rem);
            margin-top: 0.9rem;
            color: var(--success);
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
        .score-band {
            display: block;
            width: fit-content;
            margin-top: 0.7rem;
            padding: 0.25rem 0.5rem;
            border-radius: 999px;
            background: rgba(28, 167, 108, 0.12);
            color: var(--success);
            font-size: 0.72rem;
            font-weight: 700;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: minmax(13rem, 1.15fr) repeat(2, minmax(10rem, 1fr));
            gap: 0.7rem;
            align-items: stretch;
            margin: 0.7rem 0 0.85rem;
        }
        .metric-size-hero {
            grid-row: span 3;
        }
        @media (max-width: 1100px) {
            .dashboard-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .metric-size-hero {
                grid-row: auto;
                grid-column: 1 / -1;
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
            color: var(--accent);
            font-size: 0.72rem;
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
            border-radius: 0.65rem;
            border: 1px solid var(--surface-border);
            min-height: 2.55rem;
            background: var(--button-bg);
            color: var(--button-text);
            box-shadow: none;
            font-weight: 650;
        }
        div[data-testid="stButton"] > button p,
        div[data-testid="stButton"] > button span,
        div[data-testid="stButton"] > button div[data-testid="stMarkdownContainer"] p {
            color: var(--button-text) !important;
        }
        div[data-testid="stButton"] > button:hover {
            border-color: rgba(99, 91, 255, 0.42);
            color: var(--text-strong);
            transform: translateY(-1px);
        }
        div[data-testid="stButton"] > button[kind="primary"] {
            background: linear-gradient(135deg, var(--accent), #a157f5) !important;
            border-color: transparent !important;
            box-shadow: 0 8px 18px rgba(99, 91, 255, 0.23) !important;
        }
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
            border-radius: 0.75rem !important;
            border: 1px dashed rgba(99, 91, 255, 0.32) !important;
            background: var(--chat-wrapper-bg) !important;
            color: var(--text-body) !important;
            min-height: 7.35rem !important;
            padding: 1.15rem 0.75rem !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        div[data-testid="stFileUploaderDropzone"] > div {
            width: 100% !important;
            text-align: center !important;
        }
        div[data-testid="stFileUploaderDropzone"] *,
        div[data-testid="stFileUploaderDropzone"] svg {
            color: var(--text-body) !important;
            fill: var(--text-body) !important;
        }
        div[data-testid="stFileUploaderDropzone"] button {
            background: linear-gradient(135deg, var(--accent), #a157f5) !important;
            border: 1px solid transparent !important;
            color: #ffffff !important;
            box-shadow: none !important;
            font-weight: 700 !important;
            opacity: 1 !important;
        }
        div[data-testid="stFileUploaderDropzone"] button *,
        div[data-testid="stFileUploaderDropzone"] button p,
        div[data-testid="stFileUploaderDropzone"] button span {
            color: #ffffff !important;
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
        div[data-testid="stChatInput"] {
            background: transparent !important;
            border: none !important;
            border-radius: 0 !important;
            padding: 0 !important;
            box-shadow: none !important;
        }
        div[data-testid="stChatInput"] form {
            background: var(--chat-wrapper-bg) !important;
            border: 1px solid var(--surface-border) !important;
            border-radius: 1rem !important;
            padding: 0.52rem 0.6rem !important;
            box-shadow: 0 10px 28px rgba(2, 6, 23, 0.18) !important;
        }
        div[data-testid="stChatInput"] [data-baseweb="textarea"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        div[data-testid="stChatInput"] div[data-baseweb="textarea"],
        div[data-testid="stChatInput"] div[data-baseweb="base-input"] {
            background: var(--input-bg) !important;
            border: 1px solid var(--surface-border) !important;
            border-radius: 14px !important;
            box-shadow: none !important;
            overflow: hidden !important;
        }
        div[data-testid="stChatInput"] div[data-baseweb="textarea"] > div,
        div[data-testid="stChatInput"] div[data-baseweb="base-input"] > div {
            background: transparent !important;
        }
        div[data-testid="stChatInput"] button {
            background: linear-gradient(135deg, var(--accent), #a157f5) !important;
            border: 1px solid transparent !important;
            color: #ffffff !important;
            border-radius: 50% !important;
            min-height: 2.4rem !important;
            min-width: 2.4rem !important;
        }
        div[data-testid="stChatInput"] button svg {
            color: #ffffff !important;
            fill: #ffffff !important;
        }
        div[data-testid="stChatInput"] textarea::placeholder,
        div[data-testid="stTextInput"] input::placeholder {
            color: var(--input-muted) !important;
        }
        div[data-testid="stForm"] {
            background: var(--chat-wrapper-bg) !important;
            border: 1px solid var(--surface-border) !important;
            border-radius: 0.9rem !important;
            padding: 0.42rem 0.5rem !important;
            box-shadow: 0 8px 22px rgba(2, 6, 23, 0.12) !important;
        }
        div[data-testid="stForm"] div[data-testid="stTextInput"] {
            margin-bottom: 0 !important;
        }
        div[data-testid="stForm"] div[data-testid="stTextInput"] input {
            min-height: 2.45rem !important;
            padding: 0.55rem 0.75rem !important;
            border-radius: 0.72rem !important;
        }
        div[data-testid="stForm"] div[data-testid="stFormSubmitButton"] {
            margin: 0 !important;
        }
        div[data-testid="stForm"] div[data-testid="stFormSubmitButton"] > button {
            width: 2.45rem !important;
            min-width: 2.45rem !important;
            height: 2.45rem !important;
            min-height: 2.45rem !important;
            padding: 0 !important;
            border-radius: 50% !important;
            background: linear-gradient(135deg, var(--accent), #a157f5) !important;
            border-color: transparent !important;
            box-shadow: 0 6px 15px rgba(99, 91, 255, 0.24) !important;
        }
        div[data-testid="stForm"] div[data-testid="stFormSubmitButton"] > button p {
            font-size: 0 !important;
        }
        div[data-testid="stForm"] div[data-testid="stFormSubmitButton"] > button p::after {
            content: "\2191";
            color: #ffffff;
            font-size: 1.2rem;
            font-weight: 500;
            line-height: 1;
        }
        div[data-testid="stForm"] div[data-testid="stFormSubmitButton"] > button:disabled {
            opacity: 0.45 !important;
            box-shadow: none !important;
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
            .rail-bottom {
                display: none;
            }
            .health-heading {
                display: block;
            }
            .health-freshness {
                margin-top: 0.35rem;
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
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([0.98, 1.02], gap="large")
    active_directory, active_collection = get_active_storage()
    data_loaded = st.session_state.data_loaded
    agent: LangChainFinanceAgent | None = None
    if data_loaded:
        agent, _ = get_finance_agent(str(active_directory), active_collection)

    with left_col:
        with st.container(border=True):
            st.markdown("### Workspace")
            st.markdown(
                """
                <div class="section-kicker">Start with sample data or upload your own CSV. Uploaded statements stay in temporary session storage.</div>
                """,
                unsafe_allow_html=True,
            )
            action_col, reset_col = st.columns(2, gap="small")
            with action_col:
                if st.button("Try Sample Data", use_container_width=True, type="primary"):
                    with st.spinner("Loading sample transactions..."):
                        reset_session_storage()
                        st.session_state.status_message = load_sample_dataset(replace_existing=True)
                        st.session_state.data_loaded = True
                    active_directory, active_collection = get_active_storage()
                    agent, _ = get_finance_agent(str(active_directory), active_collection)
                    data_loaded = True
            with reset_col:
                if st.button("Reset Chat", use_container_width=True, disabled=not data_loaded):
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
                "Uploads use temporary per-session storage and are cleared when the session ends or you return to sample data."
            )
            if uploaded_file is not None and st.button("Process Statement", use_container_width=True, type="primary"):
                try:
                    with st.spinner("Parsing statement, generating embeddings, and creating a temporary session index..."):
                        st.session_state.status_message = ingest_uploaded_csv(uploaded_file)
                        st.session_state.data_loaded = True
                    active_directory, active_collection = get_active_storage()
                    agent, _ = get_finance_agent(str(active_directory), active_collection)
                    data_loaded = True
                except ValueError as error:
                    reset_session_storage()
                    st.session_state.data_loaded = False
                    st.session_state.status_message = ""
                    st.error(str(error))
                except Exception:
                    reset_session_storage()
                    st.session_state.data_loaded = False
                    st.session_state.status_message = ""
                    st.error("The statement could not be processed. Confirm it is a CSV bank export and try again.")

            if st.session_state.status_message:
                st.success(st.session_state.status_message)

    with right_col:
        with st.container(border=True):
            render_chat_panel(agent, data_loaded)

    with st.container(border=True):
        if data_loaded:
            render_health_dashboard(str(active_directory), active_collection)
        else:
            st.markdown("### Financial Health")
            st.caption("Load a statement to calculate an estimated health score and spending breakdown.")


if __name__ == "__main__":
    main()
