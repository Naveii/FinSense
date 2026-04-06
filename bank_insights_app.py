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


def load_sample_dataset() -> str:
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
    return f"Loaded {len(transactions)} sample transactions."


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


def render_health_dashboard(financial_tools: FinancialTools) -> None:
    health_json = financial_tools.financial_health_tool().invoke({"query": "dashboard"})
    health_data = json.loads(health_json)
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
                size="compact",
            ),
            metric_card_html(
                "Savings Rate",
                format_percent(metrics.get("savings_rate_pct", "0")),
                size="compact",
            ),
            metric_card_html(
                "EMI / Income",
                format_percent(metrics.get("emi_to_income_ratio_pct", "0")),
                size="compact",
            ),
            metric_card_html(
                "Discretionary Spend",
                format_percent(metrics.get("discretionary_spend_pct", "0")),
                size="compact",
            ),
            metric_card_html(
                "Total Income",
                format_currency(metrics.get("total_income", "0")),
                size="wide",
            ),
            metric_card_html(
                "Total Expenses",
                format_currency(metrics.get("total_expenses", "0")),
                size="wide",
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
        <div class="section-kicker">Ask natural questions about merchants, large debits, spending mix, or overall financial health.</div>
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

    prompt_columns = st.columns(len(EXAMPLE_PROMPTS))
    for index, example_prompt in enumerate(EXAMPLE_PROMPTS):
        with prompt_columns[index]:
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

    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 224, 178, 0.35), transparent 28%),
                linear-gradient(180deg, #f7f1e6 0%, #f4f7fb 55%, #edf3fb 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        div[data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.76);
            border: 1px solid rgba(31, 41, 55, 0.08);
            border-radius: 18px;
            padding: 0.7rem 0.9rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.05);
        }
        .hero-card, .panel-card {
            background: rgba(255, 255, 255, 0.74);
            border: 1px solid rgba(31, 41, 55, 0.07);
            border-radius: 24px;
            padding: 1rem 1.1rem;
            box-shadow: 0 18px 44px rgba(15, 23, 42, 0.06);
            backdrop-filter: blur(8px);
        }
        .hero-card h1 {
            margin: 0;
            font-size: 3.3rem;
            line-height: 1.02;
            color: #20253a;
        }
        .hero-card p {
            margin: 0.65rem 0 0;
            color: #5b6477;
            font-size: 1rem;
        }
        .section-kicker {
            margin-top: -0.2rem;
            margin-bottom: 0.8rem;
            color: #6a7284;
            font-size: 0.96rem;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.84);
            border: 1px solid rgba(31, 41, 55, 0.06);
            border-radius: 22px;
            padding: 1rem 1.05rem;
            min-height: 150px;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.04);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-width: 0;
            overflow: hidden;
        }
        .metric-primary {
            background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(255,243,224,0.92));
            min-height: 312px;
        }
        .metric-label {
            color: #5b6477;
            font-size: 0.92rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        .metric-value {
            color: #1f2a44;
            font-size: clamp(1.55rem, 2.15vw, 2.25rem);
            font-weight: 700;
            margin-top: 0.55rem;
            line-height: 1.02;
            letter-spacing: -0.03em;
            overflow-wrap: anywhere;
            word-break: break-word;
        }
        .metric-primary .metric-value {
            font-size: clamp(3.9rem, 6vw, 5.1rem);
            margin-top: 1.2rem;
            white-space: nowrap;
            overflow-wrap: normal;
            word-break: normal;
        }
        .metric-size-compact .metric-value {
            font-size: clamp(1.3rem, 1.75vw, 1.95rem);
        }
        .metric-size-wide .metric-value {
            font-size: clamp(1.75rem, 2.2vw, 2.45rem);
        }
        .metric-subtitle {
            margin-top: 1rem;
            color: #6a7284;
            font-size: 0.96rem;
            line-height: 1.45;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.45fr) repeat(2, minmax(0, 1fr));
            gap: 1rem;
            align-items: stretch;
            margin: 0.8rem 0 1rem;
        }
        .metric-size-hero {
            grid-row: span 2;
        }
        .metric-size-wide {
            min-height: 138px;
        }
        @media (max-width: 1100px) {
            .dashboard-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .metric-size-hero {
                grid-column: 1 / -1;
                grid-row: auto;
                min-height: 240px;
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
            margin: 0.2rem 0 1rem;
        }
        .prompt-chip {
            background: rgba(255,255,255,0.76);
            border: 1px solid rgba(31,41,55,0.08);
            border-radius: 999px;
            padding: 0.35rem 0.7rem;
            color: #4c5567;
            font-size: 0.88rem;
        }
        div[data-testid="stButton"] > button {
            border-radius: 999px;
            border: 1px solid rgba(31, 41, 55, 0.08);
            min-height: 2.6rem;
        }
        div[data-testid="stFileUploader"] section {
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.72);
        }
        div[data-testid="stExpander"] {
            border-radius: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="hero-card">
            <h1>Bank Statement Insights</h1>
            <p>Upload a statement, review your financial health, and chat with a finance copilot that answers with evidence from your transactions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"[Open live app]({LIVE_APP_URL})")

    left_col, right_col = st.columns([0.92, 1.08], gap="large")
    agent, financial_tools = get_finance_agent()

    with left_col:
        st.markdown("### Workspace")
        st.markdown(
            """
            <div class="section-kicker">Start with a sample statement or upload your own CSV, then review the dashboard before asking questions.</div>
            """,
            unsafe_allow_html=True,
        )
        action_col, reset_col = st.columns(2, gap="small")
        with action_col:
            if st.button("Try Sample Data", use_container_width=True):
                with st.spinner("Loading sample transactions..."):
                    st.session_state.status_message = load_sample_dataset()
                agent, financial_tools = get_finance_agent()
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
        if uploaded_file is not None and st.button("Process Statement", use_container_width=True):
            with st.spinner("Parsing statement, generating embeddings, and updating ChromaDB..."):
                st.session_state.status_message = ingest_uploaded_csv(uploaded_file)
            agent, financial_tools = get_finance_agent()

        if st.session_state.status_message:
            st.success(st.session_state.status_message)
        render_health_dashboard(financial_tools)

    with right_col:
        render_chat_panel(agent)


if __name__ == "__main__":
    main()
