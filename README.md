# Bank Statement Insights

A local-first personal finance copilot for bank statement analysis.

It ingests bank statement CSVs into ChromaDB, answers finance questions with citation-backed retrieval, computes a financial health score, groups spending by merchant category, and includes a small RAGAS evaluation harness. The current version also supports smarter finance-aware filtering for natural-language questions like "large debits above 5000", "between Jan and Mar", and "total spent on a merchant".

Live app: [bankinsightsapppy-hewucidkqvbxdmstv84vyu.streamlit.app](https://bankinsightsapppy-hewucidkqvbxdmstv84vyu.streamlit.app/)

## Demo

Overview screen:

<p align="center">
  <img src="https://github.com/user-attachments/assets/564797a2-57d1-4fc8-92b1-faded103d701" alt="Bank Statement Insights app overview" width="900" />
</p>

Health dashboard:

<p align="center">
  <img src="https://github.com/user-attachments/assets/7c2169c1-10e8-45af-beda-8e8b3ffbec97" alt="Bank Statement Insights health dashboard" width="900" />
</p>

## Architecture

```mermaid
flowchart LR
    A["Bank Statement CSV"] --> B["CSV parser and normalizer<br/>bank_statement_to_chroma.py"]
    B --> C["Local embedding model<br/>all-MiniLM-L6-v2"]
    C --> D["ChromaDB persistent store<br/>data/chroma_bank_transactions"]

    E["Streamlit UI<br/>bank_insights_app.py"] --> F["LangChain finance agent<br/>bank_langchain_agent.py"]
    F --> G["Finance scope guardrail"]
    F --> H["RAG retrieval tool"]
    F --> I["Spending category analyser"]
    F --> J["Financial health score tool"]

    H --> D
    I --> D
    J --> D

    I --> K["Local instruction model<br/>Qwen2.5-0.5B-Instruct"]
    F --> K

    L["RAGAS evaluation harness<br/>evaluate_finance_agent_ragas.py"] --> F
    L --> M["Faithfulness and context relevance outputs"]
```

## Features

- Local embeddings with `sentence-transformers`
- ChromaDB-backed semantic retrieval over transactions
- Hybrid retrieval that combines embeddings with structured finance filters
- LangChain agent with three tools:
  - `rag_retrieval_tool`
  - `spending_category_analyser`
  - `financial_health_score_tool`
- Finance-scope guardrail for non-finance questions
- Streamlit two-panel UI with guided upload actions, health dashboard, and chat
- Citation-backed answers with supporting tables
- Smarter query handling for:
  - amount thresholds like `above 5000`
  - month windows like `between Jan and Mar`
  - merchant-focused totals like `total spent on Swiggy`
- Lazy model loading so the dashboard renders before the local chat model is initialized
- Faster dashboard computation via cached health summaries and heuristic-first categorization
- Prebuilt sample Chroma snapshot for lighter cloud cold starts
- Fixed 10-question RAGAS benchmark harness

## Project Structure

```text
bank-statement-insights/
|-- README.md
|-- .gitignore
|-- .env.example
|-- bank_insights_app.py
|-- bank_langchain_agent.py
|-- bank_statement_to_chroma.py
|-- query_bank_transactions.py
|-- evaluate_finance_agent_ragas.py
|-- requirements.txt
|-- requirements-eval.txt
|-- .streamlit/
|   `-- config.toml
|-- sample_data/
|   |-- sample_bank_statement.csv
|   |-- chroma_bank_transactions_snapshot/
|   |-- sample_ragas_eval.csv
|   `-- sample_ragas_summary.json
|-- docs/
|   `-- .gitkeep
|-- data/
|   `-- chroma_bank_transactions/
`-- eval/
    `-- ragas_eval_results/
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
```

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install runtime dependencies:

```powershell
pip install -r requirements.txt
```

Install RAG evaluation extras if needed:

```powershell
pip install -r requirements-eval.txt
```

Optional configuration:

```powershell
Copy-Item .env.example .env
```

```bash
cp .env.example .env
```

## Run Locally

Start the Streamlit app:

```powershell
streamlit run bank_insights_app.py
```

Useful CLI commands:

```powershell
python bank_statement_to_chroma.py "/path/to/statement.csv"
python query_bank_transactions.py "show all large UPI debits"
python query_bank_transactions.py "total spent on merchant alpha between Mar and Apr"
python bank_langchain_agent.py "what is my financial health score?"
python evaluate_finance_agent_ragas.py
```

Example app questions:

- `show all large UPI debits above 5000`
- `find debit transactions between Jan and Mar`
- `total spent on Merchant Alpha`
- `group my spending by merchant type`
- `what is my financial health score?`

## Safe Demo Data

The repo includes fictional demo-safe files:

- `sample_data/sample_bank_statement.csv`
- `sample_data/chroma_bank_transactions_snapshot/`
- `sample_data/sample_ragas_eval.csv`
- `sample_data/sample_ragas_summary.json`

The deployed app prefers the prebuilt sample Chroma snapshot on startup, which avoids regenerating embeddings during many cold starts.
If the snapshot is missing, the app can still rebuild the sample collection from the fictional CSV.
Uploaded statements are kept in isolated per-session temporary storage so private uploads do not leak into the shared public demo dataset.

## Deployment

This project is prepared for Streamlit Community Cloud.

Deployment notes:

1. Push the repo to GitHub.
2. Create a Streamlit app from `bank_insights_app.py`.
3. Redeploy when `requirements.txt` or model/runtime code changes.

Important notes:

- First chat request can still be slower than the initial page render because the local Hugging Face model is loaded lazily.
- The app now avoids loading the chat model on first paint and uses a prebuilt sample Chroma snapshot when available.
- The cloud app can still seed ChromaDB from the sample CSV if the sample collection is missing.
- `requirements.txt` pins `protobuf==3.20.3` to stay compatible with ChromaDB on Streamlit Cloud.
- Uploaded statements are now indexed in isolated per-session temporary storage instead of the shared demo Chroma collection.
- The health dashboard uses cached summaries and heuristic-first categorization to reduce rerun latency.

## Guardrail Behavior

The assistant only answers questions within personal finance scope.

Allowed examples:

- "What was my largest UPI debit?"
- "Group my spending by merchant type"
- "What is my financial health score?"

Refused examples:

- "What is the weather today?"
- "Write me a poem about space"

## RAG Evaluation

The evaluation harness measures:

- faithfulness
- context relevance

Local outputs are written to:

- `eval/ragas_eval_results/finance_agent_ragas_eval.csv`
- `eval/ragas_eval_results/finance_agent_ragas_summary.json`

Demo-safe sample outputs are committed in:

- `sample_data/sample_ragas_eval.csv`
- `sample_data/sample_ragas_summary.json`

## Product Notes

The app now has a more guided UX:

- `Try Sample Data` for instant demo setup
- `Reset Chat` to clear the conversation state
- one-click example prompts in the chat panel
- a cleaner dashboard layout with a primary score card and supporting metric cards
- citations and supporting transactions tucked into expanders to reduce clutter

The finance copilot is also more deliberate than a plain semantic-search demo:

- it uses rule-based routing before falling back to the small local LLM router
- it applies finance-aware filters for amount, month range, merchant phrases, and debit/credit type
- it surfaces applied filters in citations so the answer is easier to trust

Recent performance fixes:

- the transaction embedding model now loads only when semantic retrieval actually runs
- the local chat model now loads only when a query needs LLM routing or ambiguous classification
- dashboard health summaries are cached separately from chat state
- merchant classification now uses heuristics first and only falls back to the LLM for unclear cases

## What I Learned

- Bank exports are messy in real life, so encoding, delimiter, and header handling matter as much as retrieval quality.
- The best product feeling came from pairing semantic retrieval with deterministic finance-specific rules instead of relying entirely on an LLM.
- Deployment work mattered more than expected because ChromaDB and protobuf compatibility can break on hosted runtimes.
- Citation quality improved when the answer layer and UI both used the same structured context payload.
- The UX got much better once the app guided the user through sample data, dashboard review, and question prompts instead of dropping them into a blank interface.

## What I Would Improve Next

- Add `.xls` and `.xlsx` ingestion, not just CSV-first parsing.
- Add stronger merchant normalization for UPI-heavy statements.
- Improve retrieval filtering so queries like "large debits above 5000" combine semantic and numeric constraints.
- Add richer time expressions like `last month`, `this quarter`, and `past 30 days`.
- Add merchant summary cards like `top merchants by spend`.
- Add tests around routing, parsing, and health score calculations.
- Add a real demo GIF alongside the screenshots.

## Notes

- `sample_data/sample_bank_statement.csv` is fictional and safe for demos.
- The app is designed for personal finance analysis, not general-purpose chat.
- The local ChromaDB store lives under `data/chroma_bank_transactions`.
