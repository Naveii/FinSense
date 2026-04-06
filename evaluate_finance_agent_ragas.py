from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference

from bank_langchain_agent import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_COLLECTION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_CHROMA_DIR,
    PROJECT_ROOT,
    FinancialTools,
    LangChainFinanceAgent,
    MerchantClassifier,
    TransactionStore,
    build_local_chat_model,
)


TEST_CASES = [
    {
        "question": "What was the largest UPI debit in the statement?",
        "reference": "The largest UPI debit in the statement was a payment to Merchant Alpha on 2026-04-01.",
    },
    {
        "question": "How many debit transactions are in the statement?",
        "reference": "There are 6 debit transactions in the statement.",
    },
    {
        "question": "How many credit transactions are in the statement?",
        "reference": "There are 5 credit transactions in the statement.",
    },
    {
        "question": "What was the highest credit transaction?",
        "reference": "The highest credit transaction was a transfer from Contact Alpha on 2026-04-01.",
    },
    {
        "question": "What is the total credited amount in the statement period?",
        "reference": "The total credited amount in the statement period was a little above INR 20,000.",
    },
    {
        "question": "How much did I pay to Merchant Beta?",
        "reference": "You made a large payment to Merchant Beta on 2026-04-05.",
    },
    {
        "question": "How much did I transfer to Contact Beta?",
        "reference": "You transferred money to Contact Beta on 2026-04-05.",
    },
    {
        "question": "What is my financial health score?",
        "reference": "Your financial health score is low because expenses are much higher than income in this sample period.",
    },
    {
        "question": "What was my net savings in this statement period?",
        "reference": "Your net savings in this statement period were negative.",
    },
    {
        "question": "What were my total expenses in this statement period?",
        "reference": "Your total expenses in this statement period were above INR 100,000.",
    },
]


def build_agent() -> LangChainFinanceAgent:
    llm_cache: dict[str, Any] = {}

    def llm_loader():
        if "model" not in llm_cache:
            llm_cache["model"] = build_local_chat_model(DEFAULT_AGENT_MODEL)
        return llm_cache["model"]

    store = TransactionStore(
        persist_directory=DEFAULT_CHROMA_DIR,
        collection_name=DEFAULT_COLLECTION,
        embedding_model_name=DEFAULT_EMBEDDING_MODEL,
    )
    financial_tools = FinancialTools(store=store, classifier=MerchantClassifier(llm_loader))
    return LangChainFinanceAgent(
        tools=[
            financial_tools.retrieval_tool(),
            financial_tools.spending_category_tool(),
            financial_tools.financial_health_tool(),
        ],
        llm_loader=llm_loader,
    )


def load_dataset(agent: LangChainFinanceAgent) -> tuple[EvaluationDataset, list[dict[str, str]]]:
    samples = []
    raw_rows = []
    for case in TEST_CASES:
        response = agent.invoke(case["question"])
        sample = SingleTurnSample(
            user_input=case["question"],
            response=response.get("answer_text", ""),
            reference=case["reference"],
            retrieved_contexts=response.get("contexts", []),
        )
        samples.append(sample)
        raw_rows.append(
            {
                "question": case["question"],
                "selected_tool": response.get("selected_tool", ""),
                "answer_text": response.get("answer_text", ""),
                "reference": case["reference"],
                "citations": " | ".join(response.get("citations", [])),
            }
        )
    return EvaluationDataset(samples=samples), raw_rows


def main() -> None:
    output_dir = PROJECT_ROOT / "eval" / "ragas_eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    agent = build_agent()
    dataset, raw_rows = load_dataset(agent)
    llm = build_local_chat_model(DEFAULT_AGENT_MODEL)

    metrics = [
        Faithfulness(llm=llm),
        LLMContextPrecisionWithReference(llm=llm),
    ]
    result = evaluate(dataset=dataset, metrics=metrics, llm=llm, show_progress=True)
    result_df = result.to_pandas()

    combined_df = pd.concat([pd.DataFrame(raw_rows), result_df], axis=1)
    csv_path = output_dir / "finance_agent_ragas_eval.csv"
    json_path = output_dir / "finance_agent_ragas_summary.json"
    combined_df.to_csv(csv_path, index=False)

    summary = {
        "num_questions": len(TEST_CASES),
        "average_faithfulness": float(result_df["faithfulness"].mean()),
        "average_context_relevance": float(
            result_df["llm_context_precision_with_reference"].mean()
        ),
        "results_csv": str(csv_path),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved detailed results to {csv_path}")


if __name__ == "__main__":
    main()
