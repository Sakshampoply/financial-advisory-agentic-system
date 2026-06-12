"""One-time setup script to create Langfuse evaluation datasets.

Run once after configuring LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST:
    uv run python evals/setup_langfuse_datasets.py
"""
import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"

RAG_KNOWN_QUERIES = json.loads((FIXTURES / "rag_test_cases.json").read_text())
ADVISOR_CASES = json.loads((FIXTURES / "advisor_test_cases.json").read_text())
INTAKE_CASES = json.loads((FIXTURES / "intake_test_cases.json").read_text())


def create_datasets():
    try:
        from langfuse import Langfuse
        from app.config import settings
        if not settings.LANGFUSE_PUBLIC_KEY:
            print("ERROR: LANGFUSE_PUBLIC_KEY is not set in ../.env")
            print("Get your keys at https://cloud.langfuse.com → Settings → API Keys")
            return
        lf = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
    except Exception as e:
        print(f"Failed to initialise Langfuse client: {e}")
        return

    # ---- RAG faithfulness dataset ----
    ds_name = "rag_faithfulness_v1"
    try:
        lf.create_dataset(name=ds_name, description="RAG retrieval ground truth: query → expected KB sources")
        print(f"Created dataset: {ds_name}")
    except Exception:
        print(f"Dataset {ds_name} already exists — adding items only")

    for case in RAG_KNOWN_QUERIES:
        try:
            lf.create_dataset_item(
                dataset_name=ds_name,
                input={"query": case["query"]},
                expected_output={"source_filenames": case["expected_sources"]},
            )
        except Exception as e:
            print(f"  Warning: could not add item: {e}")

    print(f"  {len(RAG_KNOWN_QUERIES)} items added to {ds_name}")

    # ---- Advisor quality dataset ----
    ds_name = "advisor_quality_v1"
    try:
        lf.create_dataset(
            name=ds_name,
            description="Advisor response quality: (user_message, intent) → must_contain assertions",
        )
        print(f"Created dataset: {ds_name}")
    except Exception:
        print(f"Dataset {ds_name} already exists — adding items only")

    for case in ADVISOR_CASES:
        try:
            lf.create_dataset_item(
                dataset_name=ds_name,
                input={"user_message": case["input"], "intent": case["intent"]},
                expected_output={
                    "must_contain": case["must_contain"],
                    "must_not_contain": case.get("must_not_contain", []),
                },
            )
        except Exception as e:
            print(f"  Warning: could not add item: {e}")

    print(f"  {len(ADVISOR_CASES)} items added to {ds_name}")

    # ---- Intake extraction dataset ----
    ds_name = "intake_extraction_v1"
    try:
        lf.create_dataset(
            name=ds_name,
            description="Intake LLM extraction accuracy: user message → expected fields",
        )
        print(f"Created dataset: {ds_name}")
    except Exception:
        print(f"Dataset {ds_name} already exists — adding items only")

    for case in INTAKE_CASES:
        if "expected_fields" not in case and "expected_portfolio_tickers" not in case:
            continue
        try:
            lf.create_dataset_item(
                dataset_name=ds_name,
                input={"user_message": case.get("single_message", ""), "description": case["description"]},
                expected_output={
                    "fields": case.get("expected_fields", {}),
                    "portfolio_tickers": case.get("expected_portfolio_tickers", []),
                    "wants_new_portfolio": case.get("expected_wants_new_portfolio", False),
                },
            )
        except Exception as e:
            print(f"  Warning: could not add item: {e}")

    print(f"  Intake items added to {ds_name}")

    # ---- RAG retrieval quality v2 — 38 queries, all 40 sources, source_tier metadata ----
    ds_name = "rag_retrieval_quality_v2"
    try:
        lf.create_dataset(
            name=ds_name,
            description="38 queries covering all 40 KB sources across 5 categories with source_tier metadata",
        )
        print(f"Created dataset: {ds_name}")
    except Exception:
        print(f"Dataset {ds_name} already exists — adding items only")

    for case in RAG_KNOWN_QUERIES:
        try:
            lf.create_dataset_item(
                dataset_name=ds_name,
                input={"query": case["query"], "category": case["category"]},
                expected_output={
                    "source_filenames": case["expected_sources"],
                    "source_tier": case["source_tier"],
                },
            )
        except Exception as e:
            print(f"  Warning: could not add item: {e}")

    print(f"  {len(RAG_KNOWN_QUERIES)} items added to {ds_name}")

    # ---- Advisor quality v2 — 8 cases covering all 4 intents + edge cases ----
    ds_name = "advisor_quality_v2"
    try:
        lf.create_dataset(
            name=ds_name,
            description="8 advisor quality cases covering all 4 intents + edge cases (no-KB, out-of-scope)",
        )
        print(f"Created dataset: {ds_name}")
    except Exception:
        print(f"Dataset {ds_name} already exists — adding items only")

    for case in ADVISOR_CASES:
        try:
            lf.create_dataset_item(
                dataset_name=ds_name,
                input={"user_message": case["input"], "intent": case["intent"]},
                expected_output={
                    "must_contain": case["must_contain"],
                    "must_not_contain": case.get("must_not_contain", []),
                    "description": case["description"],
                },
            )
        except Exception as e:
            print(f"  Warning: could not add item: {e}")

    print(f"  {len(ADVISOR_CASES)} items added to {ds_name}")
    print("\nDataset setup complete. Check your Langfuse dashboard → Datasets tab.")


if __name__ == "__main__":
    create_datasets()
