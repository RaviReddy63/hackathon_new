"""
Synthetic Data Generator Agent
Creates realistic, schema-compliant mock datasets and saves to CSV.
"""

import os
import json
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from google.adk import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types
from tachyon_adk_client import TachyonAdkClient


SYNTHETIC_DATA_PROMPT = """
You are a Synthetic Data Generator Agent for performance testing.

Given a pipeline specification, return a JSON plan for generating synthetic test data:
- For each dataset: row_count, columns with data_type and distribution info
- Include skew_factor (0.0 to 1.0) to simulate data skew
- Include null_rate (0.0 to 0.1) per column
- Include late_arriving_pct (0.0 to 0.2) for streaming datasets
- generation_seed: integer for reproducibility

Always respond with valid JSON only. No explanation text.
"""


async def run_synthetic_data_agent(pipeline_spec: dict, model_name: str) -> dict:
    agent = Agent(
        model=TachyonAdkClient(model_name),
        name="synthetic_data_agent",
        instruction=SYNTHETIC_DATA_PROMPT,
    )

    runner = InMemoryRunner(agent=agent)
    await runner.session_service.create_session(
        app_name="InMemoryRunner", user_id="user_1", session_id="session_synth"
    )

    prompt = f"Generate a synthetic data plan for this pipeline:\n{json.dumps(pipeline_spec, indent=2)}"
    message = types.Content(role="user", parts=[types.Part(text=prompt)])

    result_text = ""
    async for event in runner.run_async(
        user_id="user_1", session_id="session_synth", new_message=message
    ):
        if event.is_final_response() and event.content and event.content.parts:
            result_text = event.content.parts[0].text

    try:
        clean = result_text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception:
        return None


def generate_synthetic_csvs(pipeline_spec: dict, output_dir: str = "data") -> dict:
    """Generate synthetic CSVs based on pipeline spec. Always available (no LLM needed)."""
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    generated_files = {}
    row_counts = {}

    for dataset in pipeline_spec.get("datasets", []):
        name = dataset["name"]
        schema = dataset.get("schema", [])
        estimated_rows = min(dataset.get("estimated_rows", 10000), 50000)  # cap for demo

        df = _generate_dataset(name, schema, estimated_rows)
        filepath = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(filepath, index=False)
        generated_files[name] = filepath
        row_counts[name] = len(df)

    return {
        "generated_files": generated_files,
        "row_counts": row_counts,
        "total_rows": sum(row_counts.values()),
        "output_dir": output_dir,
    }


def _generate_dataset(name: str, schema: list, rows: int) -> pd.DataFrame:
    np.random.seed(42)
    data = {}

    column_generators = {
        "cust_id": lambda n: [f"CUST_{i:07d}" for i in np.random.randint(1, 500000, n)],
        "txn_date": lambda n: [
            (datetime(2024, 1, 1) + timedelta(days=int(d))).strftime("%Y-%m-%d")
            for d in np.random.randint(0, 365, n)
        ],
        "amount": lambda n: np.round(np.random.lognormal(4.5, 1.2, n), 2).tolist(),
        "category": lambda n: np.random.choice(
            ["retail", "food", "travel", "utilities", "healthcare", "other"], n,
            p=[0.30, 0.25, 0.15, 0.15, 0.10, 0.05]
        ).tolist(),
        "segment": lambda n: np.random.choice(
            ["enterprise", "mid_market", "small_biz", "micro"], n,
            p=[0.10, 0.20, 0.45, 0.25]
        ).tolist(),
        "region": lambda n: np.random.choice(
            ["WEST", "EAST", "CENTRAL", "SOUTH", "NORTHWEST"], n
        ).tolist(),
        "risk_score": lambda n: np.round(np.random.beta(2, 5, n), 3).tolist(),
    }

    for col in schema:
        if col in column_generators:
            data[col] = column_generators[col](rows)
        else:
            data[col] = np.random.randint(1, 1000, rows).tolist()

    # Add skew: 20% of customers generate 80% of transactions
    if "cust_id" in data and "amount" in data:
        skew_mask = np.random.random(rows) < 0.20
        amounts = np.array(data["amount"])
        amounts[skew_mask] = amounts[skew_mask] * 5.0
        data["amount"] = np.round(amounts, 2).tolist()

    return pd.DataFrame(data)


def get_demo_data_summary() -> dict:
    """Returns synthetic data generation summary for demo mode."""
    return {
        "generated_files": {
            "customer_transactions": "data/customer_transactions.csv",
            "customer_profile": "data/customer_profile.csv",
        },
        "row_counts": {
            "customer_transactions": 50000,
            "customer_profile": 10000,
        },
        "total_rows": 60000,
        "skew_factor": 0.78,
        "null_rate_avg": 0.02,
        "generation_seed": 42,
        "output_dir": "data",
    }
