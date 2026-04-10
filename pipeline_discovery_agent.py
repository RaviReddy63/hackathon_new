"""
Pipeline Discovery Agent
Reads pipeline definitions and builds a unified test specification.
"""

import os
import json
import asyncio
from google.adk import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types
from tachyon_adk_client import TachyonAdkClient


PIPELINE_DISCOVERY_PROMPT = """
You are a Pipeline Discovery Agent for performance testing.

Given a pipeline definition (mock or real), extract and return a JSON specification with:
- pipeline_name: name of the pipeline
- pipeline_type: one of [batch, streaming, api]
- stages: list of stages with name, type, dependencies
- datasets: list of input/output datasets with schema info
- slos: performance SLOs (latency_ms, throughput_rps, error_rate_pct)
- tunables: configurable parameters (parallelism, partitions, memory, etc.)
- estimated_complexity: low/medium/high

Always respond with valid JSON only. No explanation text.
"""


async def run_pipeline_discovery_agent(pipeline_input: dict, model_name: str) -> dict:
    agent = Agent(
        model=TachyonAdkClient(model_name),
        name="pipeline_discovery_agent",
        instruction=PIPELINE_DISCOVERY_PROMPT,
    )

    runner = InMemoryRunner(agent=agent)
    await runner.session_service.create_session(
        app_name="InMemoryRunner", user_id="user_1", session_id="session_pipeline"
    )

    prompt = f"Analyze this pipeline definition and return the specification:\n{json.dumps(pipeline_input, indent=2)}"
    message = types.Content(role="user", parts=[types.Part(text=prompt)])

    result_text = ""
    async for event in runner.run_async(
        user_id="user_1", session_id="session_pipeline", new_message=message
    ):
        if event.is_final_response() and event.content and event.content.parts:
            result_text = event.content.parts[0].text

    try:
        clean = result_text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception:
        return {"error": "Failed to parse pipeline spec", "raw": result_text}


def get_demo_pipeline_spec() -> dict:
    """Returns synthetic pipeline spec for demo mode."""
    return {
        "pipeline_name": "small_business_etl_pipeline",
        "pipeline_type": "batch",
        "stages": [
            {"name": "ingest_raw_data", "type": "source", "dependencies": []},
            {"name": "validate_schema", "type": "transform", "dependencies": ["ingest_raw_data"]},
            {"name": "enrich_customer_data", "type": "transform", "dependencies": ["validate_schema"]},
            {"name": "aggregate_metrics", "type": "transform", "dependencies": ["enrich_customer_data"]},
            {"name": "write_to_warehouse", "type": "sink", "dependencies": ["aggregate_metrics"]},
        ],
        "datasets": [
            {"name": "customer_transactions", "format": "parquet", "estimated_rows": 5000000, "schema": ["cust_id", "txn_date", "amount", "category"]},
            {"name": "customer_profile", "format": "csv", "estimated_rows": 500000, "schema": ["cust_id", "segment", "region", "risk_score"]},
        ],
        "slos": {
            "latency_ms": 30000,
            "throughput_rps": 50000,
            "error_rate_pct": 0.1,
        },
        "tunables": {
            "spark_parallelism": 200,
            "partition_count": 100,
            "executor_memory_gb": 4,
            "executor_cores": 2,
        },
        "estimated_complexity": "high",
    }
