"""
Workload Execution Agent
Simulates and orchestrates workload execution, collecting execution metrics.
"""

import os
import json
import time
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from google.adk import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types
from tachyon_adk_client import TachyonAdkClient


WORKLOAD_EXECUTION_PROMPT = """
You are a Workload Execution Agent for performance testing.

Given a pipeline spec and data summary, simulate execution and return JSON metrics:
- stage_metrics: list per stage with duration_ms, rows_processed, cpu_pct, memory_mb, status
- total_duration_ms: total pipeline execution time
- throughput_rps: rows processed per second
- error_count: number of errors encountered
- warnings: list of warning messages
- execution_status: success/partial/failed

Simulate realistic performance degradation due to data skew and volume.
Always respond with valid JSON only. No explanation text.
"""


async def run_workload_execution_agent(
    pipeline_spec: dict, data_summary: dict, model_name: str
) -> dict:
    agent = Agent(
        model=TachyonAdkClient(model_name),
        name="workload_execution_agent",
        instruction=WORKLOAD_EXECUTION_PROMPT,
    )

    runner = InMemoryRunner(agent=agent)
    await runner.session_service.create_session(
        app_name="InMemoryRunner", user_id="user_1", session_id="session_exec"
    )

    prompt = (
        f"Simulate execution for this pipeline:\n{json.dumps(pipeline_spec, indent=2)}\n\n"
        f"Data summary:\n{json.dumps(data_summary, indent=2)}"
    )
    message = types.Content(role="user", parts=[types.Part(text=prompt)])

    result_text = ""
    async for event in runner.run_async(
        user_id="user_1", session_id="session_exec", new_message=message
    ):
        if event.is_final_response() and event.content and event.content.parts:
            result_text = event.content.parts[0].text

    try:
        clean = result_text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception:
        return None


def simulate_workload_execution(pipeline_spec: dict, data_summary: dict, output_dir: str = "outputs") -> dict:
    """Simulate workload execution locally and save metrics CSV. Always available."""
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(123)

    stages = pipeline_spec.get("stages", [])
    total_rows = data_summary.get("total_rows", 60000)
    stage_metrics = []

    base_durations = {
        "source": (800, 1500),
        "transform": (3000, 8000),
        "sink": (1200, 2500),
    }

    for i, stage in enumerate(stages):
        stage_type = stage.get("type", "transform")
        lo, hi = base_durations.get(stage_type, (2000, 6000))
        duration_ms = int(np.random.uniform(lo, hi))

        # Simulate skew impact on enrich stage
        if "enrich" in stage["name"]:
            duration_ms = int(duration_ms * 2.4)  # skew penalty

        rows = int(total_rows * np.random.uniform(0.85, 1.0))
        cpu = round(np.random.uniform(45, 92), 1)
        memory = int(np.random.uniform(1800, 7200))
        errors = np.random.choice([0, 0, 0, 1, 2], p=[0.7, 0.15, 0.10, 0.03, 0.02])

        stage_metrics.append({
            "stage": stage["name"],
            "stage_type": stage_type,
            "duration_ms": duration_ms,
            "rows_processed": rows,
            "cpu_pct": cpu,
            "memory_mb": memory,
            "error_count": int(errors),
            "status": "warning" if cpu > 85 or duration_ms > 10000 else "success",
            "timestamp": datetime.now().isoformat(),
        })

    total_duration = sum(s["duration_ms"] for s in stage_metrics)
    throughput = round((total_rows / total_duration) * 1000, 1)

    df = pd.DataFrame(stage_metrics)
    filepath = os.path.join(output_dir, "execution_metrics.csv")
    df.to_csv(filepath, index=False)

    return {
        "stage_metrics": stage_metrics,
        "total_duration_ms": total_duration,
        "throughput_rps": throughput,
        "error_count": sum(s["error_count"] for s in stage_metrics),
        "execution_status": "success",
        "output_file": filepath,
    }


def get_demo_execution_metrics() -> dict:
    """Returns pre-built execution metrics for demo mode."""
    stage_metrics = [
        {"stage": "ingest_raw_data", "stage_type": "source", "duration_ms": 1243, "rows_processed": 60000, "cpu_pct": 48.2, "memory_mb": 2048, "error_count": 0, "status": "success"},
        {"stage": "validate_schema", "stage_type": "transform", "duration_ms": 3871, "rows_processed": 59820, "cpu_pct": 61.5, "memory_mb": 3200, "error_count": 1, "status": "success"},
        {"stage": "enrich_customer_data", "stage_type": "transform", "duration_ms": 18340, "rows_processed": 59820, "cpu_pct": 91.7, "memory_mb": 6800, "error_count": 2, "status": "warning"},
        {"stage": "aggregate_metrics", "stage_type": "transform", "duration_ms": 5120, "rows_processed": 59820, "cpu_pct": 74.3, "memory_mb": 4100, "error_count": 0, "status": "success"},
        {"stage": "write_to_warehouse", "stage_type": "sink", "duration_ms": 1980, "rows_processed": 59820, "cpu_pct": 52.1, "memory_mb": 2400, "error_count": 0, "status": "success"},
    ]
    total_duration = sum(s["duration_ms"] for s in stage_metrics)
    return {
        "stage_metrics": stage_metrics,
        "total_duration_ms": total_duration,
        "throughput_rps": round((60000 / total_duration) * 1000, 1),
        "error_count": 3,
        "execution_status": "success",
        "output_file": "outputs/execution_metrics.csv",
    }
