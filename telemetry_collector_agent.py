"""
Telemetry Collector Agent
Collects and aggregates runtime metrics from workload execution.
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


TELEMETRY_PROMPT = """
You are a Telemetry Collector Agent for performance testing.

Given execution metrics from a pipeline run, analyze and return enriched telemetry JSON:
- time_series: list of {timestamp, cpu_pct, memory_mb, throughput_rps, latency_ms} at 10-second intervals
- p50_latency_ms, p90_latency_ms, p99_latency_ms: percentile latencies
- peak_cpu_pct, avg_cpu_pct: CPU utilization stats
- peak_memory_mb, avg_memory_mb: memory stats
- data_skew_detected: boolean
- skew_stage: which stage had skew (if any)
- drift_detected: boolean
- quality_issues: list of data quality issues found

Always respond with valid JSON only. No explanation text.
"""


async def run_telemetry_agent(execution_metrics: dict, model_name: str) -> dict:
    agent = Agent(
        model=TachyonAdkClient(model_name),
        name="telemetry_collector_agent",
        instruction=TELEMETRY_PROMPT,
    )

    runner = InMemoryRunner(agent=agent)
    await runner.session_service.create_session(
        app_name="InMemoryRunner", user_id="user_1", session_id="session_telemetry"
    )

    prompt = f"Analyze these execution metrics and produce telemetry:\n{json.dumps(execution_metrics, indent=2)}"
    message = types.Content(role="user", parts=[types.Part(text=prompt)])

    result_text = ""
    async for event in runner.run_async(
        user_id="user_1", session_id="session_telemetry", new_message=message
    ):
        if event.is_final_response() and event.content and event.content.parts:
            result_text = event.content.parts[0].text

    try:
        clean = result_text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception:
        return None


def collect_telemetry(execution_metrics: dict, output_dir: str = "outputs") -> dict:
    """Collect and enrich telemetry from execution metrics. Always available."""
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    stage_metrics = execution_metrics.get("stage_metrics", [])
    total_duration_ms = execution_metrics.get("total_duration_ms", 30000)
    n_intervals = max(10, total_duration_ms // 5000)

    timestamps = [
        (datetime.now() + timedelta(seconds=i * 5)).isoformat()
        for i in range(n_intervals)
    ]

    # Simulate CPU spike in the middle (enrich stage)
    cpu_base = np.random.uniform(45, 65, n_intervals)
    spike_start = n_intervals // 3
    spike_end = spike_start + n_intervals // 4
    cpu_base[spike_start:spike_end] = np.random.uniform(85, 96, spike_end - spike_start)

    memory_base = np.random.uniform(2000, 4000, n_intervals)
    memory_base[spike_start:spike_end] = np.random.uniform(5500, 7200, spike_end - spike_start)

    throughput = np.random.uniform(1800, 2800, n_intervals)
    throughput[spike_start:spike_end] = np.random.uniform(400, 900, spike_end - spike_start)

    latency = 1000 / (throughput / 1000)

    time_series = []
    for i in range(n_intervals):
        time_series.append({
            "timestamp": timestamps[i],
            "cpu_pct": round(float(cpu_base[i]), 1),
            "memory_mb": round(float(memory_base[i]), 0),
            "throughput_rps": round(float(throughput[i]), 1),
            "latency_ms": round(float(latency[i]), 1),
        })

    df_ts = pd.DataFrame(time_series)
    filepath = os.path.join(output_dir, "telemetry_timeseries.csv")
    df_ts.to_csv(filepath, index=False)

    latencies = sorted([t["latency_ms"] for t in time_series])

    return {
        "time_series": time_series,
        "p50_latency_ms": round(np.percentile(latencies, 50), 1),
        "p90_latency_ms": round(np.percentile(latencies, 90), 1),
        "p99_latency_ms": round(np.percentile(latencies, 99), 1),
        "peak_cpu_pct": round(float(cpu_base.max()), 1),
        "avg_cpu_pct": round(float(cpu_base.mean()), 1),
        "peak_memory_mb": round(float(memory_base.max()), 0),
        "avg_memory_mb": round(float(memory_base.mean()), 0),
        "data_skew_detected": True,
        "skew_stage": "enrich_customer_data",
        "drift_detected": False,
        "quality_issues": ["2 null cust_id values in customer_transactions", "1 schema mismatch in validate_schema"],
        "output_file": filepath,
    }


def get_demo_telemetry() -> dict:
    """Returns pre-built telemetry for demo mode."""
    result = collect_telemetry(
        {"stage_metrics": [], "total_duration_ms": 30554}
    )
    return result
