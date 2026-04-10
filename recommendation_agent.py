"""
Remediation Agent
Proposes and generates tuning recommendations based on RCA output.
"""

import os
import json
import asyncio
import pandas as pd
from google.adk import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types
from tachyon_adk_client import TachyonAdkClient


REMEDIATION_PROMPT = """
You are a Remediation Agent for data pipeline performance testing.

Given RCA results and pipeline spec, return a JSON with:
- recommendations: list of {
    id, title, priority (P0/P1/P2/P3),
    category (configuration/architecture/data/infrastructure),
    current_value, recommended_value,
    expected_improvement_pct,
    effort (low/medium/high),
    description
  }
- auto_tuned_config: dict of parameter changes to apply immediately
- estimated_improvement: {latency_pct, throughput_pct, cost_pct}
- canary_experiment_suggested: boolean
- implementation_order: list of recommendation ids in order of priority

Always respond with valid JSON only. No explanation text.
"""


async def run_remediation_agent(
    rca_result: dict, pipeline_spec: dict, model_name: str
) -> dict:
    agent = Agent(
        model=TachyonAdkClient(model_name),
        name="remediation_agent",
        instruction=REMEDIATION_PROMPT,
    )

    runner = InMemoryRunner(agent=agent)
    await runner.session_service.create_session(
        app_name="InMemoryRunner", user_id="user_1", session_id="session_remediation"
    )

    prompt = (
        f"Generate remediation recommendations for this RCA:\n{json.dumps(rca_result, indent=2)}\n\n"
        f"Pipeline tunables:\n{json.dumps(pipeline_spec.get('tunables', {}), indent=2)}"
    )
    message = types.Content(role="user", parts=[types.Part(text=prompt)])

    result_text = ""
    async for event in runner.run_async(
        user_id="user_1", session_id="session_remediation", new_message=message
    ):
        if event.is_final_response() and event.content and event.content.parts:
            result_text = event.content.parts[0].text

    try:
        clean = result_text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception:
        return None


def generate_remediations(rca_result: dict, pipeline_spec: dict, output_dir: str = "outputs") -> dict:
    """Generate rule-based remediation recommendations. Always available."""
    os.makedirs(output_dir, exist_ok=True)

    tunables = pipeline_spec.get("tunables", {})
    bottlenecks = rca_result.get("bottlenecks", [])
    root_causes = rca_result.get("root_causes", [])

    recommendations = []

    # Check for skew
    has_skew = any("skew" in rc.get("cause", "").lower() for rc in root_causes)
    if has_skew:
        recommendations.append({
            "id": "REC-001",
            "title": "Enable Adaptive Query Execution (AQE) to handle data skew",
            "priority": "P0",
            "category": "configuration",
            "current_value": "spark.sql.adaptive.enabled=false",
            "recommended_value": "spark.sql.adaptive.enabled=true, spark.sql.adaptive.skewJoin.enabled=true",
            "expected_improvement_pct": 45,
            "effort": "low",
            "description": "AQE dynamically coalesces shuffle partitions and handles skewed joins automatically, directly addressing the 2.4x slowdown in enrich_customer_data.",
        })

    # CPU saturation
    high_cpu = any(b.get("issue", "").startswith("CPU") for b in bottlenecks)
    if high_cpu:
        current_cores = tunables.get("executor_cores", 2)
        recommendations.append({
            "id": "REC-002",
            "title": "Increase executor cores and enable dynamic allocation",
            "priority": "P0",
            "category": "infrastructure",
            "current_value": f"executor_cores={current_cores}, dynamic_allocation=disabled",
            "recommended_value": f"executor_cores={current_cores + 2}, dynamic_allocation=enabled, max_executors=20",
            "expected_improvement_pct": 30,
            "effort": "low",
            "description": "CPU at 91.7% indicates under-provisioned executors. Increasing cores and enabling dynamic allocation allows the cluster to scale with load.",
        })

    # Memory pressure
    high_mem = any("memory" in b.get("issue", "").lower() for b in bottlenecks)
    if high_mem:
        current_mem = tunables.get("executor_memory_gb", 4)
        recommendations.append({
            "id": "REC-003",
            "title": "Increase executor memory and tune memory fractions",
            "priority": "P1",
            "category": "configuration",
            "current_value": f"executor_memory={current_mem}g, memory_fraction=0.6",
            "recommended_value": f"executor_memory={current_mem + 4}g, memory_fraction=0.7, storage_fraction=0.4",
            "expected_improvement_pct": 20,
            "effort": "low",
            "description": "Memory pressure at 6.8GB risks GC pauses. Increasing to 8g with tuned fractions reduces spill to disk.",
        })

    # Partition optimization
    recommendations.append({
        "id": "REC-004",
        "title": "Optimize partition count based on data volume",
        "priority": "P1",
        "category": "configuration",
        "current_value": f"partition_count={tunables.get('partition_count', 100)}",
        "recommended_value": "partition_count=400, target_partition_size=128mb",
        "expected_improvement_pct": 25,
        "effort": "medium",
        "description": "Current partition count causes large partitions. Target 128MB per partition for optimal parallelism across 60M rows.",
    })

    # Data quality
    recommendations.append({
        "id": "REC-005",
        "title": "Add upstream data quality checks before enrich stage",
        "priority": "P2",
        "category": "architecture",
        "current_value": "No pre-validation before enrich_customer_data",
        "recommended_value": "Add Great Expectations checks: null check on cust_id, schema validation",
        "expected_improvement_pct": 10,
        "effort": "medium",
        "description": "Null cust_id values cause silent failures in enrichment. Early validation prevents downstream errors and wasted compute.",
    })

    auto_tuned_config = {
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.skewJoin.enabled": "true",
        "spark.executor.cores": str(tunables.get("executor_cores", 2) + 2),
        "spark.executor.memory": f"{tunables.get('executor_memory_gb', 4) + 4}g",
        "spark.sql.shuffle.partitions": "400",
        "spark.dynamicAllocation.enabled": "true",
        "spark.dynamicAllocation.maxExecutors": "20",
    }

    result = {
        "recommendations": recommendations,
        "auto_tuned_config": auto_tuned_config,
        "estimated_improvement": {
            "latency_pct": 52,
            "throughput_pct": 68,
            "cost_pct": -15,
        },
        "canary_experiment_suggested": True,
        "implementation_order": [r["id"] for r in sorted(recommendations, key=lambda x: x["priority"])],
    }

    df = pd.DataFrame(recommendations)
    filepath = os.path.join(output_dir, "remediation_recommendations.csv")
    df.to_csv(filepath, index=False)
    result["output_file"] = filepath

    return result


def get_demo_remediations() -> dict:
    from agents.rca_agent import get_demo_rca
    from agents.pipeline_discovery_agent import get_demo_pipeline_spec
    return generate_remediations(get_demo_rca(), get_demo_pipeline_spec())
