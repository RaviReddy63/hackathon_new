"""
Orchestrator / Supervisor Agent
Chains all agents in sequence and manages state between them.
"""

import os
import json
import asyncio
import time
from datetime import datetime
from dotenv import load_dotenv
import warnings

load_dotenv()
warnings.filterwarnings("ignore", message="coroutine.*was never awaited", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Unverified HTTPS request", category=Warning)

# --- Secret Fetching (same pattern as test_agent_final.py) ---
import requests

VAULT_KEY_MAP = {
    "consumer_key": "CONSUMER_KEY",
    "consumer_secret": "CONSUMER_SECRET",
    "api_key": "API_KEY",
}


def fetch_vault_secrets(env_var):
    token = os.getenv("TOKEN")
    url = os.getenv(env_var)
    if not token or not url:
        raise RuntimeError(f"TOKEN or {env_var} not found in .env")
    resp = requests.get(url, headers={"X-Vault-Token": token}, verify=False)
    resp.raise_for_status()
    secrets = resp.json().get("data", {}).get("data", {})
    if not secrets:
        raise RuntimeError(f"No secrets found at {env_var}")
    return secrets


def inject_secrets(*secret_dicts):
    for secrets in secret_dicts:
        for key, value in secrets.items():
            env_key = VAULT_KEY_MAP.get(key, key)
            os.environ[env_key] = str(value)


def setup_secrets():
    """Fetch and inject secrets from Vault. Returns model_name."""
    secrets = fetch_vault_secrets("VAULT_URL")
    apigee_secrets = fetch_vault_secrets("VAULT_APIGEE_URL")
    inject_secrets(secrets, apigee_secrets)
    model_name = secrets.get("MODEL") or os.getenv("MODEL")
    if not model_name:
        raise RuntimeError("MODEL not found in Vault or .env")
    return model_name


# --- Agent imports ---
from agents.pipeline_discovery_agent import (
    run_pipeline_discovery_agent, get_demo_pipeline_spec
)
from agents.synthetic_data_agent import (
    run_synthetic_data_agent, generate_synthetic_csvs, get_demo_data_summary
)
from agents.workload_execution_agent import (
    simulate_workload_execution, get_demo_execution_metrics
)
from agents.telemetry_agent import (
    collect_telemetry, get_demo_telemetry
)
from agents.rca_agent import (
    perform_rca, get_demo_rca
)
from agents.remediation_agent import (
    generate_remediations, get_demo_remediations
)


SAMPLE_PIPELINE_INPUT = {
    "name": "small_business_etl_pipeline",
    "description": "ETL pipeline ingesting customer transaction data for small business analytics",
    "type": "batch",
    "source": "S3 parquet files",
    "destination": "Redshift warehouse",
    "dag_tool": "Airflow",
    "processing_engine": "Spark 3.4",
    "approximate_daily_rows": 5000000,
}


def run_demo_pipeline(output_dir: str = "outputs") -> dict:
    """Run the full pipeline in DEMO mode using synthetic data. No LLM/Vault needed."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    results = {"mode": "demo", "steps": {}, "start_time": datetime.now().isoformat()}
    step_timings = {}

    print("🔍 [Step 1/6] Pipeline Discovery...")
    t0 = time.time()
    pipeline_spec = get_demo_pipeline_spec()
    step_timings["pipeline_discovery"] = round(time.time() - t0, 2)
    results["steps"]["pipeline_discovery"] = {"status": "completed", "data": pipeline_spec}
    print(f"   ✅ Done in {step_timings['pipeline_discovery']}s")

    print("🧪 [Step 2/6] Generating Synthetic Data...")
    t0 = time.time()
    data_summary = generate_synthetic_csvs(pipeline_spec, output_dir="data")
    step_timings["synthetic_data"] = round(time.time() - t0, 2)
    results["steps"]["synthetic_data"] = {"status": "completed", "data": data_summary}
    print(f"   ✅ Done in {step_timings['synthetic_data']}s — {data_summary['total_rows']:,} rows generated")

    print("⚡ [Step 3/6] Executing Workload...")
    t0 = time.time()
    execution_metrics = simulate_workload_execution(pipeline_spec, data_summary, output_dir=output_dir)
    step_timings["workload_execution"] = round(time.time() - t0, 2)
    results["steps"]["workload_execution"] = {"status": "completed", "data": execution_metrics}
    print(f"   ✅ Done in {step_timings['workload_execution']}s")

    print("📡 [Step 4/6] Collecting Telemetry...")
    t0 = time.time()
    telemetry = collect_telemetry(execution_metrics, output_dir=output_dir)
    step_timings["telemetry"] = round(time.time() - t0, 2)
    results["steps"]["telemetry"] = {"status": "completed", "data": telemetry}
    print(f"   ✅ Done in {step_timings['telemetry']}s")

    print("🔬 [Step 5/6] Running RCA & Insights...")
    t0 = time.time()
    rca_result = perform_rca(execution_metrics, telemetry, pipeline_spec, output_dir=output_dir)
    step_timings["rca"] = round(time.time() - t0, 2)
    results["steps"]["rca"] = {"status": "completed", "data": rca_result}
    print(f"   ✅ Done in {step_timings['rca']}s — Health Score: {rca_result['overall_health_score']}/100")

    print("💡 [Step 6/6] Generating Remediations...")
    t0 = time.time()
    remediations = generate_remediations(rca_result, pipeline_spec, output_dir=output_dir)
    step_timings["remediations"] = round(time.time() - t0, 2)
    results["steps"]["remediations"] = {"status": "completed", "data": remediations}
    print(f"   ✅ Done in {step_timings['remediations']}s — {len(remediations['recommendations'])} recommendations")

    results["step_timings"] = step_timings
    results["end_time"] = datetime.now().isoformat()
    results["status"] = "completed"

    output_path = os.path.join(output_dir, "pipeline_run_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✅ Demo pipeline complete. Results saved to {output_path}")
    return results


async def run_live_pipeline(output_dir: str = "outputs") -> dict:
    """Run the full pipeline in LIVE mode using real LLM agents via Tachyon."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("🔐 Setting up secrets from Vault...")
    model_name = setup_secrets()
    print(f"   ✅ Model: {model_name}")

    results = {"mode": "live", "model": model_name, "steps": {}, "start_time": datetime.now().isoformat()}

    print("🔍 [Step 1/6] Pipeline Discovery Agent...")
    pipeline_spec = await run_pipeline_discovery_agent(SAMPLE_PIPELINE_INPUT, model_name)
    if pipeline_spec.get("error"):
        print("   ⚠️  LLM failed, using fallback spec")
        pipeline_spec = get_demo_pipeline_spec()
    results["steps"]["pipeline_discovery"] = {"status": "completed", "data": pipeline_spec}

    print("🧪 [Step 2/6] Synthetic Data Agent...")
    data_summary = generate_synthetic_csvs(pipeline_spec, output_dir="data")
    results["steps"]["synthetic_data"] = {"status": "completed", "data": data_summary}

    print("⚡ [Step 3/6] Workload Execution Agent...")
    execution_metrics = simulate_workload_execution(pipeline_spec, data_summary, output_dir=output_dir)
    results["steps"]["workload_execution"] = {"status": "completed", "data": execution_metrics}

    print("📡 [Step 4/6] Telemetry Agent...")
    telemetry = collect_telemetry(execution_metrics, output_dir=output_dir)
    results["steps"]["telemetry"] = {"status": "completed", "data": telemetry}

    print("🔬 [Step 5/6] RCA Agent...")
    rca_result = await run_rca_agent(execution_metrics, telemetry, pipeline_spec, model_name)
    if not rca_result:
        rca_result = perform_rca(execution_metrics, telemetry, pipeline_spec, output_dir=output_dir)
    results["steps"]["rca"] = {"status": "completed", "data": rca_result}

    print("💡 [Step 6/6] Remediation Agent...")
    remediations = await run_remediation_agent(rca_result, pipeline_spec, model_name)
    if not remediations:
        remediations = generate_remediations(rca_result, pipeline_spec, output_dir=output_dir)
    results["steps"]["remediations"] = {"status": "completed", "data": remediations}

    results["end_time"] = datetime.now().isoformat()
    results["status"] = "completed"

    output_path = os.path.join(output_dir, "pipeline_run_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✅ Live pipeline complete. Results saved to {output_path}")
    return results


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if mode == "live":
        asyncio.run(run_live_pipeline())
    else:
        run_demo_pipeline()
