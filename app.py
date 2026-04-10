"""
Execution Agent - Performance Testing Platform
Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Execution Agent | Perf Testing",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}

.stApp { background-color: #0d1117; }

.metric-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
}

.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.1;
}

.metric-label {
    font-size: 0.75rem;
    font-weight: 500;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

.health-score-a { color: #3fb950; }
.health-score-b { color: #58a6ff; }
.health-score-c { color: #d29922; }
.health-score-d { color: #f85149; }

.stage-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
}

.badge-success { background:#1f4f2d; color:#3fb950; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-weight:600; }
.badge-warning { background:#3d2e05; color:#d29922; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-weight:600; }
.badge-critical { background:#4d1111; color:#f85149; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-weight:600; }
.badge-p0 { background:#4d1111; color:#f85149; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-weight:700; }
.badge-p1 { background:#3d2e05; color:#d29922; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-weight:700; }
.badge-p2 { background:#1c2e4a; color:#58a6ff; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-weight:700; }

.pipeline-step {
    display: flex; align-items: center; gap: 12px;
    background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 14px 20px; margin-bottom: 8px;
}
.step-icon { font-size: 1.4rem; }
.step-name { font-weight: 600; font-size: 0.9rem; }
.step-status { font-size: 0.75rem; color: #8b949e; margin-top: 2px; }

.section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    color: #58a6ff;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 1px solid #21262d;
    padding-bottom: 8px;
    margin-bottom: 16px;
    margin-top: 24px;
}

.rca-card {
    background: #161b22; border-left: 3px solid #f85149;
    border-radius: 0 8px 8px 0; padding: 14px 16px; margin-bottom: 8px;
}

.rec-card {
    background: #161b22; border-left: 3px solid #3fb950;
    border-radius: 0 8px 8px 0; padding: 14px 16px; margin-bottom: 8px;
}

.summary-box {
    background: linear-gradient(135deg, #1c2333 0%, #0d1117 100%);
    border: 1px solid #388bfd40;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
    font-size: 0.9rem;
    line-height: 1.6;
    color: #c9d1d9;
}

[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
[data-testid="stSidebar"] .stMarkdown { color: #e6edf3; }

.stButton > button {
    background: linear-gradient(135deg, #238636, #2ea043);
    color: white; border: none; border-radius: 6px;
    font-weight: 600; font-family: 'Inter', sans-serif;
    padding: 10px 24px; width: 100%;
}
.stButton > button:hover { background: linear-gradient(135deg, #2ea043, #3fb950); }

div[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 16px;
}
</style>
""", unsafe_allow_html=True)


# ── Plotly dark theme helper ───────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(family="Inter", color="#c9d1d9", size=11),
    margin=dict(l=40, r=20, t=30, b=40),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
)


# ── Data loading helpers ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_results_from_file(path: str):
    with open(path) as f:
        return json.load(f)


def get_results(demo_mode: bool):
    results_path = "outputs/pipeline_run_results.json"
    if not demo_mode and os.path.exists(results_path):
        return load_results_from_file(results_path)

    # Generate demo results
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from orchestrator import run_demo_pipeline
    return run_demo_pipeline()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Execution Agent")
    st.markdown("<div style='color:#8b949e; font-size:0.8rem; margin-bottom:20px;'>Multi-Agent Performance Testing Platform</div>", unsafe_allow_html=True)

    st.markdown("---")
    demo_mode = st.toggle("🎭 Demo Mode", value=True, help="Use synthetic data (no Tachyon/Vault required)")

    if demo_mode:
        st.info("Running with synthetic data. All metrics are simulated.", icon="🧪")
    else:
        st.warning("Live mode requires Vault & Tachyon access.", icon="🔐")

    st.markdown("---")
    st.markdown("**Pipeline Config**")
    pipeline_name = st.text_input("Pipeline Name", value="small_business_etl_pipeline")
    pipeline_type = st.selectbox("Type", ["batch", "streaming", "api"])

    st.markdown("---")
    run_btn = st.button("▶ Run Pipeline", use_container_width=True)
    st.markdown("---")

    st.markdown("""
    <div style='font-size:0.72rem; color:#8b949e;'>
    <b>Agents</b><br>
    🔍 Pipeline Discovery<br>
    🧪 Synthetic Data Gen<br>
    ⚡ Workload Execution<br>
    📡 Telemetry Collector<br>
    🔬 RCA & Insights<br>
    💡 Remediation
    </div>
    """, unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None
if "running" not in st.session_state:
    st.session_state.running = False


# ── Header ─────────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("# ⚡ Execution Agent")
    st.markdown(f"<div style='color:#8b949e; font-size:0.85rem;'>Autonomous Multi-Agent Performance Testing Platform &nbsp;|&nbsp; {'🎭 Demo Mode' if demo_mode else '🔴 Live Mode'}</div>", unsafe_allow_html=True)
with col_h2:
    mode_badge = "🧪 DEMO" if demo_mode else "🔴 LIVE"
    mode_color = "#d29922" if demo_mode else "#f85149"
    st.markdown(f"<div style='text-align:right; padding-top:10px;'><span style='background:#161b22; border:1px solid {mode_color}; color:{mode_color}; padding:6px 14px; border-radius:20px; font-size:0.8rem; font-weight:700; font-family:JetBrains Mono;'>{mode_badge}</span></div>", unsafe_allow_html=True)

st.markdown("---")


# ── Pipeline execution with step-by-step progress ─────────────────────────────
STEPS = [
    ("🔍", "Pipeline Discovery", "Parsing DAGs, configs, and lineage"),
    ("🧪", "Synthetic Data Generation", "Creating schema-compliant mock datasets"),
    ("⚡", "Workload Execution", "Running batch workloads at scale"),
    ("📡", "Telemetry Collection", "Gathering runtime metrics"),
    ("🔬", "RCA & Insights", "Identifying bottlenecks and root causes"),
    ("💡", "Remediation", "Generating optimization recommendations"),
]

if run_btn or st.session_state.results is None:
    st.session_state.running = True
    progress_container = st.container()

    with progress_container:
        st.markdown("<div class='section-header'>Pipeline Execution</div>", unsafe_allow_html=True)
        step_placeholders = []
        for icon, name, desc in STEPS:
            ph = st.empty()
            ph.markdown(
                f"<div class='stage-card' style='opacity:0.4;'>{icon} <b>{name}</b> — <span style='color:#8b949e;'>{desc}</span> &nbsp; ⏳</div>",
                unsafe_allow_html=True,
            )
            step_placeholders.append(ph)

        prog = st.progress(0, text="Initializing agents...")
        time.sleep(0.3)

        # Simulate step-by-step execution
        import sys
        sys.path.insert(0, os.path.dirname(__file__))

        if demo_mode:
            from orchestrator import run_demo_pipeline
            from agents.pipeline_discovery_agent import get_demo_pipeline_spec
            from agents.synthetic_data_agent import generate_synthetic_csvs
            from agents.workload_execution_agent import simulate_workload_execution
            from agents.telemetry_agent import collect_telemetry
            from agents.rca_agent import perform_rca
            from agents.remediation_agent import generate_remediations

            pipeline_spec = None
            data_summary = None
            execution_metrics = None
            telemetry = None
            rca_result = None
            remediations = None

            for i, (icon, name, desc) in enumerate(STEPS):
                prog.progress((i) / len(STEPS), text=f"Running: {name}...")

                if i == 0:
                    pipeline_spec = get_demo_pipeline_spec()
                elif i == 1:
                    os.makedirs("data", exist_ok=True)
                    data_summary = generate_synthetic_csvs(pipeline_spec, output_dir="data")
                elif i == 2:
                    os.makedirs("outputs", exist_ok=True)
                    execution_metrics = simulate_workload_execution(pipeline_spec, data_summary, output_dir="outputs")
                elif i == 3:
                    telemetry = collect_telemetry(execution_metrics, output_dir="outputs")
                elif i == 4:
                    rca_result = perform_rca(execution_metrics, telemetry, pipeline_spec, output_dir="outputs")
                elif i == 5:
                    remediations = generate_remediations(rca_result, pipeline_spec, output_dir="outputs")

                time.sleep(0.4)
                step_placeholders[i].markdown(
                    f"<div class='stage-card'>{icon} <b>{name}</b> — <span style='color:#8b949e;'>{desc}</span> &nbsp; <span style='color:#3fb950;'>✓ Done</span></div>",
                    unsafe_allow_html=True,
                )
                prog.progress((i + 1) / len(STEPS), text=f"Completed: {name}")

            st.session_state.results = {
                "pipeline_spec": pipeline_spec,
                "data_summary": data_summary,
                "execution_metrics": execution_metrics,
                "telemetry": telemetry,
                "rca_result": rca_result,
                "remediations": remediations,
            }
        prog.progress(1.0, text="✅ All agents completed!")
        time.sleep(0.5)
        prog.empty()
        st.session_state.running = False


# ── Dashboard rendering ────────────────────────────────────────────────────────
results = st.session_state.results

if results:
    pipeline_spec = results["pipeline_spec"]
    data_summary = results["data_summary"]
    execution_metrics = results["execution_metrics"]
    telemetry = results["telemetry"]
    rca_result = results["rca_result"]
    remediations = results["remediations"]

    stage_metrics = execution_metrics.get("stage_metrics", [])

    # ── KPI Row ────────────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Key Performance Indicators</div>", unsafe_allow_html=True)
    k1, k2, k3, k4, k5, k6 = st.columns(6)

    health = rca_result.get("overall_health_score", 0)
    grade = rca_result.get("health_grade", "?")
    grade_class = f"health-score-{grade.lower()}"

    with k1:
        st.metric("Health Score", f"{health}/100", delta=f"Grade {grade}")
    with k2:
        st.metric("Total Duration", f"{execution_metrics['total_duration_ms']/1000:.1f}s", delta=f"SLO: 30s")
    with k3:
        st.metric("Throughput", f"{execution_metrics['throughput_rps']:.0f} RPS", delta=f"SLO: 50K RPS", delta_color="inverse")
    with k4:
        st.metric("P99 Latency", f"{telemetry.get('p99_latency_ms', 0):.0f}ms")
    with k5:
        st.metric("Peak CPU", f"{telemetry.get('peak_cpu_pct', 0)}%", delta_color="inverse")
    with k6:
        st.metric("Total Rows", f"{data_summary.get('total_rows', 0):,}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Stage Metrics",
        "📡 Telemetry",
        "🔬 RCA & Bottlenecks",
        "💡 Recommendations",
        "📁 Data & Files",
    ])

    # ── TAB 1: Stage Metrics ───────────────────────────────────────────────────
    with tab1:
        st.markdown("<div class='section-header'>Stage-by-Stage Execution</div>", unsafe_allow_html=True)

        df_stages = pd.DataFrame(stage_metrics)

        col_l, col_r = st.columns([3, 2])

        with col_l:
            # Duration bar chart
            fig = go.Figure()
            colors = ["#f85149" if s["status"] == "warning" else "#3fb950" for s in stage_metrics]
            fig.add_trace(go.Bar(
                x=[s["stage"] for s in stage_metrics],
                y=[s["duration_ms"] / 1000 for s in stage_metrics],
                marker_color=colors,
                text=[f"{s['duration_ms']/1000:.1f}s" for s in stage_metrics],
                textposition="outside",
            ))
            fig.update_layout(
                title="Stage Duration (seconds)",
                **PLOT_LAYOUT,
                yaxis_title="Duration (s)",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            # CPU vs Memory scatter
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=[s["cpu_pct"] for s in stage_metrics],
                y=[s["memory_mb"] for s in stage_metrics],
                mode="markers+text",
                marker=dict(
                    size=16,
                    color=[s["cpu_pct"] for s in stage_metrics],
                    colorscale=[[0, "#3fb950"], [0.7, "#d29922"], [1.0, "#f85149"]],
                    showscale=True,
                    colorbar=dict(title="CPU %", thickness=10),
                ),
                text=[s["stage"].replace("_", "<br>") for s in stage_metrics],
                textposition="top center",
                textfont=dict(size=9),
            ))
            fig2.update_layout(
                title="CPU % vs Memory (MB)",
                xaxis_title="CPU %",
                yaxis_title="Memory (MB)",
                **PLOT_LAYOUT,
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Stage detail table
        st.markdown("<div class='section-header'>Stage Detail</div>", unsafe_allow_html=True)
        for s in stage_metrics:
            badge = f"<span class='badge-warning'>⚠ WARNING</span>" if s["status"] == "warning" else f"<span class='badge-success'>✓ SUCCESS</span>"
            st.markdown(f"""
            <div class='stage-card'>
                <b>{s['stage']}</b> &nbsp; {badge} &nbsp;
                | ⏱ {s['duration_ms']/1000:.2f}s &nbsp;
                | 🔢 {s['rows_processed']:,} rows &nbsp;
                | 💻 CPU {s['cpu_pct']}% &nbsp;
                | 🧠 Mem {s['memory_mb']:,}MB &nbsp;
                | ❌ Errors: {s['error_count']}
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 2: Telemetry ───────────────────────────────────────────────────────
    with tab2:
        st.markdown("<div class='section-header'>Runtime Telemetry Time-Series</div>", unsafe_allow_html=True)

        ts = telemetry.get("time_series", [])
        if ts:
            df_ts = pd.DataFrame(ts)

            col_a, col_b = st.columns(2)
            with col_a:
                fig_cpu = go.Figure()
                fig_cpu.add_trace(go.Scatter(
                    x=list(range(len(ts))), y=df_ts["cpu_pct"],
                    fill="tozeroy", line=dict(color="#58a6ff", width=2),
                    fillcolor="rgba(88, 166, 255, 0.1)",
                    name="CPU %",
                ))
                fig_cpu.add_hline(y=85, line=dict(color="#f85149", dash="dash"), annotation_text="85% threshold")
                fig_cpu.update_layout(title="CPU Utilization %", yaxis_title="CPU %", **PLOT_LAYOUT)
                st.plotly_chart(fig_cpu, use_container_width=True)

            with col_b:
                fig_mem = go.Figure()
                fig_mem.add_trace(go.Scatter(
                    x=list(range(len(ts))), y=df_ts["memory_mb"],
                    fill="tozeroy", line=dict(color="#d29922", width=2),
                    fillcolor="rgba(210, 153, 34, 0.1)",
                    name="Memory MB",
                ))
                fig_mem.update_layout(title="Memory Usage (MB)", yaxis_title="Memory (MB)", **PLOT_LAYOUT)
                st.plotly_chart(fig_mem, use_container_width=True)

            col_c, col_d = st.columns(2)
            with col_c:
                fig_tp = go.Figure()
                fig_tp.add_trace(go.Scatter(
                    x=list(range(len(ts))), y=df_ts["throughput_rps"],
                    fill="tozeroy", line=dict(color="#3fb950", width=2),
                    fillcolor="rgba(63, 185, 80, 0.1)",
                ))
                fig_tp.update_layout(title="Throughput (RPS)", yaxis_title="Rows/sec", **PLOT_LAYOUT)
                st.plotly_chart(fig_tp, use_container_width=True)

            with col_d:
                fig_lat = go.Figure()
                fig_lat.add_trace(go.Scatter(
                    x=list(range(len(ts))), y=df_ts["latency_ms"],
                    fill="tozeroy", line=dict(color="#f85149", width=2),
                    fillcolor="rgba(248, 81, 73, 0.1)",
                ))
                fig_lat.update_layout(title="Latency (ms)", yaxis_title="ms", **PLOT_LAYOUT)
                st.plotly_chart(fig_lat, use_container_width=True)

        # Percentile stats
        st.markdown("<div class='section-header'>Latency Percentiles</div>", unsafe_allow_html=True)
        pc1, pc2, pc3 = st.columns(3)
        with pc1: st.metric("P50 Latency", f"{telemetry.get('p50_latency_ms', 0):.1f}ms")
        with pc2: st.metric("P90 Latency", f"{telemetry.get('p90_latency_ms', 0):.1f}ms")
        with pc3: st.metric("P99 Latency", f"{telemetry.get('p99_latency_ms', 0):.1f}ms")

        if telemetry.get("data_skew_detected"):
            st.warning(f"⚠️ **Data Skew Detected** in stage: `{telemetry.get('skew_stage')}`")
        if telemetry.get("quality_issues"):
            st.error("**Data Quality Issues:** " + " | ".join(telemetry["quality_issues"]))

    # ── TAB 3: RCA ─────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("<div class='section-header'>Executive Summary</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='summary-box'>{rca_result.get('executive_summary', '')}</div>", unsafe_allow_html=True)

        col_score, col_violations = st.columns([1, 2])
        with col_score:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=rca_result.get("overall_health_score", 0),
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": f"Health Score (Grade {rca_result.get('health_grade', '?')})", "font": {"color": "#c9d1d9"}},
                delta={"reference": 80},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                    "bar": {"color": "#3fb950" if health >= 75 else "#d29922" if health >= 50 else "#f85149"},
                    "bgcolor": "#161b22",
                    "bordercolor": "#30363d",
                    "steps": [
                        {"range": [0, 45], "color": "#1c1010"},
                        {"range": [45, 75], "color": "#1c1a0a"},
                        {"range": [75, 100], "color": "#0f1c11"},
                    ],
                },
                number={"font": {"color": "#e6edf3", "family": "JetBrains Mono"}},
            ))
            fig_gauge.update_layout(paper_bgcolor="#0d1117", font=dict(color="#c9d1d9"), height=280, margin=dict(t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_violations:
            st.markdown("<div class='section-header'>SLO Violations</div>", unsafe_allow_html=True)
            slo_violations = rca_result.get("slo_violations", [])
            if slo_violations:
                for v in slo_violations:
                    st.markdown(f"""
                    <div class='rca-card'>
                        <b>{v['metric']}</b><br>
                        Actual: <span style='color:#f85149; font-family:JetBrains Mono;'>{v['actual_value']:,}</span>
                        &nbsp;/&nbsp; SLO: <span style='color:#3fb950; font-family:JetBrains Mono;'>{v['slo_threshold']:,}</span>
                        &nbsp; <span class='badge-critical'>+{v['violation_pct']}% over SLO</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ No SLO violations detected")

        st.markdown("<div class='section-header'>Bottlenecks</div>", unsafe_allow_html=True)
        for b in rca_result.get("bottlenecks", []):
            sev = b["severity"]
            badge = f"<span class='badge-critical'>{sev.upper()}</span>" if sev in ("critical","high") else f"<span class='badge-p2'>{sev.upper()}</span>"
            st.markdown(f"""
            <div class='rca-card'>
                {badge} &nbsp; <b>{b['stage']}</b> — {b['issue']}<br>
                <span style='color:#8b949e; font-size:0.82rem;'>↳ {b['impact_description']}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='section-header'>Root Causes</div>", unsafe_allow_html=True)
        for rc in rca_result.get("root_causes", []):
            st.markdown(f"""
            <div class='rca-card' style='border-left-color:#d29922;'>
                <b>🔎 {rc['cause']}</b><br>
                <span style='color:#8b949e; font-size:0.82rem;'>Evidence: {rc['evidence']}</span><br>
                <span style='color:#58a6ff; font-size:0.78rem;'>Affected stages: {', '.join(rc['affected_stages'])}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 4: Recommendations ─────────────────────────────────────────────────
    with tab4:
        est = remediations.get("estimated_improvement", {})
        ei1, ei2, ei3 = st.columns(3)
        with ei1: st.metric("Expected Latency Improvement", f"-{est.get('latency_pct', 0)}%", delta="after tuning")
        with ei2: st.metric("Expected Throughput Gain", f"+{est.get('throughput_pct', 0)}%")
        with ei3: st.metric("Expected Cost Change", f"{est.get('cost_pct', 0)}%")

        st.markdown("<div class='section-header'>Optimization Recommendations</div>", unsafe_allow_html=True)

        for rec in remediations.get("recommendations", []):
            p = rec["priority"]
            badge = f"<span class='badge-p0'>{p}</span>" if p == "P0" else f"<span class='badge-p1'>{p}</span>" if p == "P1" else f"<span class='badge-p2'>{p}</span>"
            effort_color = "#3fb950" if rec["effort"] == "low" else "#d29922" if rec["effort"] == "medium" else "#f85149"
            st.markdown(f"""
            <div class='rec-card'>
                {badge} &nbsp; <b>{rec['title']}</b> &nbsp;
                <span style='color:{effort_color}; font-size:0.75rem;'>● {rec['effort'].upper()} effort</span>
                &nbsp; <span style='color:#3fb950; font-size:0.75rem; font-family:JetBrains Mono;'>+{rec['expected_improvement_pct']}% improvement</span><br>
                <span style='color:#8b949e; font-size:0.82rem;'>↳ {rec['description']}</span><br>
                <span style='font-family:JetBrains Mono; font-size:0.75rem; color:#c9d1d9;'>
                    Before: <span style='color:#f85149;'>{rec['current_value']}</span> &nbsp;→&nbsp;
                    After: <span style='color:#3fb950;'>{rec['recommended_value']}</span>
                </span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='section-header'>Auto-Tuned Configuration</div>", unsafe_allow_html=True)
        auto_config = remediations.get("auto_tuned_config", {})
        config_str = "\n".join([f"{k}={v}" for k, v in auto_config.items()])
        st.code(config_str, language="properties")

        if remediations.get("canary_experiment_suggested"):
            st.info("💡 **Canary experiment recommended** before applying all changes to production.")

    # ── TAB 5: Data & Files ────────────────────────────────────────────────────
    with tab5:
        st.markdown("<div class='section-header'>Generated Data Files</div>", unsafe_allow_html=True)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.metric("Total Rows Generated", f"{data_summary.get('total_rows', 0):,}")
            for name, path in data_summary.get("generated_files", {}).items():
                if os.path.exists(path):
                    df_preview = pd.read_csv(path, nrows=5)
                    st.markdown(f"**{name}.csv** ({data_summary['row_counts'].get(name, 0):,} rows)")
                    st.dataframe(df_preview, use_container_width=True)

        with col_d2:
            st.markdown("**Output Files**")
            output_files = [
                ("outputs/execution_metrics.csv", "Execution Metrics"),
                ("outputs/telemetry_timeseries.csv", "Telemetry Time-Series"),
                ("outputs/rca_bottlenecks.csv", "RCA Bottlenecks"),
                ("outputs/remediation_recommendations.csv", "Remediation Recommendations"),
            ]
            for path, label in output_files:
                if os.path.exists(path):
                    df_out = pd.read_csv(path)
                    st.markdown(f"**{label}**")
                    st.dataframe(df_out, use_container_width=True)

else:
    st.info("👈 Click **Run Pipeline** in the sidebar to start the performance test.")
