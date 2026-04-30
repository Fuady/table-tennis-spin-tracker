"""
src/api/dashboard.py
──────────────────────────────────────
Streamlit dashboard for:
  1. Live video upload + spin analysis
  2. Historical session analytics
  3. Player profile: spin tendency heatmap
  4. Model performance monitoring

Run:
    streamlit run src/api/dashboard.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yaml

# ─── Page config ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TT Spin Tracker",
    page_icon="🏓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Table_tennis_ball_ITTF.png/200px-Table_tennis_ball_ITTF.png", width=80)
    st.title("🏓 TT Spin Tracker")
    st.markdown("---")

    api_url = st.text_input("API URL", value="http://localhost:8000")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["📤 Analyze Video", "📊 Session Analytics", "🎯 Player Profile", "🤖 Model Monitor"],
    )

    st.markdown("---")

    # API health check
    try:
        resp = requests.get(f"{api_url}/health", timeout=2)
        if resp.status_code == 200:
            health = resp.json()
            st.success(f"✅ API Online")
            if health.get("model_ready"):
                st.success("✅ Model Loaded")
            else:
                st.warning("⚠️ Model not loaded (using mock)")
        else:
            st.error("❌ API Error")
    except Exception:
        st.error("❌ API Offline")
        st.info("Start with: `uvicorn src.api.main:app`")


# ─── Helper ──────────────────────────────────────────────────────────────

SPIN_COLORS = {
    "topspin": "#2196F3",
    "backspin": "#4CAF50",
    "sidespin": "#FF9800",
    "float": "#9C27B0",
}

SPIN_ICONS = {
    "topspin": "🔵",
    "backspin": "🟢",
    "sidespin": "🟠",
    "float": "🟣",
}


def mock_analysis_result(n_rallies: int = 8) -> dict:
    """Generate mock data for demo/testing."""
    spins = ["topspin", "backspin", "sidespin", "float"]
    results = []
    for i in range(n_rallies):
        probs = np.random.dirichlet([2, 2, 1, 1])
        spin = spins[np.argmax(probs)]
        traj = [[np.random.uniform(0, 1), np.random.uniform(0, 1)] for _ in range(20)]
        results.append({
            "rally_id": i + 1,
            "frame_start": i * 60,
            "frame_end": (i + 1) * 60 - 5,
            "spin_type": spin,
            "confidence": float(np.max(probs)),
            "all_probs": {s: float(p) for s, p in zip(spins, probs)},
            "trajectory_points": traj,
            "ball_speed_px_per_frame": np.random.uniform(5, 20),
            "arc_direction": np.random.choice(["concave_down", "concave_up", "neutral"]),
            "lateral_drift": np.random.uniform(0, 30),
        })
    spin_summary = {s: sum(1 for r in results if r["spin_type"] == s) for s in spins}
    return {
        "job_id": "demo",
        "video_filename": "demo_video.mp4",
        "processing_time_sec": 3.7,
        "total_rallies": n_rallies,
        "spin_summary": spin_summary,
        "results": results,
        "model_ready": True,
    }


# ─── Page: Analyze Video ──────────────────────────────────────────────────

if page == "📤 Analyze Video":
    st.title("📤 Analyze Video")
    st.markdown("Upload a table tennis rally video to detect and classify ball spin.")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded = st.file_uploader("Choose a video", type=["mp4", "avi", "mov"])

    with col2:
        demo_mode = st.checkbox("Use demo data (no video needed)", value=True)
        n_mock = st.slider("Mock rallies", 4, 20, 8, disabled=not demo_mode)

    if st.button("🚀 Analyze", type="primary"):
        with st.spinner("Analyzing video..."):
            if demo_mode or uploaded is None:
                time.sleep(1.5)  # simulate processing
                result = mock_analysis_result(n_mock)
            else:
                try:
                    files = {"video": (uploaded.name, uploaded.getvalue(), "video/mp4")}
                    resp = requests.post(f"{api_url}/analyze", files=files, timeout=300)
                    resp.raise_for_status()
                    result = resp.json()
                except Exception as e:
                    st.error(f"API error: {e}")
                    st.stop()

        # Store in session state
        st.session_state["last_result"] = result

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]

        # Summary metrics
        st.markdown("---")
        st.subheader("📊 Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rallies", result["total_rallies"])
        c2.metric("Processing Time", f"{result['processing_time_sec']}s")
        c3.metric("Top Spin", max(result["spin_summary"], key=result["spin_summary"].get))
        c4.metric("Job ID", result["job_id"])

        # Spin distribution pie
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Spin Distribution")
            spin_df = pd.DataFrame([
                {"Spin Type": k.title(), "Count": v, "Color": SPIN_COLORS[k]}
                for k, v in result["spin_summary"].items() if v > 0
            ])
            fig = px.pie(
                spin_df,
                values="Count",
                names="Spin Type",
                color="Spin Type",
                color_discrete_map={k.title(): v for k, v in SPIN_COLORS.items()},
                hole=0.4,
            )
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=280)
            st.plotly_chart(fig, use_container_width=True)

        # Confidence over rallies
        with col_b:
            st.subheader("Confidence per Rally")
            df_res = pd.DataFrame(result["results"])
            fig2 = px.bar(
                df_res,
                x="rally_id",
                y="confidence",
                color="spin_type",
                color_discrete_map=SPIN_COLORS,
                labels={"rally_id": "Rally #", "confidence": "Confidence"},
            )
            fig2.update_layout(margin=dict(t=0, b=0), height=280, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        # Rally table
        st.subheader("Rally Details")
        df_display = df_res[["rally_id", "spin_type", "confidence", "arc_direction",
                              "ball_speed_px_per_frame", "lateral_drift"]].copy()
        df_display["confidence"] = df_display["confidence"].apply(lambda x: f"{x:.1%}")
        df_display["ball_speed_px_per_frame"] = df_display["ball_speed_px_per_frame"].apply(lambda x: f"{x:.1f}")
        df_display["lateral_drift"] = df_display["lateral_drift"].apply(lambda x: f"{x:.1f}")
        df_display.columns = ["Rally", "Spin Type", "Confidence", "Arc", "Speed (px/f)", "Drift (px)"]
        st.dataframe(df_display, use_container_width=True)

        # Trajectory visualization
        st.subheader("Trajectory Visualization")
        selected_rally = st.selectbox("Select rally", options=range(1, len(result["results"]) + 1))
        rally_data = result["results"][selected_rally - 1]

        if rally_data["trajectory_points"]:
            pts = np.array(rally_data["trajectory_points"])
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=pts[:, 0],
                y=1 - pts[:, 1],  # flip y for display
                mode="lines+markers",
                line=dict(color=SPIN_COLORS[rally_data["spin_type"]], width=3),
                marker=dict(size=6, symbol="circle"),
                name="Ball trajectory",
            ))
            # Color gradient by time
            for i in range(len(pts) - 1):
                alpha = i / max(len(pts) - 1, 1)
                fig3.add_trace(go.Scatter(
                    x=[pts[i, 0], pts[i + 1, 0]],
                    y=[1 - pts[i, 1], 1 - pts[i + 1, 1]],
                    mode="lines",
                    line=dict(color=SPIN_COLORS[rally_data["spin_type"]], width=max(1, int(alpha * 5))),
                    showlegend=False,
                ))
            fig3.update_layout(
                xaxis_title="X (normalized)",
                yaxis_title="Y (normalized)",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                height=300,
                title=f"Rally {selected_rally} — {rally_data['spin_type'].title()} (conf: {rally_data['confidence']:.1%})",
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig3, use_container_width=True)


# ─── Page: Session Analytics ──────────────────────────────────────────────

elif page == "📊 Session Analytics":
    st.title("📊 Session Analytics")
    st.markdown("Aggregate statistics across multiple analyzed sessions.")

    # Load features if available
    features_path = Path("data/processed/features.parquet")
    if features_path.exists():
        df = pd.read_parquet(features_path)
        labeled = df[df["spin_int"] >= 0].copy()
    else:
        # Generate synthetic session data
        np.random.seed(42)
        n = 200
        spins = ["topspin", "backspin", "sidespin", "float"]
        labeled = pd.DataFrame({
            "spin_label": np.random.choice(spins, n, p=[0.4, 0.25, 0.2, 0.15]),
            "speed_mean": np.random.uniform(3, 25, n),
            "y_quadratic_coeff": np.random.randn(n) * 0.1,
            "lateral_drift_max": np.abs(np.random.randn(n)) * 15,
            "curvature_mean": np.abs(np.random.randn(n)) * 10,
            "trajectory_length": np.random.randint(10, 50, n),
        })

    if len(labeled) == 0:
        st.warning("No labeled data yet. Analyze some videos first!")
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Trajectories", len(labeled))
    col2.metric("Unique Spin Types", labeled["spin_label"].nunique())
    col3.metric("Most Common", labeled["spin_label"].value_counts().index[0].title())

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Spin Type Distribution")
        vc = labeled["spin_label"].value_counts()
        fig = px.bar(
            x=vc.index.str.title(),
            y=vc.values,
            color=vc.index.str.title(),
            color_discrete_map={k.title(): v for k, v in SPIN_COLORS.items()},
            labels={"x": "Spin Type", "y": "Count"},
        )
        fig.update_layout(showlegend=False, height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Speed Distribution by Spin")
        if "speed_mean" in labeled.columns:
            fig = px.violin(
                labeled,
                x="spin_label",
                y="speed_mean",
                color="spin_label",
                color_discrete_map=SPIN_COLORS,
                box=True,
                points="outliers",
                labels={"spin_label": "Spin Type", "speed_mean": "Mean Speed (px/frame)"},
            )
            fig.update_layout(showlegend=False, height=300, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    # Feature scatter
    st.subheader("Feature Scatter — Curvature vs Lateral Drift")
    if "curvature_mean" in labeled.columns and "lateral_drift_max" in labeled.columns:
        fig = px.scatter(
            labeled,
            x="curvature_mean",
            y="lateral_drift_max",
            color="spin_label",
            color_discrete_map=SPIN_COLORS,
            opacity=0.6,
            size_max=8,
            labels={
                "curvature_mean": "Mean Curvature (px)",
                "lateral_drift_max": "Max Lateral Drift (px)",
                "spin_label": "Spin Type",
            },
        )
        fig.update_layout(height=350, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)


# ─── Page: Player Profile ─────────────────────────────────────────────────

elif page == "🎯 Player Profile":
    st.title("🎯 Player Profile")
    st.markdown("Analyze a player's spin tendencies across serves and rallies.")

    player_name = st.text_input("Player name", value="Player A")

    # Simulate player data
    np.random.seed(99)
    n_serves = 50
    spins = ["topspin", "backspin", "sidespin", "float"]
    player_data = pd.DataFrame({
        "serve_id": range(1, n_serves + 1),
        "spin": np.random.choice(spins, n_serves, p=[0.45, 0.30, 0.15, 0.10]),
        "speed": np.random.uniform(8, 22, n_serves),
        "score": np.random.choice([0, 1], n_serves, p=[0.35, 0.65]),
        "x_pos": np.random.uniform(0.1, 0.9, n_serves),
        "y_pos": np.random.uniform(0.3, 0.7, n_serves),
    })

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Serves Analyzed", len(player_data))
    c2.metric("Win Rate", f"{player_data['score'].mean():.1%}")
    dominant = player_data["spin"].value_counts().index[0]
    c3.metric("Dominant Spin", dominant.title())

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Serve Tendency Map")
        fig = px.scatter(
            player_data,
            x="x_pos",
            y="y_pos",
            color="spin",
            symbol="spin",
            color_discrete_map=SPIN_COLORS,
            size=[8] * len(player_data),
            labels={"x_pos": "Table Width", "y_pos": "Table Depth"},
            opacity=0.8,
        )
        fig.update_xaxes(range=[0, 1])
        fig.update_yaxes(range=[0, 1])
        fig.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Win Rate by Spin Type")
        wr = player_data.groupby("spin")["score"].agg(["mean", "count"]).reset_index()
        wr.columns = ["spin", "win_rate", "n"]
        wr["spin_title"] = wr["spin"].str.title()
        fig = px.bar(
            wr,
            x="spin_title",
            y="win_rate",
            color="spin_title",
            color_discrete_map={k.title(): v for k, v in SPIN_COLORS.items()},
            text=wr["win_rate"].apply(lambda x: f"{x:.1%}"),
            labels={"spin_title": "Spin Type", "win_rate": "Win Rate"},
        )
        fig.update_traces(textposition="outside")
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        fig.update_layout(showlegend=False, height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)


# ─── Page: Model Monitor ──────────────────────────────────────────────────

elif page == "🤖 Model Monitor":
    st.title("🤖 Model Performance Monitor")

    eval_report = Path("reports/evaluation/evaluation_report.json")
    if eval_report.exists():
        with open(eval_report) as f:
            report = json.load(f)

        c1, c2, c3 = st.columns(3)
        c1.metric("Test Accuracy", f"{report['accuracy']:.2%}")
        c2.metric("Macro F1", f"{report['macro_f1']:.4f}")
        speed = report.get("inference_speed", {})
        c3.metric("Latency (P95)", f"{speed.get('p95_ms', '?')} ms")

        st.markdown("---")
        st.subheader("Confusion Matrix")

        cm = np.array(report["confusion_matrix"])
        classes = report["classes"]
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-6)

        fig = px.imshow(
            cm_norm,
            x=[c.title() for c in classes],
            y=[c.title() for c in classes],
            color_continuous_scale="Blues",
            text_auto=".1%",
            labels={"x": "Predicted", "y": "Actual", "color": "Rate"},
        )
        fig.update_layout(height=400, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Full Classification Report")
        st.code(report.get("classification_report", "Not available"))

        # Check for saved plots
        for plot_name, title in [
            ("roc_curves.png", "ROC Curves"),
            ("feature_distributions.png", "Feature Distributions"),
        ]:
            plot_path = Path("reports/evaluation") / plot_name
            if plot_path.exists():
                st.subheader(title)
                st.image(str(plot_path))

    else:
        st.info("No evaluation report found. Run evaluation first:")
        st.code("python src/modeling/evaluate.py --model-path models/spin_classifier_best.pt")

    st.markdown("---")
    st.subheader("MLflow Experiments")
    st.info("Track experiments in MLflow UI:")
    st.code("mlflow ui --host 0.0.0.0 --port 5000\n# Then open: http://localhost:5000")
