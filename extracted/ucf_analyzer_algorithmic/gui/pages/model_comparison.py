import datetime
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
from sklearn.preprocessing import LabelEncoder

from algorithms.storing import bubble_sort, generate_data
from experiment.experiment_runner import run_experiment, run_ml_experiment
from algorithms.ml_models import (
    decision_tree_model,
    random_forest_model,
    logistic_regression_model,
    knn_model,
    svm_model,
    gradient_boosting_model,
    xgboost_model,
    lightgbm_model,
    adaboost_model,
    extra_trees_model,
    lda_model,
    qda_model,
    mlp_model,
)
from algorithms.dataset_loader import generate_dataset
from algorithms.cic_loader import load_cic_ids
from gui import utlis


def render():

        st.header("🔬 Compare Two Models")

        if st.session_state.results_df.empty:
            st.info("Run analyses first.")
            st.stop()

        from scipy.stats import ttest_ind, mannwhitneyu

        df = utlis.sanitise_df(st.session_state.results_df.copy())
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols) == 0:
            st.warning("No numeric metrics available.")
            st.stop()

        grouped = df.groupby("Model")[numeric_cols].mean()
        models  = grouped.index.tolist()

        model_a = st.selectbox("Select Model A", models, key="model_a")
        model_b = st.selectbox("Select Model B", models, key="model_b")

        if model_a == model_b:
            st.warning("Please select two different models.")
            st.stop()

        comp_df = grouped.loc[[model_a, model_b]].T
        comp_df.columns = [model_a, model_b]
        comp_df["Δ (A - B)"] = comp_df[model_a] - comp_df[model_b]
        comp_df["Winner"]    = comp_df.apply(
            lambda row: model_a if row[model_a] >= row[model_b] else model_b, axis=1)

        # ── Full metrics table ──
        st.markdown("---")
        st.subheader(f"📊 {model_a} vs {model_b} — Full Metrics Table")
        st.dataframe(utlis.safe_gradient(utlis.sanitise_df(comp_df).style, cmap="RdYlGn",
                      subset=[model_a, model_b]), use_container_width=True)

        # ── Grouped metric category charts ──
        def cat_bar(title, metrics):
            avail = [m for m in metrics if m in comp_df.index]
            if not avail:
                return
            sub = comp_df.loc[avail, [model_a, model_b]].reset_index()
            sub.columns = ["Metric", model_a, model_b]
            fig = px.bar(sub, x="Metric", y=[model_a, model_b],
                         barmode="group", title=title,
                         color_discrete_sequence=["#636EFA","#EF553B"])
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("🔬 UCF Metrics Comparison")
        cat_bar("UCF Core", ["pS","pD","UCF_mean","TC_mean","R_mean","Efficiency_Score"])

        st.markdown("---")
        st.subheader("📊 Classification Metrics Comparison")
        cat_bar("Classification", ["Accuracy_mean","F1_Score","F1_mean","Precision",
                                    "Recall","MCC","Balanced_Accuracy","Cohen_Kappa","ROC_AUC"])

        st.markdown("---")
        st.subheader("⚙️ Runtime & System Metrics Comparison")
        cat_bar("Runtime", ["Training_Time","Inference_Time","Throughput",
                             "CPU_Usage_pct","Time_mean","Time_std"])

        st.markdown("---")
        st.subheader("🗂️ Memory & Space Complexity Comparison")
        cat_bar("Memory", ["Memory_mean","RAM_Delta_MB","Peak_Heap_MB",
                            "Model_Size_MB","Space_Complexity_MB"])

        # ── Radar chart ──
        st.markdown("---")
        st.subheader("🕸️ Radar Comparison")
        from sklearn.preprocessing import MinMaxScaler
        radar_metrics = [m for m in ["Accuracy_mean","F1_Score","Precision","Recall",
                                      "MCC","Balanced_Accuracy","Efficiency_Score"]
                         if m in grouped.columns]
        if len(radar_metrics) >= 3:
            scaler    = MinMaxScaler()
            sub_g     = grouped.loc[[model_a, model_b], radar_metrics]
            scaled    = scaler.fit_transform(sub_g)
            scaled_df = pd.DataFrame(scaled, columns=radar_metrics, index=[model_a, model_b])
            fig_r = go.Figure()
            for m in [model_a, model_b]:
                fig_r.add_trace(go.Scatterpolar(
                    r=scaled_df.loc[m].values, theta=radar_metrics,
                    fill="toself", name=m))
            fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                                 showlegend=True)
            st.plotly_chart(fig_r, use_container_width=True)

        # ── Statistical significance ──
        st.markdown("---")
        st.subheader("📐 Statistical Significance Tests")
        stat_metric = st.selectbox("Select metric for significance testing", numeric_cols,
                                    key="stat_metric_comp")
        d1 = df[df["Model"]==model_a][stat_metric].dropna()
        d2 = df[df["Model"]==model_b][stat_metric].dropna()

        if len(d1) >= 2 and len(d2) >= 2:
            t_stat, t_p = ttest_ind(d1, d2, equal_var=False)
            try:
                u_stat, u_p = mannwhitneyu(d1, d2, alternative="two-sided")
            except Exception:
                u_stat, u_p = float("nan"), float("nan")

            tc1, tc2 = st.columns(2)
            tc1.metric("Welch t-statistic", f"{t_stat:.4f}")
            tc2.metric("t-test p-value",    f"{t_p:.4f}")
            uc1, uc2 = st.columns(2)
            uc1.metric("Mann-Whitney U",    f"{u_stat:.1f}")
            uc2.metric("MWU p-value",       f"{u_p:.4f}")

            if t_p < 0.05:
                st.success("✅ Statistically significant difference (p < 0.05)")
            else:
                st.info("ℹ️ No significant difference detected (p ≥ 0.05)")
        else:
            st.warning("Need at least 2 runs per model for significance testing.")

        # ── Download ──
        st.markdown("---")
        csv = comp_df.to_csv().encode("utf-8")
        st.download_button("⬇️ Download Comparison CSV", csv, "model_comparison.csv")

    # =====================================================
    # 5️⃣ PERFORMANCE METRICS
    # =====================================================
