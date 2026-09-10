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
    decision_tree_model, random_forest_model, logistic_regression_model,
    knn_model, svm_model, gradient_boosting_model, xgboost_model,
    lightgbm_model, adaboost_model, extra_trees_model, lda_model,
    qda_model, mlp_model,
)
from algorithms.dataset_loader import generate_dataset
from algorithms.cic_loader import load_cic_ids
from gui import utlis


def render():

        st.header("📈 Performance Metrics — Full Analysis")

        if st.session_state.results_df.empty:
            st.info("Run analyses first.")
            st.stop()

        from sklearn.preprocessing import MinMaxScaler
        from scipy.stats import f_oneway, kruskal

        df= utlis.sanitise_df(st.session_state.results_df.copy())
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        grouped      = df.groupby("Model")[numeric_cols].mean().reset_index()

        # ── Summary statistics ──
        st.markdown("---")
        st.subheader("📋 Full Statistical Summary")
        st.dataframe(utlis.safe_gradient(utlis.sanitise_df(df).describe().T.style, cmap="Blues"),
                     use_container_width=True)

        # ── Section helper ──
        def multi_bar(title, metrics, df_src, color_scale="teal"):
            avail = [m for m in metrics if m in df_src.columns]
            if not avail:
                st.info(f"No data for: {title}")
                return
            fig = px.bar(df_src, x="Model", y=avail, barmode="group",
                         title=title, color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)

        # ── UCF Metrics ──
        st.markdown("---")
        st.subheader("🔬 UCF Metrics per Model")
        multi_bar("UCF Core Metrics",
                  ["pS","pD","UCF_mean","Efficiency_Score"], grouped)
        multi_bar("UCF Complexity Metrics",
                  ["TC_mean","R_mean","Time_mean","Time_std"], grouped)

        # ── Classification ──
        st.markdown("---")
        st.subheader("📊 Classification Metrics per Model")
        multi_bar("Accuracy, F1, Precision, Recall",
                  ["Accuracy_mean","F1_Score","F1_mean","Precision","Recall"], grouped)
        multi_bar("Advanced Classification Metrics",
                  ["MCC","Balanced_Accuracy","Cohen_Kappa","ROC_AUC"], grouped)

        # ── Runtime ──
        st.markdown("---")
        st.subheader("⚙️ Runtime & CPU Metrics per Model")
        multi_bar("Training & Inference Time",
                  ["Training_Time","Inference_Time","Time_mean"], grouped)

        if "Throughput" in grouped.columns:
            fig_tp = px.bar(grouped, x="Model", y="Throughput", color="Model",
                            title="Throughput (samples / second)")
            st.plotly_chart(fig_tp, use_container_width=True)

        if "CPU_Usage_pct" in grouped.columns:
            fig_cpu = px.bar(grouped, x="Model", y="CPU_Usage_pct", color="Model",
                             title="CPU Usage (%) per Model")
            st.plotly_chart(fig_cpu, use_container_width=True)

        # ── Memory / Space ──
        st.markdown("---")
        st.subheader("🗂️ Memory & Space Complexity per Model")
        multi_bar("Memory Metrics",
                  ["Memory_mean","RAM_Delta_MB","Peak_Heap_MB",
                   "Model_Size_MB","Space_Complexity_MB"], grouped)

        # ── Box plots for distribution ──
        st.markdown("---")
        st.subheader("📦 Distribution Box Plots")
        box_metric = st.selectbox("Select metric for distribution view", numeric_cols,
                                   key="perf_box_metric")
        fig_box = px.box(df, x="Model", y=box_metric, color="Model",
                         title=f"Distribution of {box_metric} across Models")
        st.plotly_chart(fig_box, use_container_width=True)

        # ── Correlation heatmap ──
        st.markdown("---")
        st.subheader("🔗 Metric Correlation Heatmap")
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig_corr = px.imshow(corr, text_auto=".2f",
                                  color_continuous_scale="RdBu_r",
                                  title="Pearson Correlation Matrix — All Metrics")
            st.plotly_chart(fig_corr, use_container_width=True)

        # ── Normalised radar ──
        st.markdown("---")
        st.subheader("🕸️ Normalised Performance Radar")
        radar_cols = [m for m in ["Accuracy_mean","F1_Score","Precision","Recall",
                                   "Balanced_Accuracy","Efficiency_Score","Throughput"]
                      if m in grouped.columns]
        if len(radar_cols) >= 3:
            scaler   = MinMaxScaler()
            sub      = grouped.set_index("Model")[radar_cols]
            scaled   = pd.DataFrame(scaler.fit_transform(sub),
                                     columns=radar_cols, index=sub.index)
            fig_rad  = go.Figure()
            for m in scaled.index:
                fig_rad.add_trace(go.Scatterpolar(
                    r=scaled.loc[m].values, theta=radar_cols,
                    fill="toself", name=m))
            fig_rad.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                showlegend=True, title="Normalised Multi-Metric Radar")
            st.plotly_chart(fig_rad, use_container_width=True)

        # ── ANOVA & Kruskal-Wallis ──
        st.markdown("---")
        st.subheader("📐 Statistical Tests Across All Models")
        stat_m = st.selectbox("Select metric", numeric_cols, key="perf_stat_metric")
        groups = [df[df["Model"]==m][stat_m].dropna()
                  for m in df["Model"].unique()
                  if len(df[df["Model"]==m][stat_m].dropna()) >= 2]
        if len(groups) >= 2:
            f_stat, f_p = f_oneway(*groups)
            try:
                k_stat, k_p = kruskal(*groups)
            except Exception:
                k_stat, k_p = float("nan"), float("nan")

            tc1,tc2,tc3,tc4 = st.columns(4)
            tc1.metric("ANOVA F-statistic", f"{f_stat:.4f}")
            tc2.metric("ANOVA p-value",     f"{f_p:.4f}")
            tc3.metric("Kruskal-Wallis H",  f"{k_stat:.4f}")
            tc4.metric("Kruskal p-value",   f"{k_p:.4f}")

            if f_p < 0.05:
                st.success("✅ ANOVA: Significant difference among models (p < 0.05)")
            else:
                st.info("ℹ️ ANOVA: No significant difference (p ≥ 0.05)")
            if k_p < 0.05:
                st.success("✅ Kruskal-Wallis: Significant difference (p < 0.05)")
            else:
                st.info("ℹ️ Kruskal-Wallis: No significant difference (p ≥ 0.05)")
        else:
            st.warning("Need at least 2 models with 2+ runs each for statistical tests.")

        # ── Download ──
        st.markdown("---")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Full Metrics CSV", csv, "performance_metrics_full.csv")

    # =====================================================
    # 6️⃣ UCF VS BIG-O ANALYSIS
    # =====================================================
