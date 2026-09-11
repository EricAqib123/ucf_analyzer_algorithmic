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

        st.header("📊 UCF Metrics vs Big-O Complexity — Full Analysis")

        if st.session_state.results_df.empty:
            st.info("Run experiments first.")
            st.stop()

        from scipy.stats import pearsonr, spearmanr
        from sklearn.preprocessing import MinMaxScaler

        df=utlis.sanitise_df(st.session_state.results_df.copy())
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        # ─────────────────────────────────────────────
        # Big-O reference table
        # ─────────────────────────────────────────────
        big_o_data = {
            "Model": [
                "Decision Tree","Random Forest","Logistic Regression",
                "K-Nearest Neighbors","Support Vector Machine","Gradient Boosting",
                "XGBoost","LightGBM","AdaBoost","Extra Trees",
                "Linear Discriminant Analysis","Quadratic Discriminant Analysis",
                "MLP Neural Network","Bubble Sort"
            ],
            "Training_Complexity": [
                "O(p·nlogn)","O(ntrees·p·nlogn)","O(n·p)","O(1)",
                "O(n·p) to O(n²·p)","O(ntrees·p·nlogn)","O(ntrees·p·nlogn)",
                "O(ntrees·p·nlogn)","O(ntrees·n·p)","O(ntrees·p·n)",
                "O(n·p² + p³)","O(n·p² + p³)","O(n·p·h)","O(n²)"
            ],
            "Prediction_Complexity": [
                "O(depth)","O(ntrees·depth)","O(p)","O(n·p)","O(p)",
                "O(ntrees·depth)","O(ntrees·depth)","O(ntrees·depth)",
                "O(ntrees·depth)","O(ntrees·depth)","O(p)","O(p²)",
                "O(p·h)","O(n²)"
            ],
            "Space_Complexity": [
                "O(ntrees·depth)","O(ntrees·n)","O(p)","O(n·p)","O(n·p)",
                "O(ntrees·depth)","O(ntrees·depth)","O(ntrees·depth)",
                "O(ntrees)","O(ntrees·depth)","O(p²)","O(k·p²)",
                "O(p·h)","O(1)"
            ]
        }
        big_o_df = pd.DataFrame(big_o_data)
        df       = pd.merge(df, big_o_df, on="Model", how="left")

        def big_o_to_score(c):
            if pd.isna(c): return 4
            c = c.lower()
            if "o(1)" in c:             return 1
            elif "nlogn" in c:          return 4
            elif "log" in c:            return 2
            elif "n·p" in c or "n*p" in c: return 3
            elif "n²" in c or "n^2" in c:  return 5
            elif "p³" in c or "n³" in c:   return 6
            else:                       return 4

        def complexity_label(s):
            return {1:"O(1)",2:"O(log n)",3:"O(n·p)",
                    4:"O(n log n)",5:"O(n²)",6:"O(n³)"}.get(s,"O(?)")

        df["BigO_Score"]      = df["Training_Complexity"].apply(big_o_to_score)
        df["Space_BigO_Score"]= df["Space_Complexity"].apply(big_o_to_score)
        df["Complexity_Level"]= df["BigO_Score"].apply(
            lambda s: "Low" if s<=2 else ("Medium" if s<=4 else "High"))

        metrics_ext = ["Time_mean","Memory_mean","BigO_Score","Space_BigO_Score",
                       "Accuracy_mean","F1_Score","UCF_mean","TC_mean",
                       "R_mean","Efficiency_Score","Training_Time","Inference_Time",
                       "Throughput","CPU_Usage_pct","Peak_Heap_MB","Space_Complexity_MB"]
        avail_ext   = [m for m in metrics_ext if m in df.columns]
        grouped     = df.groupby("Model")[avail_ext].mean().reset_index()

        # ── Big-O reference table ──
        st.markdown("---")
        st.subheader("📐 Big-O Complexity Reference Table")
        st.dataframe(big_o_df, use_container_width=True)

        # ── Model complexity summary ──
        st.markdown("---")
        st.subheader("📋 Model Complexity Summary")
        st.dataframe(utlis.safe_gradient(utlis.sanitise_df(grouped).style, cmap="RdYlGn_r",
                          subset=[c for c in avail_ext if c in grouped.columns]),
            use_container_width=True)

        # ── UCF vs BigO scatter ──
        st.markdown("---")
        st.subheader("🔬 UCF Score vs Big-O Complexity")
        if "UCF_mean" in grouped.columns:
            fig_ucf_bo = px.scatter(
                grouped, x="BigO_Score", y="UCF_mean",
                text="Model", size="UCF_mean", color="Model",
                title="UCF Score vs Theoretical Big-O (lower UCF = more complex in practice)",
                labels={"BigO_Score":"Big-O Score (1=O(1) … 6=O(n³))","UCF_mean":"f(u) UCF Score"})
            fig_ucf_bo.update_traces(textposition="top center")
            st.plotly_chart(fig_ucf_bo, use_container_width=True)

        # ── TC vs BigO ──
        if "TC_mean" in grouped.columns:
            fig_tc = px.scatter(
                grouped, x="BigO_Score", y="TC_mean",
                text="Model", size="TC_mean", color="Model",
                title="Total Complexity TC(A,D) vs Big-O Score")
            fig_tc.update_traces(textposition="top center")
            st.plotly_chart(fig_tc, use_container_width=True)

        # ── Remainder R vs BigO ──
        if "R_mean" in grouped.columns:
            fig_r = px.scatter(
                grouped, x="BigO_Score", y="R_mean",
                text="Model", size="R_mean", color="Model",
                title="Remainder R vs Big-O Score")
            fig_r.update_traces(textposition="top center")
            st.plotly_chart(fig_r, use_container_width=True)

        # ── Runtime vs BigO ──
        st.markdown("---")
        st.subheader("⚡ Runtime vs Big-O & Space Complexity")
        if "Time_mean" in grouped.columns:
            fig_rt = px.scatter(
                grouped, x="BigO_Score", y="Time_mean",
                text="Model", size="Time_mean", color="Model",
                title="Measured Runtime vs Training Big-O Score")
            fig_rt.update_traces(textposition="top center")
            st.plotly_chart(fig_rt, use_container_width=True)

        if "Training_Time" in grouped.columns:
            fig_tr = px.scatter(
                grouped, x="BigO_Score", y="Training_Time",
                text="Model", size="Training_Time", color="Model",
                title="Training Time vs Big-O Score")
            fig_tr.update_traces(textposition="top center")
            st.plotly_chart(fig_tr, use_container_width=True)

        if "Inference_Time" in grouped.columns:
            fig_inf = px.scatter(
                grouped, x="BigO_Score", y="Inference_Time",
                text="Model", size="Inference_Time", color="Model",
                title="Inference Time vs Prediction Big-O")
            fig_inf.update_traces(textposition="top center")
            st.plotly_chart(fig_inf, use_container_width=True)

        # ── Memory vs BigO ──
        st.markdown("---")
        st.subheader("🗂️ Memory & Space vs Big-O")
        if "Memory_mean" in grouped.columns:
            fig_mem = px.scatter(
                grouped, x="BigO_Score", y="Memory_mean",
                text="Model", size="Memory_mean", color="Model",
                title="Peak Heap Memory vs Training Big-O Score")
            fig_mem.update_traces(textposition="top center")
            st.plotly_chart(fig_mem, use_container_width=True)

        if "Space_Complexity_MB" in grouped.columns:
            fig_sp = px.scatter(
                grouped, x="Space_BigO_Score", y="Space_Complexity_MB",
                text="Model", size="Space_Complexity_MB", color="Model",
                title="Practical Space Complexity (MB) vs Theoretical Space Big-O")
            fig_sp.update_traces(textposition="top center")
            st.plotly_chart(fig_sp, use_container_width=True)

        # ── Accuracy & Efficiency vs BigO ──
        st.markdown("---")
        st.subheader("📊 Accuracy & Efficiency vs Big-O")
        if "Accuracy_mean" in grouped.columns:
            fig_acc = px.scatter(
                grouped, x="BigO_Score", y="Accuracy_mean",
                text="Model", size="Accuracy_mean", color="Model",
                title="Accuracy vs Big-O Score (bigger bubble = more accurate)")
            fig_acc.update_traces(textposition="top center")
            st.plotly_chart(fig_acc, use_container_width=True)

        if "Efficiency_Score" in grouped.columns:
            fig_eff = px.scatter(
                grouped, x="BigO_Score", y="Efficiency_Score",
                text="Model", size="Efficiency_Score", color="Model",
                title="UCF Efficiency Score vs Big-O Score")
            fig_eff.update_traces(textposition="top center")
            st.plotly_chart(fig_eff, use_container_width=True)

        # ── Pareto: Accuracy vs Runtime ──
        st.markdown("---")
        st.subheader("⚡ Pareto Frontier: Accuracy vs Runtime")
        if "Accuracy_mean" in grouped.columns and "Time_mean" in grouped.columns:
            fig_par = px.scatter(
                grouped, x="Time_mean", y="Accuracy_mean",
                text="Model", size="Accuracy_mean", color="Model",
                title="Accuracy vs Runtime Trade-off")
            fig_par.update_traces(textposition="top center")
            st.plotly_chart(fig_par, use_container_width=True)

            pareto = []
            for i, ri in grouped.iterrows():
                dominated = any(
                    rj["Accuracy_mean"] >= ri["Accuracy_mean"] and
                    rj["Time_mean"]     <= ri["Time_mean"] and
                    (rj["Accuracy_mean"] > ri["Accuracy_mean"] or rj["Time_mean"] < ri["Time_mean"])
                    for j, rj in grouped.iterrows()
                )
                if not dominated:
                    pareto.append(ri["Model"])
            st.success(f"🏆 Pareto Optimal Models: {', '.join(pareto)}")

        # ── Pareto: Efficiency vs Space ──
        if "Efficiency_Score" in grouped.columns and "Space_Complexity_MB" in grouped.columns:
            st.subheader("⚡ Pareto Frontier: Efficiency vs Space Complexity")
            fig_par2 = px.scatter(
                grouped, x="Space_Complexity_MB", y="Efficiency_Score",
                text="Model", size="Efficiency_Score", color="Model",
                title="Efficiency vs Space Complexity Trade-off")
            fig_par2.update_traces(textposition="top center")
            st.plotly_chart(fig_par2, use_container_width=True)

        # ── Theory vs Practical gap ──
        st.markdown("---")
        st.subheader("📉 Theory vs Practical Gap")
        if "Time_mean" in df.columns:
            df["Time_per_BigO"]  = df["Time_mean"]  / df["BigO_Score"].replace(0,1)
            df["Mem_per_BigO"]   = df["Memory_mean"] / df["BigO_Score"].replace(0,1)
            gap_df = df.groupby("Model")[["Time_per_BigO","Mem_per_BigO"]].mean().reset_index()
            fig_gap = px.bar(gap_df, x="Model", y=["Time_per_BigO","Mem_per_BigO"],
                             barmode="group",
                             title="Practical Time & Memory per Big-O Unit")
            st.plotly_chart(fig_gap, use_container_width=True)

        # ── Correlation: BigO vs practical ──
        st.markdown("---")
        st.subheader("📐 Statistical Correlation: Big-O Score vs Practical Metrics")
        corr_target = st.selectbox("Correlate BigO_Score with:",
                                    [m for m in ["Time_mean","Memory_mean","Training_Time",
                                                 "Inference_Time","UCF_mean","TC_mean"]
                                     if m in df.columns],
                                    key="bigo_corr_target")
        valid = df[["BigO_Score", corr_target]].dropna()
        if len(valid) >= 3:
            pr, pp = pearsonr(valid["BigO_Score"], valid[corr_target])
            sr, sp = spearmanr(valid["BigO_Score"], valid[corr_target])
            cc1,cc2,cc3,cc4 = st.columns(4)
            cc1.metric("Pearson r",   f"{pr:.4f}")
            cc2.metric("Pearson p",   f"{pp:.4f}")
            cc3.metric("Spearman ρ",  f"{sr:.4f}")
            cc4.metric("Spearman p",  f"{sp:.4f}")
            if pp < 0.05:
                st.success(f"✅ Significant Pearson correlation between BigO_Score and {corr_target}")
            else:
                st.info(f"ℹ️ No significant Pearson correlation detected")

        # ── Full radar: UCF + BigO ──
        st.markdown("---")
        st.subheader("🕸️ Radar: UCF + Big-O Combined View")
        radar_m = [m for m in ["UCF_mean","TC_mean","R_mean","Efficiency_Score",
                                 "Accuracy_mean","F1_Score","BigO_Score"]
                   if m in grouped.columns]
        if len(radar_m) >= 3:
            scaler  = MinMaxScaler()
            sub     = grouped.set_index("Model")[radar_m]
            scaled  = pd.DataFrame(scaler.fit_transform(sub),
                                    columns=radar_m, index=sub.index)
            fig_rad = go.Figure()
            for m in scaled.index:
                fig_rad.add_trace(go.Scatterpolar(
                    r=scaled.loc[m].values, theta=radar_m,
                    fill="toself", name=m))
            fig_rad.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                showlegend=True, title="Normalised UCF + Big-O Radar")
            st.plotly_chart(fig_rad, use_container_width=True)

        # ── Download ──
        st.markdown("---")
        csv = grouped.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download UCF + Big-O Metrics CSV", csv, "ucf_bigo_full.csv")

    # =====================================================
    # 7️⃣ EXPLAINABLE AI
    # =====================================================
