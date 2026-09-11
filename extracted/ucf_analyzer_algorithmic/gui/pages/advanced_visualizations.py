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
        st.header("📊 Advanced Visualizations — 25+ Charts")

        if st.session_state.results_df.empty:
            st.info("Run experiments first to populate data.")
            st.stop()

        from sklearn.preprocessing import MinMaxScaler as _MMS3
        from sklearn.decomposition import PCA

        df  = utlis.sanitise_df(st.session_state.results_df.copy())
        num = df.select_dtypes(include=np.number).columns.tolist()
        grp = df.groupby("Model")[num].mean().reset_index()

        st.subheader("1️⃣ Animated Metric Explorer")
        sel_metric = st.selectbox("Choose metric to animate", [c for c in num if c in grp.columns], key="anim_m")
        fig_anim = px.bar(grp.sort_values(sel_metric, ascending=False),
                          x="Model", y=sel_metric, color="Model",
                          title=f"Model Ranking by {sel_metric}",
                          color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig_anim, use_container_width=True)

        st.subheader("2️⃣ Parallel Coordinates — Multi-Metric Overview")
        par_cols = [c for c in ["Accuracy_mean","F1_Score","Precision","Recall",
                                  "Training_Time","Peak_Heap_MB","UCF_mean","Efficiency_Score"] if c in grp.columns]
        if len(par_cols) >= 3:
            grp_enc = grp.copy()
            grp_enc["Model_ID"] = pd.Categorical(grp_enc["Model"]).codes
            fig_par = px.parallel_coordinates(grp_enc, color="Model_ID",
                                               dimensions=par_cols + ["Model_ID"],
                                               color_continuous_scale="viridis",
                                               title="Parallel Coordinates — All Key Metrics")
            st.plotly_chart(fig_par, use_container_width=True)

        st.subheader("3️⃣ Sunburst Chart — Model × Metric Hierarchy")
        sb_rows = []
        sb_cats = {"Classification": ["Accuracy_mean","F1_Score","Precision","Recall"],
                   "UCF":            ["UCF_mean","Efficiency_Score","pS","pD"],
                   "Runtime":        ["Training_Time","Inference_Time","Throughput"],
                   "Memory":         ["Peak_Heap_MB","RAM_Delta_MB","Space_Complexity_MB"]}
        for _, row in grp.iterrows():
            for cat, cols in sb_cats.items():
                for col in cols:
                    if col in row.index and not pd.isna(row[col]) and row[col] > 0:
                        sb_rows.append({"Model": row["Model"], "Category": cat,
                                        "Metric": col, "Value": abs(row[col])})
        if sb_rows:
            sb_df = pd.DataFrame(sb_rows)
            fig_sun = px.sunburst(sb_df, path=["Category","Model","Metric"], values="Value",
                                   color="Category", title="Model Metric Hierarchy — Sunburst")
            st.plotly_chart(fig_sun, use_container_width=True)

            st.subheader("4️⃣ Treemap — Model × Metric Value")
            fig_tree = px.treemap(sb_df, path=["Category","Model","Metric"], values="Value",
                                   color="Value", color_continuous_scale="RdYlGn",
                                   title="Treemap — Metric Value by Model & Category")
            st.plotly_chart(fig_tree, use_container_width=True)

        st.subheader("5️⃣ Violin Plots — Metric Distribution")
        viol_m = st.selectbox("Select metric for violin", [c for c in num if c in df.columns], key="viol_m")
        fig_viol = px.violin(df, x="Model", y=viol_m, color="Model", box=True, points="all",
                              title=f"Violin: Distribution of {viol_m}")
        st.plotly_chart(fig_viol, use_container_width=True)

        st.subheader("6️⃣ Normalised Heatmap — All Models × All Metrics")
        heat_cols = [c for c in num if c in grp.columns]
        if heat_cols:
            heat_df   = grp.set_index("Model")[heat_cols]
            norm_heat = pd.DataFrame(_MMS3().fit_transform(heat_df),
                                      columns=heat_cols, index=heat_df.index)
            fig_heat  = px.imshow(norm_heat.T, text_auto=".2f", aspect="auto",
                                   color_continuous_scale="RdYlGn",
                                   title="Min-Max Normalised Metric Heatmap")
            st.plotly_chart(fig_heat, use_container_width=True)

        st.subheader("7️⃣ 3D Scatter — Accuracy × F1 × Training Time")
        if all(c in grp.columns for c in ["Accuracy_mean","F1_Score","Training_Time"]):
            fig_3d = px.scatter_3d(grp, x="Accuracy_mean", y="F1_Score", z="Training_Time",
                                    color="Model", text="Model", size_max=12,
                                    title="3D: Accuracy × F1 × Training Time")
            st.plotly_chart(fig_3d, use_container_width=True)

        st.subheader("8️⃣ 3D Scatter — BigO × Runtime × Memory")
        big_o_scale2 = {
            "Decision Tree":4,"Random Forest":5,"Logistic Regression":3,
            "K-Nearest Neighbors":1,"Support Vector Machine":5,"Gradient Boosting":5,
            "XGBoost":5,"LightGBM":5,"AdaBoost":5,"Extra Trees":5,
            "Linear Discriminant Analysis":6,"Quadratic Discriminant Analysis":6,
            "MLP Neural Network":6,"Bubble Sort":5
        }
        if "Time_mean" in grp.columns and "Memory_mean" in grp.columns:
            grp_3d = grp.copy()
            grp_3d["BigO_Score"] = grp_3d["Model"].map(big_o_scale2).fillna(4)
            fig_surf = px.scatter_3d(grp_3d, x="BigO_Score", y="Time_mean", z="Memory_mean",
                                      color="Model", text="Model",
                                      title="3D: BigO Complexity × Runtime × Memory")
            st.plotly_chart(fig_surf, use_container_width=True)

        st.subheader("9️⃣ Pearson Correlation Network Heatmap")
        if len(num) >= 4:
            corr_adv = df[num].corr()
            fig_cn = px.imshow(corr_adv, text_auto=".2f", color_continuous_scale="RdBu_r",
                                aspect="auto", title="Full Pearson Correlation Matrix")
            st.plotly_chart(fig_cn, use_container_width=True)

        st.subheader("🔟 Bubble Chart — F1 × Time (size=Memory)")
        if all(c in grp.columns for c in ["F1_Score","Training_Time","Peak_Heap_MB"]):
            fig_bub = px.scatter(grp, x="Training_Time", y="F1_Score",
                                  size="Peak_Heap_MB", color="Model", text="Model",
                                  title="Bubble: F1 vs Training Time (size = Peak Heap MB)")
            fig_bub.update_traces(textposition="top center")
            st.plotly_chart(fig_bub, use_container_width=True)

        st.subheader("1️⃣1️⃣ Stacked Bar — Metric Contributions per Model")
        stack_cols = [c for c in ["Accuracy_mean","Precision","Recall","F1_Score","MCC"] if c in grp.columns]
        if stack_cols:
            fig_stack = px.bar(grp, x="Model", y=stack_cols, barmode="stack",
                                title="Stacked Classification Metrics per Model")
            st.plotly_chart(fig_stack, use_container_width=True)

        st.subheader("1️⃣2️⃣ PCA — Model Clustering in Metric Space")
        pca_cols = [c for c in num if c in grp.columns]
        if len(pca_cols) >= 3 and len(grp) >= 3:
            try:
                pca_data   = grp[pca_cols].fillna(0)
                pca_scaled = _MMS3().fit_transform(pca_data)
                pca        = PCA(n_components=min(3, len(pca_cols), len(grp)))
                coords     = pca.fit_transform(pca_scaled)
                pca_df     = pd.DataFrame(coords[:, :2], columns=["PC1","PC2"])
                pca_df["Model"] = grp["Model"].values
                fig_pca = px.scatter(pca_df, x="PC1", y="PC2", text="Model", color="Model",
                                      title=f"PCA — Models in Metric Space "
                                            f"(Var: {pca.explained_variance_ratio_[:2].sum()*100:.1f}%)")
                fig_pca.update_traces(textposition="top center")
                st.plotly_chart(fig_pca, use_container_width=True)
            except Exception as e:
                st.warning(f"PCA failed: {e}")

        st.subheader("1️⃣3️⃣ Efficiency Frontier — UCF Score vs Accuracy")
        if all(c in grp.columns for c in ["UCF_mean","Accuracy_mean"]):
            fig_ef = px.scatter(grp, x="UCF_mean", y="Accuracy_mean",
                                 text="Model", size="Accuracy_mean", color="Model",
                                 title="UCF Score vs Accuracy — Efficiency Frontier")
            fig_ef.update_traces(textposition="top center")
            st.plotly_chart(fig_ef, use_container_width=True)

        st.subheader("1️⃣4️⃣ Runtime Waterfall — Training vs Inference per Model")
        if all(c in grp.columns for c in ["Training_Time","Inference_Time"]):
            fig_wf = px.bar(grp, x="Model", y=["Training_Time","Inference_Time"],
                             barmode="group", color_discrete_sequence=["#003399","#FF6B35"],
                             title="Training vs Inference Time per Model")
            st.plotly_chart(fig_wf, use_container_width=True)

        st.subheader("1️⃣5️⃣ Scatter Matrix — Cross-Metric Relationships")
        scat_cols = [c for c in ["Accuracy_mean","F1_Score","Training_Time","Peak_Heap_MB","UCF_mean"] if c in df.columns]
        if len(scat_cols) >= 3:
            fig_sm = px.scatter_matrix(df, dimensions=scat_cols, color="Model",
                                        title="Scatter Matrix — Key Metric Relationships")
            st.plotly_chart(fig_sm, use_container_width=True)

        st.subheader("1️⃣6️⃣ Funnel Chart — Throughput Ranking")
        if "Throughput" in grp.columns:
            fig_fun = px.funnel(grp.sort_values("Throughput", ascending=False),
                                 x="Throughput", y="Model",
                                 title="Throughput Funnel — Samples/Second")
            st.plotly_chart(fig_fun, use_container_width=True)

        st.subheader("1️⃣7️⃣ Box Plots — Classification Metrics Distribution")
        clf_cols = [c for c in ["Accuracy_mean","F1_Score","Precision","Recall","MCC"] if c in df.columns]
        for c in clf_cols:
            fig_bx2 = px.box(df, x="Model", y=c, color="Model",
                              title=f"Box: {c} across Models", points="all")
            st.plotly_chart(fig_bx2, use_container_width=True)

        st.subheader("1️⃣8️⃣ Memory Composition Pie Chart")
        mem_model = st.selectbox("Select Model for Memory Pie", grp["Model"].tolist(), key="mem_pie")
        row_m = grp[grp["Model"]==mem_model].iloc[0]
        mem_pie_cols = {c: row_m[c] for c in ["RAM_Delta_MB","Peak_Heap_MB","Model_Size_MB"]
                        if c in row_m.index and row_m[c] > 0}
        if mem_pie_cols:
            fig_mp = px.pie(names=list(mem_pie_cols.keys()), values=list(mem_pie_cols.values()),
                             title=f"Memory Composition — {mem_model}")
            st.plotly_chart(fig_mp, use_container_width=True)

        st.subheader("1️⃣9️⃣ UCF Metrics Radar — All Models")
        ucf_rad = [c for c in ["pS","pD","UCF_mean","TC_mean","R_mean","Efficiency_Score"] if c in grp.columns]
        if len(ucf_rad) >= 3:
            sub_ucf = grp.set_index("Model")[ucf_rad]
            sc_ucf  = pd.DataFrame(_MMS3().fit_transform(sub_ucf), columns=ucf_rad, index=sub_ucf.index)
            fig_ucfr = go.Figure()
            for m in sc_ucf.index:
                fig_ucfr.add_trace(go.Scatterpolar(r=sc_ucf.loc[m].values, theta=ucf_rad,
                                                    fill="toself", name=m))
            fig_ucfr.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                                    showlegend=True, title="UCF Metrics Radar")
            st.plotly_chart(fig_ucfr, use_container_width=True)

        st.subheader("2️⃣0️⃣ Histogram — Metric Value Distribution")
        hist_m = st.selectbox("Select metric for histogram", [c for c in num if c in df.columns], key="hist_m")
        fig_hist = px.histogram(df, x=hist_m, color="Model", nbins=30, barmode="overlay",
                                 title=f"Distribution of {hist_m}")
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("2️⃣1️⃣ Density Heatmap — F1 vs Accuracy")
        if all(c in df.columns for c in ["F1_Score","Accuracy_mean"]):
            fig_den = px.density_heatmap(df, x="Accuracy_mean", y="F1_Score",
                                          nbinsx=20, nbinsy=20,
                                          title="Density Heatmap — F1 vs Accuracy")
            st.plotly_chart(fig_den, use_container_width=True)

        st.subheader("2️⃣2️⃣ Strip Plot — All Numeric Metrics")
        strip_m = st.selectbox("Metric for strip plot", [c for c in num if c in df.columns], key="strip_m")
        fig_strip = px.strip(df, x="Model", y=strip_m, color="Model",
                              title=f"Strip Plot: {strip_m} per Model")
        st.plotly_chart(fig_strip, use_container_width=True)

        st.subheader("2️⃣3️⃣ ECDF — Empirical CDF of Key Metrics")
        ecdf_m = st.selectbox("Metric for ECDF", [c for c in num if c in df.columns], key="ecdf_m")
        fig_ecdf = px.ecdf(df, x=ecdf_m, color="Model", title=f"ECDF: {ecdf_m}")
        st.plotly_chart(fig_ecdf, use_container_width=True)

        st.subheader("2️⃣4️⃣ Classification Metrics Heatmap per Model")
        clf_heat_cols = [c for c in ["Accuracy_mean","F1_Score","Precision","Recall","MCC","Balanced_Accuracy"] if c in grp.columns]
        if clf_heat_cols:
            clf_heat  = grp.set_index("Model")[clf_heat_cols]
            fig_clf_h = px.imshow(clf_heat, text_auto=".3f", color_continuous_scale="RdYlGn",
                                   aspect="auto", title="Classification Metrics Heatmap per Model")
            st.plotly_chart(fig_clf_h, use_container_width=True)

        st.subheader("2️⃣5️⃣ Waterfall — Top Model Score Decomposition")
        rank_cols = [c for c in ["Accuracy_mean","F1_Score","Precision","Recall","Efficiency_Score"] if c in grp.columns]
        if rank_cols and len(grp) > 0:
            top_m_row = grp.loc[grp[rank_cols[0]].idxmax()]
            wf_vals   = [top_m_row[c] for c in rank_cols if c in top_m_row.index]
            fig_wfall = go.Figure(go.Waterfall(
                name="Score", orientation="v",
                x=rank_cols[:len(wf_vals)], y=wf_vals,
                connector={"line": {"color": "rgb(63,63,63)"}}
            ))
            fig_wfall.update_layout(title=f"Score Decomposition — {top_m_row['Model']}")
            st.plotly_chart(fig_wfall, use_container_width=True)

        st.markdown("---")
        csv_adv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download All Data for Custom Visualization", csv_adv, "advanced_viz_data.csv")

    # =====================================================
    # 1️⃣1️⃣ UCF FORMULA LAB
    # =====================================================
