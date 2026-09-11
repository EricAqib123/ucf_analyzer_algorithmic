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

        st.header("UCF Intelligent Dashboard")

        if st.session_state.results_df.empty:
            st.info("Run analyses first.")
            st.stop()

        from scipy.stats import (ttest_ind, f_oneway, kruskal, mannwhitneyu,
                                  levene, shapiro, pearsonr, spearmanr, kendalltau)
        from sklearn.preprocessing import MinMaxScaler

        df= utlis.sanitise_df(st.session_state.results_df.copy())
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols) == 0:
            st.warning("No numeric columns available.")
            st.stop()

        grouped_all  = df.groupby("Model")[numeric_cols].mean()
        models_list  = df["Model"].unique().tolist()

        # ─────────────────────────────────────────────────────────
        # SECTION 0 — Full Data Tables
        # ─────────────────────────────────────────────────────────
        compare_all = st.checkbox("Show Full Experiment Results")
        if compare_all:
            st.markdown("---")
            st.subheader("📋 All Raw Experiment Rows")
            st.dataframe(df, use_container_width=True)

            st.subheader("📈 Full Statistical Summary (all metrics)")
            st.dataframe(utlis.safe_gradient(utlis.sanitise_df(df).describe().T.style, cmap="Blues"),
                         use_container_width=True)

            time_cols = [c for c in ["Time_mean","Training_Time","Inference_Time"] if c in df.columns]
            if time_cols:
                st.plotly_chart(px.bar(df, x="Model", y=time_cols, barmode="group",
                                       title="Execution & Training Time"),
                                use_container_width=True)

            st.subheader("🔥 Heatmap — All Raw Rows")
            st.plotly_chart(px.imshow(df[numeric_cols], text_auto=True, aspect="auto",
                                       color_continuous_scale="RdYlGn_r"),
                            use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Model-wise Average — All Metrics")
        st.dataframe(utlis.safe_gradient(utlis.sanitise_df(grouped_all).style, cmap="RdYlGn_r"),
                     use_container_width=True)

        st.subheader("🔥 Heatmap — Model Averages")
        st.plotly_chart(px.imshow(grouped_all, text_auto=True, aspect="auto",
                                   color_continuous_scale="RdYlGn_r",
                                   title="Model-wise Average Metrics Heatmap"),
                        use_container_width=True)

        # ─────────────────────────────────────────────────────────
        # SECTION 1 — Distribution Box Plots (every metric)
        # ─────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📦 Distribution Box Plots — All Metrics")
        st.caption("One box plot per metric showing spread across models.")

        # Group metrics into categories for organised display
        metric_groups = {
            "🔬 UCF Metrics":         [c for c in ["pS","pD","UCF_mean","TC_mean","R_mean",
                                                    "Efficiency_Score","Time_mean","Time_std",
                                                    "Memory_mean"] if c in numeric_cols],
            "📊 Classification":      [c for c in ["Accuracy_mean","F1_Score","F1_mean",
                                                    "Precision","Recall","MCC",
                                                    "Balanced_Accuracy","Cohen_Kappa",
                                                    "ROC_AUC"] if c in numeric_cols],
            "⚙️ Runtime & CPU":       [c for c in ["Training_Time","Inference_Time",
                                                    "Throughput","CPU_Usage_pct"] if c in numeric_cols],
            "🗂️ Memory & Space":      [c for c in ["RAM_Before_MB","RAM_After_MB","RAM_Delta_MB",
                                                    "Peak_Heap_MB","Model_Size_MB",
                                                    "Space_Complexity_MB"] if c in numeric_cols],
        }

        for group_title, cols in metric_groups.items():
            if not cols:
                continue
            with st.expander(f"{group_title} — Box Plots", expanded=True):
                for col in cols:
                    fig_box = px.box(df, x="Model", y=col, color="Model",
                                      title=f"Distribution of  {col}",
                                      points="all")
                    st.plotly_chart(fig_box, use_container_width=True)

        # ─────────────────────────────────────────────────────────
        # SECTION 2 — Per-Metric Bar Charts (model averages)
        # ─────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📈 Per-Metric Bar Charts — Model Averages")

        for group_title, cols in metric_groups.items():
            if not cols:
                continue
            avail = [c for c in cols if c in grouped_all.columns]
            if not avail:
                continue
            with st.expander(f"{group_title} — Bar Charts", expanded=False):
                grp_reset = grouped_all[avail].reset_index()
                fig_bar = px.bar(grp_reset, x="Model", y=avail, barmode="group",
                                  title=f"{group_title} — Grouped Bar",
                                  color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig_bar, use_container_width=True)

        # ─────────────────────────────────────────────────────────
        # SECTION 3 — Pairwise t-test & Mann-Whitney on ALL metrics
        # ─────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📐 Pairwise Statistical Tests — All Metrics")
        st.caption("Welch t-test + Mann-Whitney U for every numeric metric between two selected models.")

        if len(models_list) >= 2:
            col_ma, col_mb = st.columns(2)
            sel_a = col_ma.selectbox("Model A", models_list, key="ttest_a")
            sel_b = col_mb.selectbox("Model B", models_list,
                                      index=min(1, len(models_list)-1), key="ttest_b")

            if sel_a != sel_b:
                rows = []
                for col in numeric_cols:
                    d1 = df[df["Model"]==sel_a][col].dropna()
                    d2 = df[df["Model"]==sel_b][col].dropna()
                    if len(d1) >= 2 and len(d2) >= 2:
                        t_s, t_p = ttest_ind(d1, d2, equal_var=False)
                        try:
                            u_s, u_p = mannwhitneyu(d1, d2, alternative="two-sided")
                        except Exception:
                            u_s, u_p = float("nan"), float("nan")
                        rows.append({
                            "Metric":        col,
                            f"Mean {sel_a}": round(d1.mean(), 6),
                            f"Mean {sel_b}": round(d2.mean(), 6),
                            "t-statistic":   round(t_s, 4),
                            "t p-value":     round(t_p, 4),
                            "MWU U-stat":    round(u_s, 4),
                            "MWU p-value":   round(u_p, 4),
                            "Significant?":  "✅ Yes" if t_p < 0.05 else "❌ No"
                        })
                if rows:
                    ttest_df = pd.DataFrame(rows)
                    st.dataframe(
                        ttest_df.style.map(
                            lambda v: "background-color:#d4edda" if v=="✅ Yes"
                                      else ("background-color:#f8d7da" if v=="❌ No" else ""),
                            subset=["Significant?"]
                        ),
                        use_container_width=True
                    )
                    csv_tt = ttest_df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download t-test Results", csv_tt,
                                       "pairwise_ttest.csv")
                else:
                    st.warning("Each model needs at least 2 runs for significance testing.")
            else:
                st.warning("Select two different models.")
        else:
            st.info("Run at least 2 different models to enable pairwise tests.")

        # ─────────────────────────────────────────────────────────
        # SECTION 4 — ANOVA + Kruskal-Wallis on ALL metrics
        # ─────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📐 ANOVA & Kruskal-Wallis — All Metrics")
        st.caption("One-way ANOVA (parametric) and Kruskal-Wallis (non-parametric) across all models for every metric.")

        anova_rows = []
        for col in numeric_cols:
            groups = [df[df["Model"]==m][col].dropna()
                      for m in models_list
                      if len(df[df["Model"]==m][col].dropna()) >= 2]
            if len(groups) >= 2:
                f_s, f_p = f_oneway(*groups)
                try:
                    k_s, k_p = kruskal(*groups)
                except Exception:
                    k_s, k_p = float("nan"), float("nan")
                anova_rows.append({
                    "Metric":           col,
                    "ANOVA F-stat":     round(f_s, 4),
                    "ANOVA p-value":    round(f_p, 4),
                    "ANOVA Sig?":       "✅" if f_p < 0.05 else "❌",
                    "Kruskal H-stat":   round(k_s, 4),
                    "Kruskal p-value":  round(k_p, 4),
                    "Kruskal Sig?":     "✅" if k_p < 0.05 else "❌",
                })

        if anova_rows:
            anova_df = pd.DataFrame(anova_rows)
            st.dataframe(
                anova_df.style.map(
                    lambda v: "background-color:#d4edda" if v=="✅"
                              else ("background-color:#f8d7da" if v=="❌" else ""),
                    subset=["ANOVA Sig?","Kruskal Sig?"]
                ),
                use_container_width=True
            )
            csv_an = anova_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download ANOVA Results", csv_an, "anova_results.csv")
        else:
            st.warning("Need at least 2 models with 2+ runs each.")

        # ─────────────────────────────────────────────────────────
        # SECTION 5 — Levene's Test (variance equality) on ALL metrics
        # ─────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📐 Levene's Test — Variance Equality Across Models")
        st.caption("Tests whether variance of each metric is equal across models (homoscedasticity).")

        levene_rows = []
        for col in numeric_cols:
            groups = [df[df["Model"]==m][col].dropna()
                      for m in models_list
                      if len(df[df["Model"]==m][col].dropna()) >= 2]
            if len(groups) >= 2:
                try:
                    l_s, l_p = levene(*groups)
                    levene_rows.append({
                        "Metric":         col,
                        "Levene W-stat":  round(l_s, 4),
                        "p-value":        round(l_p, 4),
                        "Equal Variance?":"✅ Yes" if l_p >= 0.05 else "❌ No"
                    })
                except Exception:
                    pass

        if levene_rows:
            lev_df = pd.DataFrame(levene_rows)
            st.dataframe(
                lev_df.style.map(
                    lambda v: "background-color:#d4edda" if v=="✅ Yes"
                              else ("background-color:#f8d7da" if v=="❌ No" else ""),
                    subset=["Equal Variance?"]
                ),
                use_container_width=True
            )
            csv_lev = lev_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Levene Results", csv_lev, "levene_results.csv")

        # ─────────────────────────────────────────────────────────
        # SECTION 6 — Shapiro-Wilk Normality on ALL metrics
        # ─────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📐 Shapiro-Wilk Normality Test — All Models × All Metrics")
        st.caption("Tests whether each metric per model follows a normal distribution.")

        sw_rows = []
        for col in numeric_cols:
            for m in models_list:
                vals = df[df["Model"]==m][col].dropna()
                if 3 <= len(vals) <= 5000:
                    try:
                        sw_s, sw_p = shapiro(vals)
                        sw_rows.append({
                            "Model":      m,
                            "Metric":     col,
                            "W-stat":     round(sw_s, 4),
                            "p-value":    round(sw_p, 4),
                            "Normal?":    "✅ Yes" if sw_p >= 0.05 else "❌ No"
                        })
                    except Exception:
                        pass

        if sw_rows:
            sw_df = pd.DataFrame(sw_rows)
            with st.expander("View Shapiro-Wilk Results Table", expanded=False):
                st.dataframe(
                    sw_df.style.map(
                        lambda v: "background-color:#d4edda" if v=="✅ Yes"
                                  else ("background-color:#f8d7da" if v=="❌ No" else ""),
                        subset=["Normal?"]
                    ),
                    use_container_width=True
                )
            csv_sw = sw_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Shapiro-Wilk Results", csv_sw, "shapiro_wilk.csv")
        else:
            st.info("Need 3–5000 samples per model to run Shapiro-Wilk.")

        # ─────────────────────────────────────────────────────────
        # SECTION 7 — Pearson + Spearman + Kendall Correlation
        #             Full matrix + selectable pair deep-dive
        # ─────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔗 Correlation Analysis — All Metrics")

        st.markdown("#### Pearson Correlation Matrix")
        corr_p = df[numeric_cols].corr(method="pearson")
        st.plotly_chart(px.imshow(corr_p, text_auto=".2f",
                                   color_continuous_scale="RdBu_r",
                                   title="Pearson Correlation Matrix"),
                        use_container_width=True)

        st.markdown("#### Spearman Correlation Matrix")
        corr_s = df[numeric_cols].corr(method="spearman")
        st.plotly_chart(px.imshow(corr_s, text_auto=".2f",
                                   color_continuous_scale="RdBu_r",
                                   title="Spearman Correlation Matrix"),
                        use_container_width=True)

        # Deep-dive: any two metrics
        st.markdown("#### Deep-dive: Pairwise Correlation Between Any Two Metrics")
        ca, cb = st.columns(2)
        metric_x = ca.selectbox("Metric X", numeric_cols, key="corr_x")
        metric_y = cb.selectbox("Metric Y", numeric_cols,
                                  index=min(1,len(numeric_cols)-1), key="corr_y")

        if metric_x != metric_y:
            valid = df[[metric_x, metric_y]].dropna()
            if len(valid) >= 3:
                pr, pp   = pearsonr(valid[metric_x], valid[metric_y])
                sr, sp   = spearmanr(valid[metric_x], valid[metric_y])
                kr, kp   = kendalltau(valid[metric_x], valid[metric_y])
                cc1,cc2,cc3,cc4,cc5,cc6 = st.columns(6)
                cc1.metric("Pearson r",   f"{pr:.4f}")
                cc2.metric("Pearson p",   f"{pp:.4f}")
                cc3.metric("Spearman ρ",  f"{sr:.4f}")
                cc4.metric("Spearman p",  f"{sp:.4f}")
                cc5.metric("Kendall τ",   f"{kr:.4f}")
                cc6.metric("Kendall p",   f"{kp:.4f}")

                fig_sc = px.scatter(df, x=metric_x, y=metric_y, color="Model",
                                     trendline="ols",
                                     title=f"{metric_x} vs {metric_y} (with OLS trend)")
                st.plotly_chart(fig_sc, use_container_width=True)

        # Full pairwise correlation significance table
        st.markdown("#### Full Pairwise Pearson Significance Table")
        with st.expander("View all metric pairs (p-values)", expanded=False):
            pair_rows = []
            for i, cx in enumerate(numeric_cols):
                for cy in numeric_cols[i+1:]:
                    vals = df[[cx,cy]].dropna()
                    if len(vals) >= 3:
                        try:
                            r, p = pearsonr(vals[cx], vals[cy])
                            pair_rows.append({
                                "Metric A": cx, "Metric B": cy,
                                "Pearson r": round(r,4), "p-value": round(p,4),
                                "Significant?": "✅" if p<0.05 else "❌"
                            })
                        except Exception:
                            pass
            if pair_rows:
                pair_df = pd.DataFrame(pair_rows).sort_values("p-value")
                st.dataframe(
                    pair_df.style.map(
                        lambda v: "background-color:#d4edda" if v=="✅"
                                  else ("background-color:#f8d7da" if v=="❌" else ""),
                        subset=["Significant?"]
                    ),
                    use_container_width=True
                )
                csv_pair = pair_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Pair Correlation Table",
                                   csv_pair, "pairwise_correlations.csv")

        # ─────────────────────────────────────────────────────────
        # SECTION 8 — Pareto Frontiers
        # ─────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("⚡ Pareto Frontier Analysis")
        grouped_r = df.groupby("Model")[numeric_cols].mean().reset_index()

        def pareto_plot(x_col, y_col, title):
            if x_col not in grouped_r.columns or y_col not in grouped_r.columns:
                return
            fig = px.scatter(grouped_r, x=x_col, y=y_col, text="Model",
                              size=y_col, color="Model", title=title)
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, use_container_width=True)
            optimal = []
            for i, ri in grouped_r.iterrows():
                dominated = any(
                    rj[y_col] >= ri[y_col] and rj[x_col] <= ri[x_col] and
                    (rj[y_col] > ri[y_col] or rj[x_col] < ri[x_col])
                    for j, rj in grouped_r.iterrows()
                )
                if not dominated:
                    optimal.append(ri["Model"])
            st.success(f"🏆 Pareto Optimal: {', '.join(optimal)}")

        pareto_plot("Time_mean",       "Accuracy_mean",    "Pareto: Accuracy vs Runtime")
        pareto_plot("Space_Complexity_MB","Efficiency_Score","Pareto: Efficiency vs Space")
        pareto_plot("CPU_Usage_pct",   "Accuracy_mean",    "Pareto: Accuracy vs CPU Usage")
        pareto_plot("Training_Time",   "F1_Score",         "Pareto: F1 Score vs Training Time")
        pareto_plot("Peak_Heap_MB",    "Accuracy_mean",    "Pareto: Accuracy vs Peak Memory")

        # ─────────────────────────────────────────────────────────
        # SECTION 9 — Radar Chart (all key metrics)
        # ─────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🕸️ Model Performance Radar — All Key Metrics")
        all_radar = [c for c in ["Accuracy_mean","F1_Score","Precision","Recall",
                                   "MCC","Balanced_Accuracy","Cohen_Kappa","ROC_AUC",
                                   "Efficiency_Score","pS","pD"]
                     if c in df.columns]

        if len(all_radar) >= 3:
            scaler   = MinMaxScaler()
            sub_r    = grouped_all[all_radar]
            scaled_r = pd.DataFrame(scaler.fit_transform(sub_r),
                                     columns=all_radar, index=sub_r.index)
            fig_rad  = go.Figure()
            for m in scaled_r.index:
                fig_rad.add_trace(go.Scatterpolar(
                    r=scaled_r.loc[m].values, theta=all_radar,
                    fill="toself", name=m))
            fig_rad.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                showlegend=True, title="Normalised Radar — All Classification + UCF Metrics")
            st.plotly_chart(fig_rad, use_container_width=True)

        # Runtime radar
        runtime_radar = [c for c in ["Training_Time","Inference_Time","CPU_Usage_pct",
                                       "RAM_Delta_MB","Peak_Heap_MB","Space_Complexity_MB"]
                         if c in df.columns]
        if len(runtime_radar) >= 3:
            sub_rt   = grouped_all[runtime_radar]
            scaled_rt= pd.DataFrame(MinMaxScaler().fit_transform(sub_rt),
                                      columns=runtime_radar, index=sub_rt.index)
            fig_rt   = go.Figure()
            for m in scaled_rt.index:
                fig_rt.add_trace(go.Scatterpolar(
                    r=scaled_rt.loc[m].values, theta=runtime_radar,
                    fill="toself", name=m))
            fig_rt.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                showlegend=True, title="Normalised Radar — Runtime & Memory Metrics")
            st.plotly_chart(fig_rt, use_container_width=True)

        # ─────────────────────────────────────────────────────────
        # SECTION 10 — Overall Ranking (weighted scoring)
        # ─────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🏆 Overall Model Ranking — Weighted Score")
        st.caption("Lower time/memory = better. All others: higher = better. Weights shown below.")

        score_weights = {
            "Accuracy_mean":    0.20,
            "F1_Score":         0.15,
            "Precision":        0.05,
            "Recall":           0.05,
            "MCC":              0.05,
            "Balanced_Accuracy":0.05,
            "Cohen_Kappa":      0.05,
            "ROC_AUC":          0.05,
            "Efficiency_Score": 0.10,
            "pS":               0.05,
            "pD":               0.05,
            "Time_mean":        -0.05,   # negative = lower is better
            "Training_Time":    -0.03,
            "Inference_Time":   -0.02,
            "Peak_Heap_MB":     -0.02,
            "Space_Complexity_MB": -0.03,
        }

        rank_df  = grouped_all.copy()
        score_col = pd.Series(0.0, index=rank_df.index)

        for col, weight in score_weights.items():
            if col not in rank_df.columns:
                continue
            col_min = rank_df[col].min()
            col_max = rank_df[col].max()
            rng     = col_max - col_min + 1e-9
            if weight > 0:
                norm = (rank_df[col] - col_min) / rng
            else:
                norm = 1 - (rank_df[col] - col_min) / rng
            score_col += abs(weight) * norm

        rank_df["Final_Score"] = score_col
        rank_df = rank_df.sort_values("Final_Score", ascending=False)

        # Show weights
        with st.expander("View scoring weights", expanded=False):
            w_df = pd.DataFrame(list(score_weights.items()),
                                 columns=["Metric","Weight"])
            w_df["Direction"] = w_df["Weight"].apply(
                lambda x: "↑ Higher is better" if x>0 else "↓ Lower is better")
            st.dataframe(w_df, use_container_width=True)

        best_model = rank_df.index[0]
        best_score = rank_df["Final_Score"].iloc[0]
        st.success(f"🏆 Best Overall Model: **{best_model}** (Score: {best_score:.4f})")

        st.dataframe(utlis.safe_gradient(utlis.sanitise_df(rank_df)[["Final_Score"]].style, cmap="RdYlGn"),
                     use_container_width=True)

        # Leaderboard bar
        lb = rank_df[["Final_Score"]].reset_index()
        lb["Rank"] = range(1, len(lb)+1)
        st.dataframe(lb[["Rank","Model","Final_Score"]], use_container_width=True)
        st.plotly_chart(px.bar(lb, x="Model", y="Final_Score", color="Final_Score",
                                title="Model Leaderboard — Final Score",
                                color_continuous_scale="viridis"),
                        use_container_width=True)

        # ─────────────────────────────────────────────────────────
        # SECTION 11 — Experiment History & Downloads
        # ─────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📂 Experiment History")
        st.dataframe(df, use_container_width=True)

        csv_full = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Full Experiment Log", csv_full,
                           "experiment_history_full.csv")

        csv_avg = grouped_all.to_csv().encode("utf-8")
        st.download_button("⬇️ Download Model Averages CSV", csv_avg,
                           "model_averages.csv")

    # =====================================================
    # 4️⃣ MODEL COMPARISON
    # =====================================================
