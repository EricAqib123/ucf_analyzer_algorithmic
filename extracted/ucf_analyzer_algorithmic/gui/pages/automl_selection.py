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
        st.header("🤖 AutoML — Automated Model Selection")
        st.markdown("Automatically benchmarks **all 13 models** and selects the best one using cross-validation + UCF scoring.")

        if "X_train" not in st.session_state:
            st.info("⚠️ Run Machine Learning phase first to load dataset into session.")
            st.stop()

        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.preprocessing import MinMaxScaler as _MMS

        X_tr = st.session_state.X_train
        X_te = st.session_state.X_test
        y_tr = st.session_state.y_train
        y_te = st.session_state.y_test

        st.subheader("⚙️ AutoML Configuration")
        c1, c2 = st.columns(2)
        cv_folds = c1.slider("Cross-Validation Folds", 3, 10, 5)
        metric_w = c2.selectbox("Primary Ranking Metric", ["F1 + UCF Composite", "Accuracy", "F1", "MCC"])

        automl_models = {
            "Decision Tree":                   decision_tree_model,
            "Random Forest":                   random_forest_model,
            "Logistic Regression":             logistic_regression_model,
            "K-Nearest Neighbors":             knn_model,
            "Support Vector Machine":          svm_model,
            "Gradient Boosting":               gradient_boosting_model,
            "XGBoost":                         xgboost_model,
            "LightGBM":                        lightgbm_model,
            "AdaBoost":                        adaboost_model,
            "Extra Trees":                     extra_trees_model,
            "Linear Discriminant Analysis":    lda_model,
            "Quadratic Discriminant Analysis": qda_model,
            "MLP Neural Network":              mlp_model,
        }

        if st.button("🚀 Run AutoML Search"):
            import tracemalloc as _tm2, time as _t2, psutil as _ps2, os as _os2

            leaderboard = []
            prog = st.progress(0)
            status_txt = st.empty()

            for i, (name, factory) in enumerate(automl_models.items()):
                status_txt.text(f"Benchmarking {name}...")
                try:
                    mdl = factory()
                    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

                    cv_acc = cross_val_score(mdl, X_tr, y_tr, cv=skf, scoring="accuracy").mean()
                    cv_f1  = cross_val_score(mdl, X_tr, y_tr, cv=skf, scoring="f1_weighted").mean()

                    proc2 = _ps2.Process(_os2.getpid())
                    ram_b = proc2.memory_info().rss / 1024 / 1024
                    _tm2.start()
                    t0 = _t2.time()
                    mdl.fit(X_tr, y_tr)
                    train_t = _t2.time() - t0
                    _, peak2 = _tm2.get_traced_memory(); _tm2.stop()
                    ram_a = proc2.memory_info().rss / 1024 / 1024

                    t1 = _t2.time()
                    y_p2 = mdl.predict(X_te)
                    infer_t = _t2.time() - t1

                    from sklearn.metrics import f1_score as _f1s, matthews_corrcoef as _mcc
                    f1_fin  = _f1s(y_te, y_p2, average="weighted")
                    mcc_fin = _mcc(y_te, y_p2)
                    mem_mb  = peak2 / 1024 / 1024
                    throughput2 = len(X_te) / (infer_t + 1e-9)
                    ucf_proxy   = cv_acc / (train_t + 1e-9) / (mem_mb + 1e-3)

                    leaderboard.append({
                        "Model":        name,
                        "CV_Accuracy":  round(cv_acc,    4),
                        "CV_F1":        round(cv_f1,     4),
                        "Final_F1":     round(f1_fin,    4),
                        "MCC":          round(mcc_fin,   4),
                        "Train_Time_s": round(train_t,   4),
                        "Infer_Time_s": round(infer_t,   6),
                        "Peak_Heap_MB": round(mem_mb,    4),
                        "RAM_Delta_MB": round(ram_a - ram_b, 4),
                        "Throughput":   round(throughput2, 1),
                        "UCF_Proxy":    round(ucf_proxy, 4),
                    })
                except Exception as ex:
                    leaderboard.append({"Model": name, "Error": str(ex)})
                prog.progress((i + 1) / len(automl_models))

            status_txt.text("✅ AutoML search complete!")
            lb_df = pd.DataFrame([r for r in leaderboard if "Error" not in r])

            if lb_df.empty:
                st.error("All models failed. Check dataset compatibility.")
                st.stop()

            # Composite scoring
            scaler2     = _MMS()
            score_cols  = ["CV_Accuracy", "CV_F1", "Final_F1", "MCC", "UCF_Proxy"]
            penalty_cols= ["Train_Time_s", "Peak_Heap_MB"]
            avail_s = [c for c in score_cols   if c in lb_df.columns]
            avail_p = [c for c in penalty_cols if c in lb_df.columns]

            composite = pd.Series(0.0, index=lb_df.index)
            if avail_s:
                norm_s = pd.DataFrame(scaler2.fit_transform(lb_df[avail_s]), columns=avail_s)
                composite += norm_s.mean(axis=1) * 0.8
            if avail_p:
                norm_p = pd.DataFrame(scaler2.fit_transform(lb_df[avail_p]), columns=avail_p)
                composite -= norm_p.mean(axis=1) * 0.2

            lb_df["Composite_Score"] = composite.values
            lb_df = lb_df.sort_values("Composite_Score", ascending=False).reset_index(drop=True)
            lb_df.insert(0, "Rank", range(1, len(lb_df) + 1))

            best = lb_df.iloc[0]["Model"]
            st.success(f"🏆 **Best Model: {best}** (Composite Score: {lb_df.iloc[0]['Composite_Score']:.4f})")

            st.subheader("📋 AutoML Leaderboard")
            st.dataframe(utlis.safe_gradient(utlis.sanitise_df(lb_df.set_index("Rank")).style, cmap="RdYlGn",
                                       subset=["CV_Accuracy","CV_F1","Final_F1","Composite_Score"]),
                         use_container_width=True)

            st.subheader("📊 Composite Score Leaderboard")
            fig_lb = px.bar(lb_df, x="Model", y="Composite_Score", color="Composite_Score",
                            color_continuous_scale="viridis",
                            title="AutoML Composite Score — All Models")
            st.plotly_chart(fig_lb, use_container_width=True)

            st.subheader("⚖️ Accuracy vs Training Time Trade-off")
            fig_tt = px.scatter(lb_df, x="Train_Time_s", y="CV_Accuracy",
                                text="Model", size="CV_F1", color="Model",
                                title="CV Accuracy vs Training Time")
            fig_tt.update_traces(textposition="top center")
            st.plotly_chart(fig_tt, use_container_width=True)

            st.subheader("🧠 Memory vs F1 Trade-off")
            fig_mf = px.scatter(lb_df, x="Peak_Heap_MB", y="Final_F1",
                                text="Model", color="Model",
                                title="F1 Score vs Peak Memory Usage")
            fig_mf.update_traces(textposition="top center")
            st.plotly_chart(fig_mf, use_container_width=True)

            st.subheader("🕸️ Top-5 Models — Performance Radar")
            top5 = lb_df.head(5)
            radar_cols_auto = [c for c in ["CV_Accuracy","CV_F1","Final_F1","MCC","UCF_Proxy"] if c in top5.columns]
            if len(radar_cols_auto) >= 3:
                sc3   = _MMS()
                sub5  = top5.set_index("Model")[radar_cols_auto]
                sc5   = pd.DataFrame(sc3.fit_transform(sub5), columns=radar_cols_auto, index=sub5.index)
                fig_r5 = go.Figure()
                for m in sc5.index:
                    fig_r5.add_trace(go.Scatterpolar(r=sc5.loc[m].values, theta=radar_cols_auto,
                                                      fill="toself", name=m))
                fig_r5.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                                      showlegend=True, title="Top-5 AutoML Models — Radar")
                st.plotly_chart(fig_r5, use_container_width=True)

            csv_lb = lb_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download AutoML Leaderboard", csv_lb, "automl_leaderboard.csv")

    # =====================================================
    # 🔟 ADVANCED VISUALIZATIONS
    # =====================================================
