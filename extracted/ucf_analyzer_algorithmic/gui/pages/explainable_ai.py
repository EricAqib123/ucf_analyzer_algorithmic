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

        st.header("🧠 Unified Dashboard: UCF vs Big-O with Explainable AI")

        if st.session_state.results_df.empty:
            st.info("Run Machine Learning experiments first.")
            st.stop()

        df = utlis.sanitise_df(st.session_state.results_df.copy())
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        # =====================================================
        # Big-O Mapping
        # =====================================================
        big_o_map = {
            "Decision Tree": "O(p⋅nlogn)",
            "Random Forest": "O(ntrees⋅p⋅nlogn)",
            "Logistic Regression": "O(n⋅p)",
            "K-Nearest Neighbors": "O(1) (Lazy Learner)",
            "Support Vector Machine": "O(n⋅p) to O(n^2⋅p)",
            "Gradient Boosting": "O(ntrees⋅p⋅nlogn)",
            "XGBoost": "O(ntrees⋅p⋅nnonzero log n)",
            "LightGBM": "O(ntrees⋅p⋅nnonzero log n)",
            "AdaBoost": "O(ntrees⋅n⋅p)",
            "Extra Trees": "O(ntrees⋅p⋅n)",
            "Linear Discriminant Analysis": "O(n⋅p^2+p^3)",
            "Quadratic Discriminant Analysis": "O(n⋅p^2+p^3)",
            "MLP Neural Network": "O(n⋅p⋅h1⋅h2⋯)"
        }

        big_o_scale = {
            "O(1)": 1, "O(log n)": 2, "O(n)": 3, "O(n log n)": 4, "O(n^2)": 5, "O(n^3)": 6,
            "O(p⋅nlogn)": 4, "O(ntrees⋅p⋅nlogn)": 5, "O(n⋅p)": 3, "O(1) (Lazy Learner)": 1,
            "O(n⋅p) to O(n^2⋅p)": 5, "O(ntrees⋅p⋅nnonzero log n)": 5, "O(ntrees⋅n⋅p)": 5,
            "O(ntrees⋅p⋅n)": 5, "O(n⋅p^2+p^3)": 6, "O(n⋅p⋅h1⋅h2⋯)": 6
        }

        df["BigO"] = df["Model"].map(big_o_map)
        df["BigO_Score"] = df["BigO"].map(big_o_scale)

        # =====================================================
        # Select Model for XAI
        # =====================================================
        models_list = df["Model"].unique()
        model_choice = st.selectbox("Select Model for XAI Analysis", models_list)

        if "X_train" not in st.session_state or "y_train" not in st.session_state:
            st.warning("Train a model first in Machine Learning phase.")
            st.stop()

        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test

        model_mapping = {
            "Decision Tree": decision_tree_model,
            "Random Forest": random_forest_model,
            "Logistic Regression": logistic_regression_model,
            "K-Nearest Neighbors": knn_model,
            "Support Vector Machine": svm_model,
            "Gradient Boosting": gradient_boosting_model,
            "XGBoost": xgboost_model,
            "LightGBM": lightgbm_model,
            "AdaBoost": adaboost_model,
            "Extra Trees": extra_trees_model,
            "Linear Discriminant Analysis": lda_model,
            "Quadratic Discriminant Analysis": qda_model,
            "MLP Neural Network": mlp_model
        }

        # Labels saved from ML page are already encoded (numeric)
        y_train_enc = y_train
        y_test_enc = y_test

        model = model_mapping[model_choice]()
        model.fit(X_train, y_train_enc)
        y_pred = model.predict(X_test)

        # =====================================================
        # UCF vs Big-O Table
        # =====================================================
        st.subheader("📊 UCF vs Big-O Table")
        metrics = ["Time_mean", "Memory_mean", "Accuracy_mean", "F1_Score", "BigO_Score"]
        available_metrics = [m for m in metrics if m in df.columns]
        grouped = df.groupby("Model")[available_metrics].mean().reset_index()
        st.dataframe(grouped, use_container_width=True)

        # =====================================================
        # Big-O vs Runtime and Memory
        # =====================================================
        st.subheader("⚡ Big-O vs Runtime and Memory")
        fig_runtime = px.scatter(
            grouped, x="BigO_Score", y="Time_mean", text="Model", size="Time_mean", color="Model",
            title="Big-O Complexity vs Measured Runtime"
        )
        st.plotly_chart(fig_runtime, use_container_width=True)

        fig_memory = px.scatter(
            grouped, x="BigO_Score", y="Memory_mean", text="Model", size="Memory_mean", color="Model",
            title="Big-O Complexity vs Measured Memory"
        )
        st.plotly_chart(fig_memory, use_container_width=True)

        # =====================================================
        # Feature Importance
        # =====================================================
        st.subheader("Feature Importance")
        if hasattr(model, "feature_importances_"):
            if isinstance(X_train, pd.DataFrame):
                feature_names = X_train.columns
            else:
                feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False)

            fig_imp = px.bar(
                importance_df,
                x="Feature",
                y="Importance",
                title="Feature Importances"
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance not available for this model.")

        # =====================================================
        # Ensure DataFrames
        # =====================================================
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test, columns=X_train.columns)

        # =====================================================
        # SHAP Global & Local Explanation
        # =====================================================
        st.subheader("SHAP Global & Local Explanation")

        try:
            import shap

            sample_size = min(200, len(X_train))
            sample_X = X_train.sample(sample_size, random_state=42).copy()

            explainer = shap.Explainer(model, sample_X)
            shap_values = explainer(sample_X)

            # Global Importance
            fig1, ax1 = plt.subplots()
            shap.summary_plot(shap_values, sample_X, plot_type="bar", show=False)
            st.pyplot(fig1)

            # Local Explanation
            st.write("Local SHAP Explanation")
            i = st.slider("Select Instance for SHAP", 0, len(X_test) - 1, 0)

            fig2 = plt.figure()
            shap.plots.force(
                explainer.expected_value,
                shap_values.values[i],
                X_test.iloc[i],
                matplotlib=True,
                show=False
            )
            st.pyplot(fig2)

        except Exception as e:
            st.warning(f"SHAP explanation failed: {e}")

        # =====================================================
        # LIME Explanation
        # =====================================================
        st.subheader("LIME Explanation")

        try:
            from lime.lime_tabular import LimeTabularExplainer

            lime_explainer = LimeTabularExplainer(
                training_data=X_train.values,
                feature_names=X_train.columns.tolist(),
                class_names=np.unique(y_train_enc).astype(str),
                discretize_continuous=True
            )

            i = st.slider("Select Instance for LIME", 0, len(X_test) - 1, 0, key="lime_slider")

            def predict_fn(x):
                x_df = pd.DataFrame(x, columns=X_train.columns)
                return model.predict_proba(x_df)

            exp = lime_explainer.explain_instance(
                X_test.iloc[i].values,
                predict_fn,
                num_features=10
            )
            st.write(exp.as_list())

        except Exception as e:
            st.warning(f"LIME explanation failed: {e}")

        # =====================================================
        # Runtime & Memory Feature Contribution
        # =====================================================
        st.subheader("Runtime & Memory Feature Contribution")

        for metric in ["Time_mean", "Memory_mean"]:
            if metric in df.columns:
                st.write(f"Feature Contribution to {metric}")
                try:
                    from sklearn.ensemble import RandomForestRegressor
                    y_metric = df[metric].dropna().iloc[:len(X_train)]
                    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
                    reg_model.fit(X_train, y_metric)

                    sample_size = min(200, len(X_train))
                    sample_X = X_train.sample(sample_size, random_state=42).copy()

                    import shap
                    explainer = shap.Explainer(reg_model, sample_X)
                    shap_vals = explainer(sample_X)

                    fig3, ax3 = plt.subplots()
                    shap.summary_plot(shap_vals, sample_X, plot_type="bar", show=False)
                    st.pyplot(fig3)

                except Exception as e:
                    st.warning(f"{metric} SHAP analysis failed: {e}")

        # =====================================================
        # Permutation Feature Importance
        # =====================================================
        st.subheader("📊 Permutation Feature Importance")
        try:
            from sklearn.inspection import permutation_importance
            result = permutation_importance(
                model,
                X_test,
                y_test_enc,
                n_repeats=10,
                random_state=42
            )
            importance_df = pd.DataFrame({
                "Feature": X_test.columns,
                "Importance": result.importances_mean
            }).sort_values("Importance", ascending=False)
            st.bar_chart(importance_df.set_index("Feature"))

        except Exception as e:
            st.warning(f"Permutation importance failed: {e}")

        st.success("✅ Unified Explainability Analysis Complete")

    # =====================================================
    # 8️⃣ ABOUT
    # =====================================================
