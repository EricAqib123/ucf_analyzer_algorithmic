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

        st.header("Welcome to AI-Based Algorithmic Analyzer")

        st.write("""
                 This system is being designed and developed by Aqib Ali Buriro Student of Mehran University of Engineering and Technology under the supervision Dr Bushra Naz and Dr. Sammer Zai

    This platform implements the **UCF (Unified Complexity Framework)** for analyzing 
    both **Classical Algorithms** and **Machine Learning Models**.

    The system measures:

    • Time Complexity  
    • Memory Usage  
    • Algorithm Efficiency  
    • Accuracy & F1 Score  
    • Runtime Performance  
    • Explainability  

    The goal is to **bridge theoretical complexity (Big-O)** with **practical performance metrics**.
        """)

        st.subheader("System Architecture")

        st.info("""
    1️⃣ Classical Algorithm Analysis  
    2️⃣ Machine Learning Model Evaluation  
    3️⃣ UCF Metric Computation  
    4️⃣ Statistical Comparison  
    5️⃣ Explainable AI (SHAP + LIME)  
    6️⃣ Big-O vs Practical Performance  
    """)

        st.subheader("Modules Available")

        col1, col2, col3 = st.columns(3)

        col1.metric("ML Models", "13")
        col2.metric("Algorithms", "Sorting + ML")
        col3.metric("Evaluation Metrics", "10+")

        st.success("Use the sidebar to start experiments.")

        # ─────────────────────────────────────────────────────
        # Dataset Download
        # ─────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📦 Download Dataset")
        st.write("Download the CIC-IDS dataset to use with the Machine Learning phase.")

        from pathlib import Path

        # home.py is located at: project/gui/pages/home.py
        # Go up three levels to reach the project root.
        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
        dataset_path = PROJECT_ROOT / "dataset.rar"

        try:
            if dataset_path.is_file():
                with open(dataset_path, "rb") as f_ds:
                    dataset_bytes = f_ds.read()

                st.download_button(
                    label="⬇️ Download CIC-IDS Dataset (.rar)",
                    data=dataset_bytes,
                    file_name="dataset.rar",
                    mime="application/vnd.rar",
                )
            else:
                st.info(
                    "Dataset file (`dataset.rar`) was not found. "
                    "Make sure it is uploaded to the project root alongside `app.py`."
                )

        except Exception as e:
            st.warning(f"Could not load dataset file: {e}")

