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

     
        # ============================================================
        # HOME — INTRODUCTION
        # ============================================================
        
        st.markdown("""
        ## 🧠 Algorithmic Analyzer
        
        ### Computational Complexity • Unified Complexity Function (UCF)
        
        **Analyze • Visualize • Understand • Advance**
        
        The **Algorithmic Analyzer** is a research-oriented computational analysis
        platform designed and developed by **Aqib Ali Buriro**, a student of
        **Mehran University of Engineering and Technology (MUET)**, under the
        supervision of **Dr. Bushra Naz** and **Dr. Sammer Zai**.
        
        The platform implements the **Unified Complexity Function (UCF) Framework**
        to investigate the computational behavior of **Classical Algorithms** and
        **Machine Learning Models**.
        
        Rather than considering theoretical complexity alone, the system combines
        traditional **Big-O complexity analysis** with empirical computational
        measurements to provide a broader understanding of algorithmic performance.
        """)
        
        
        # ============================================================
        # WHAT THE SYSTEM ANALYZES
        # ============================================================
        
        st.subheader("🔬 What the System Analyzes")
        
        st.markdown("""
        The platform evaluates algorithms and machine-learning models across
        multiple computational and predictive dimensions:
        
        - ⏱️ **Execution & Runtime Performance**
        - 💾 **Memory and Space Utilization**
        - ⚙️ **Algorithm Efficiency**
        - 🧮 **Computational Complexity**
        - 📊 **Accuracy, Precision, Recall & F1-Score**
        - 🚀 **Throughput and Performance Behavior**
        - 🖥️ **CPU and Resource Utilization**
        - 🧠 **Explainable AI (XAI)**
        - 📈 **Model and Algorithm Comparison**
        - 🔬 **Unified Complexity Function (UCF) Analysis**
        """)
        
        
        # ============================================================
        # RESEARCH OBJECTIVE
        # ============================================================
        
        st.subheader("🎯 Research Objective")
        
        st.info("""
        The primary objective of the Algorithmic Analyzer is to bridge the gap
        between **theoretical computational complexity** and **observed empirical
        performance**.
        
        Traditional Big-O notation explains how computational requirements scale
        with input size, while the UCF-based analytical approach incorporates
        measurable runtime and resource characteristics.
        
        Together, these perspectives provide a more comprehensive environment for
        studying, comparing, and interpreting computational behavior.
        """)
        
        
        # ============================================================
        # SYSTEM ARCHITECTURE
        # ============================================================
        
        st.subheader("🏗️ System Architecture")
        
        st.markdown("""
        The analytical workflow follows six major stages:
        
        **1️⃣ Classical Algorithm Analysis**  
        Analyze computational behavior and complexity characteristics of
        classical algorithms.
        
        **2️⃣ Machine Learning Model Evaluation**  
        Train and evaluate machine-learning models using predictive and
        computational performance metrics.
        
        **3️⃣ UCF Metric Computation**  
        Integrate relevant computational measurements through the
        **Unified Complexity Function (UCF)** framework.
        
        **4️⃣ Statistical & Model Comparison**  
        Compare algorithms and models across multiple performance dimensions.
        
        **5️⃣ Explainable AI — SHAP & LIME**  
        Interpret model predictions and investigate feature contributions.
        
        **6️⃣ Big-O vs Empirical Performance**  
        Compare theoretical complexity expectations with observed
        runtime and resource behavior.
        """)
        
        
        # ============================================================
        # PLATFORM CAPABILITIES
        # ============================================================
        
        st.subheader("🚀 Platform Capabilities")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="ML Models",
                value="13"
            )
        
        with col2:
            st.metric(
                label="Analysis Domains",
                value="Classical + ML"
            )
        
        with col3:
            st.metric(
                label="Evaluation Metrics",
                value="10+"
            )
        
        with col4:
            st.metric(
                label="Complexity Framework",
                value="UCF"
            )
        
        
        # ============================================================
        # CORE MODULES
        # ============================================================
        
        st.subheader("🧩 Core Modules")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Algorithm & Model Analysis**
            
            - Classical Algorithm Analysis
            - Machine Learning Analysis
            - Performance Metrics
            - Model Comparison
            - Advanced Visualizations
            """)
        
        with col2:
            st.markdown("""
            **UCF & Intelligent Analysis**
            
            - UCF Dashboard
            - UCF vs Big-O Analysis
            - UCF Formula Lab
            - Explainable AI
            - AutoML Selection
            """)
        
        
        # ============================================================
        # RESEARCH VISION
        # ============================================================
        
        st.subheader("💡 Research Vision")
        
        st.success("""
        **From Complexity to Clarity**
        
        Algorithmic Analyzer aims to provide an integrated experimental environment
        where theoretical complexity, empirical performance, machine-learning
        evaluation, resource utilization, and explainability can be investigated
        within a unified analytical workflow.
        
        **Analyze • Visualize • Understand • Advance**
        """)
 

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

