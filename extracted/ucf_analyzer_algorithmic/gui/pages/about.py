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

        st.header("About This System")

        st.write("""
    **AI-Based Algorithmic Analyzer using UCF Model**

    This system was developed to evaluate both classical algorithms and 
    machine learning models using a unified complexity framework (UCF).

    The system integrates:

    • Classical algorithm analysis  
    • Machine learning benchmarking  
    • Statistical validation  
    • Explainable AI techniques  
    • Big-O theoretical comparison  

    University: **Mehran University of Engineering & Technology Jamshoro**

    Department: **Computer Systems Engineering**
    """)

        st.subheader("Technologies Used")

        st.write("""
    • Python  
    • Streamlit  
    • Scikit-learn  
    • Plotly  
    • SHAP  
    • LIME  
    • Pandas & NumPy  
    """)


    # =====================================================
    # 9️⃣ AUTO ML SELECTION
    # =====================================================
