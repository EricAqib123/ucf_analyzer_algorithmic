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
from gui.utlis as utlis


def render():
        st.header("🔬 UCF Formula Laboratory — Improved Formula Analysis")
        st.markdown(r"""
    The **Unified Complexity Function (UCF)** is defined as:

    $$f(u)=f(t)+p(s)-p(d)$$


    $$combined formula f(u) = p(S) \cdot p(D) \cdot \frac{1}{TC(A,D) + R + \epsilon}$$

    - **f(u)**= Unified Complexity Function/Universal Complexity Function
    - **combined formula f(u)**= Improved Formula For System Verfication
    - **p(S)** = Probability of Stability/Problem Sequence
    - **p(D)** = Probability of Determinism/Conplexity Difference
    - **TC(A,D)** = Total Complexity = f(time) + g(memory)
    - **R** = Remainder / Residual complexity term
    - **ε** = small constant to prevent division by zero
    """)

        st.markdown("---")
        st.subheader("🧪 Interactive UCF Calculator")
        col1, col2 = st.columns(2)
        with col1:
            pS_val  = st.slider("p(S) — Stability probability",   0.0, 1.0, 0.85, 0.01)
            pD_val  = st.slider("p(D) — Determinism probability", 0.0, 1.0, 0.90, 0.01)
            t_val   = st.number_input("Time f(t) in seconds",    min_value=0.0, value=0.05,  format="%.6f")
        with col2:
            m_val   = st.number_input("Memory g(M) in MB",       min_value=0.0, value=10.0,  format="%.4f")
            r_val   = st.number_input("Remainder R",             min_value=0.0, value=0.001, format="%.6f")
            epsilon = st.number_input("Epsilon ε",               min_value=1e-12, value=1e-9, format="%.2e")

        TC_calc  = t_val + (m_val / 1000.0)
        fu_calc  = (pS_val * pD_val) / (TC_calc + r_val + epsilon)
        eff_calc = fu_calc * (pS_val + pD_val) / 2

        st.markdown("---")
        st.subheader("📐 Computed UCF Values")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("TC(A,D)",        f"{TC_calc:.6f}",  help="f(t) + g(M)/1000")
        r2.metric("f(u) — UCF",     f"{fu_calc:.6f}",  help="Higher = less complex")
        r3.metric("Efficiency Index",f"{eff_calc:.6f}", help="f(u) × mean(pS, pD)")
        r4.metric("Complexity Class","Low" if fu_calc > 10 else ("Medium" if fu_calc > 1 else "High"))

        st.markdown("---")
        st.subheader("📊 Sensitivity Analysis — How Each Parameter Affects f(u)")
        sens_data = []
        for ps_s in np.linspace(0.1, 1.0, 20):
            tc_s = t_val + m_val/1000; fu_s = (ps_s*pD_val)/(tc_s+r_val+epsilon)
            sens_data.append({"Parameter":"p(S)", "Value":ps_s, "f(u)":fu_s})
        for pd_s in np.linspace(0.1, 1.0, 20):
            tc_s = t_val + m_val/1000; fu_s = (pS_val*pd_s)/(tc_s+r_val+epsilon)
            sens_data.append({"Parameter":"p(D)", "Value":pd_s, "f(u)":fu_s})
        for t_s in np.linspace(0.001, 1.0, 20):
            tc_s = t_s + m_val/1000; fu_s = (pS_val*pD_val)/(tc_s+r_val+epsilon)
            sens_data.append({"Parameter":"Time f(t)", "Value":t_s, "f(u)":fu_s})
        for m_s in np.linspace(1, 500, 20):
            tc_s = t_val + m_s/1000; fu_s = (pS_val*pD_val)/(tc_s+r_val+epsilon)
            sens_data.append({"Parameter":"Memory g(M)", "Value":m_s, "f(u)":fu_s})
        fig_sens = px.line(pd.DataFrame(sens_data), x="Value", y="f(u)", color="Parameter",
                            markers=True, title="UCF Sensitivity Analysis")
        st.plotly_chart(fig_sens, use_container_width=True)

        st.markdown("---")
        st.subheader("📈 Monte Carlo Confidence Intervals for f(u)")
        n_sims    = st.slider("Number of simulations", 100, 5000, 1000)
        noise_pct = st.slider("Noise level (%)", 1, 30, 10)
        noise = noise_pct / 100.0
        np.random.seed(42)
        sim_pS = np.clip(np.random.normal(pS_val, pS_val*noise, n_sims), 0.01, 1.0)
        sim_pD = np.clip(np.random.normal(pD_val, pD_val*noise, n_sims), 0.01, 1.0)
        sim_t  = np.clip(np.random.normal(t_val,  max(t_val*noise, 1e-6), n_sims), 1e-9, None)
        sim_m  = np.clip(np.random.normal(m_val,  max(m_val*noise, 0.01), n_sims), 0.01, None)
        sim_r  = np.clip(np.random.normal(r_val,  max(r_val*noise, 1e-9), n_sims), 0.0,  None)
        sim_fu = (sim_pS * sim_pD) / (sim_t + sim_m/1000.0 + sim_r + epsilon)
        ci_lo, ci_hi = np.percentile(sim_fu, [2.5, 97.5])
        ci_mean = sim_fu.mean()

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Mean f(u)",    f"{ci_mean:.4f}")
        mc2.metric("Std Dev",      f"{sim_fu.std():.4f}")
        mc3.metric("95% CI Low",   f"{ci_lo:.4f}")
        mc4.metric("95% CI High",  f"{ci_hi:.4f}")

        fig_mc = px.histogram(sim_fu, nbins=50,
                               title=f"Monte Carlo f(u) Distribution (n={n_sims}, noise={noise_pct}%)",
                               labels={"value":"f(u)", "count":"Frequency"})
        fig_mc.add_vline(x=ci_lo,   line_dash="dash",  line_color="red",   annotation_text="2.5%")
        fig_mc.add_vline(x=ci_hi,   line_dash="dash",  line_color="red",   annotation_text="97.5%")
        fig_mc.add_vline(x=ci_mean, line_dash="solid", line_color="green", annotation_text="Mean")
        st.plotly_chart(fig_mc, use_container_width=True)

        if not st.session_state.results_df.empty:
            st.markdown("---")
            st.subheader("🔬 UCF Formula Comparison — Experimental vs Variants")
            df_lab = utlis.sanitise_df(st.session_state.results_df.copy())
            if all(c in df_lab.columns for c in ["pS","pD","TC_mean","R_mean"]):
                df_lab["fu_standard"] = (df_lab["pS"]*df_lab["pD"]) / (df_lab["TC_mean"]+df_lab["R_mean"]+1e-9)
                df_lab["fu_weighted"] = (df_lab["pS"]**1.2*df_lab["pD"]**0.8) / (df_lab["TC_mean"]+df_lab["R_mean"]+1e-9)
                df_lab["fu_harmonic"] = 2*df_lab["pS"]*df_lab["pD"]/(df_lab["pS"]+df_lab["pD"]+1e-9)/(df_lab["TC_mean"]+1e-9)
                cmp_grp = df_lab.groupby("Model")[["UCF_mean","fu_standard","fu_weighted","fu_harmonic"]].mean().reset_index()
                fig_cmp = px.bar(cmp_grp, x="Model", y=["UCF_mean","fu_standard","fu_weighted","fu_harmonic"],
                                  barmode="group", title="UCF Formula Variants Comparison")
                st.plotly_chart(fig_cmp, use_container_width=True)
                st.info("**Variants:** fu_standard = p(S)×p(D)/(TC+R+ε) | "
                        "fu_weighted = p(S)^1.2×p(D)^0.8/(TC+R+ε) | "
                        "fu_harmonic = harmonic(pS,pD)/TC")

        st.success("✅ UCF Formula Lab analysis complete.")
