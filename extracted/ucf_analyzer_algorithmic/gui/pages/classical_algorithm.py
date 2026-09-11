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

        st.header("Classical Algorithm Analysis")

        algorithms = {
            "Bubble Sort": bubble_sort
        }

        algo_choice = st.selectbox("Select Algorithm", list(algorithms.keys()))

        size = st.slider("Input Size", 100, 5000, 1000)

        if st.button("Run Sorting"):
            import psutil, os, tracemalloc as _tm, threading as _thr

            data = generate_data(size)
            proc = psutil.Process(os.getpid())

            # --- RAM snapshot before ---
            ram_before   = proc.memory_info().rss / 1024 / 1024
            proc.cpu_percent(interval=None)   # prime counter

            # --- Background CPU sampler during run_experiment ---
            _cpu_s   = []
            _stop_ev = _thr.Event()
            def _sampler():
                # cpu_percent(interval=0.1) blocks 100ms then returns accurate usage
                while not _stop_ev.is_set():
                    try: _cpu_s.append(proc.cpu_percent(interval=0.1))
                    except: pass
            _th = _thr.Thread(target=_sampler, daemon=True)
            _th.start()

            _tm.start()
            results = run_experiment(algorithms[algo_choice], data)
            cur, peak = _tm.get_traced_memory(); _tm.stop()

            _stop_ev.set(); _th.join(timeout=0.5)
            cpu_during = float(np.mean(_cpu_s)) if _cpu_s else 0.0
            cpu_peak   = float(np.max(_cpu_s))  if _cpu_s else 0.0

            ram_after  = proc.memory_info().rss / 1024 / 1024
            peak_heap  = peak / 1024 / 1024

            pS   = results["p(S)"]
            pD   = results["p(D)"]
            fu   = results["f(u)"]
            tc   = results["Total Complexity (TC)"]
            r    = results["Remainder (R)"]
            mem  = results["Memory (g(M))"]
            t    = results["Time (f(t))"]
            steps= results["Steps (S)"]

            # ── UCF Core ──
            st.markdown("---")
            st.subheader("🔬 UCF Core Metrics")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("p(S) — Stability",        f"{pS:.4f}",  help="Probability of stable behaviour")
            c2.metric("p(D) — Determinism",       f"{pD:.4f}",  help="Probability of deterministic output")
            c3.metric("f(u) — UCF Score",         f"{fu:.6f}",  help="Unified Complexity score (lower = more complex)")
            c4.metric("Steps (S)",                f"{steps:,}", help="Total algorithm steps counted")

            c5,c6,c7,c8 = st.columns(4)
            c5.metric("Time f(t) (s)",            f"{t:.6f}")
            c6.metric("TC(A,D) Total Complexity", f"{tc:.6f}",  help="TC = f(time, memory)")
            c7.metric("Remainder R",              f"{r:.6f}",   help="Complexity remainder from UCFEngine")
            c8.metric("Memory g(M) (MB)",         f"{mem:.4f}", help="Peak heap during algorithm (tracemalloc)")

            # ── System Metrics ──
            st.markdown("---")
            st.subheader("⚙️ System Metrics")
            s1,s2,s3,s4 = st.columns(4)
            s1.metric("CPU Avg During Sort (%)", f"{cpu_during:.1f}",
                      help="Average CPU % sampled every 100 ms during algorithm run")
            s2.metric("CPU Peak During Sort (%)",f"{cpu_peak:.1f}",
                      help="Peak CPU % spike observed during algorithm run")
            s3.metric("RAM Before (MB)", f"{ram_before:.2f}")
            s4.metric("RAM After (MB)",  f"{ram_after:.2f}")

            s5,s6 = st.columns(2)
            s5.metric("RAM Delta (MB)",  f"{ram_after - ram_before:.2f}", help="Extra RAM consumed during sort")
            s6.metric("Peak Heap (MB)",  f"{peak_heap:.4f}",              help="tracemalloc peak during run_experiment")

            # CPU bar chart
            cpu_sort_df = pd.DataFrame({
                "Metric": ["CPU Avg During Sort (%)", "CPU Peak During Sort (%)"],
                "Value":  [cpu_during, cpu_peak]
            })
            fig_cpu_sort = px.bar(cpu_sort_df, x="Metric", y="Value",
                                   color="Value", color_continuous_scale="reds",
                                   range_y=[0, max(100, cpu_peak + 5)],
                                   title="CPU Usage During Sorting (sampled every 100 ms)",
                                   text="Value")
            fig_cpu_sort.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig_cpu_sort, use_container_width=True)

            # ── Space Complexity ──
            st.markdown("---")
            st.subheader("🗂️ Space Complexity")
            sp1,sp2,sp3 = st.columns(3)
            sp1.metric("Input Size (n)",         f"{size:,}")
            sp2.metric("Peak Heap (MB)",         f"{peak_heap:.4f}")
            sp3.metric("Memory g(M) (MB)",       f"{mem:.4f}", help="Reported by UCFEngine")

            # ── Complexity Growth Chart ──
            st.markdown("---")
            st.subheader("📈 UCF Metrics at a Glance")
            glance = pd.DataFrame({
                "Metric": ["p(S)", "p(D)", "f(u) ×100", "TC ×100", "R ×100"],
                "Value":  [pS, pD, fu*100, tc*100, r*100]
            })
            fig_g = px.bar(glance, x="Metric", y="Value",
                           color="Value", color_continuous_scale="blues",
                           title="UCF Metric Overview (scaled for visibility)")
            st.plotly_chart(fig_g, use_container_width=True)

            # ── Big-O reference ──
            st.markdown("---")
            st.subheader("📐 Theoretical Big-O Reference")
            bo_df = pd.DataFrame({
                "Property":  ["Algorithm", "Best Case", "Average Case", "Worst Case", "Space"],
                "Value":     ["Bubble Sort", "O(n)", "O(n²)", "O(n²)", "O(1)"]
            })
            st.table(bo_df)

            # ── save ──
            row = pd.DataFrame([{
                "Model":       algo_choice,
                "pS":          pS,
                "pD":          pD,
                "UCF_mean":    fu,
                "Time_mean":   t,
                "TC_mean":     tc,
                "R_mean":      r,
                "Memory_mean": mem,
                "Steps":       steps,
                "CPU_Usage_pct":   cpu_during,
                "CPU_Peak_pct":    cpu_peak,
                "RAM_Before_MB":   ram_before,
                "RAM_After_MB":    ram_after,
                "RAM_Delta_MB":    ram_after - ram_before,
                "Peak_Heap_MB":    peak_heap,
                "Timestamp":       datetime.datetime.now()
            }])
            st.session_state.results_df = pd.concat(
                [st.session_state.results_df, row], ignore_index=True)
            st.session_state.sorting_done = True
            st.success("✅ Sorting analysis complete.")

    # =====================================================
    # 2️⃣ MACHINE LEARNING
    # =====================================================
