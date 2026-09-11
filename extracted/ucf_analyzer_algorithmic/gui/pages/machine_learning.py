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

        st.header("Machine Learning UCF Analysis")

        # Available models
        models = {
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

        model_choice = st.selectbox("Select Model", list(models.keys()))
        runs = st.slider("Number of Runs", 3, 10, 5)

        dataset_choice = st.selectbox(
            "Select Dataset",
            [
                "Synthetic Dataset",
                "Upload CSV (CIC-IDS)",
                "Custom CSV (any dataset)",
                "sklearn — Iris",
                "sklearn — Wine",
                "sklearn — Breast Cancer",
                "sklearn — Digits",
                "sklearn — MNIST (subset)",
            ]
        )

        # =====================================================
        # DATASET LOADING
        # =====================================================
        X_train, X_test, y_train, y_test = None, None, None, None

        # ── helper: preview a raw DataFrame ──
        def _show_raw_preview(raw_df):
            total_rows, total_cols = raw_df.shape
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Total Rows",     f"{total_rows:,}")
            p2.metric("Total Columns",  f"{total_cols:,}")
            p3.metric("Numeric Cols",   int(raw_df.select_dtypes(include=np.number).shape[1]))
            p4.metric("Missing Values", int(raw_df.isnull().sum().sum()))
            with st.expander("🔍 First 20 Rows", expanded=True):
                st.dataframe(raw_df.head(20), use_container_width=True)
            with st.expander("📊 Column Types & Null Counts", expanded=False):
                dtype_df = pd.DataFrame({
                    "Column":     raw_df.columns,
                    "dtype":      raw_df.dtypes.values.astype(str),
                    "Non-Null":   raw_df.notnull().sum().values,
                    "Null Count": raw_df.isnull().sum().values,
                    "Null %":     (raw_df.isnull().mean() * 100).round(2).values,
                    "Unique":     raw_df.nunique().values,
                })
                st.dataframe(dtype_df, use_container_width=True)
            with st.expander("📈 Descriptive Statistics", expanded=False):
                st.dataframe(
                    utlis.safe_gradient(utlis.sanitise_df(raw_df).describe().T.style, cmap="Blues"),
                    use_container_width=True
                )
            with st.expander("🔥 Feature Correlation Heatmap (top 20 numeric)", expanded=False):
                num_cols_raw = raw_df.select_dtypes(include=np.number).columns[:20]
                if len(num_cols_raw) >= 2:
                    corr_raw = raw_df[num_cols_raw].corr()
                    fig_corr_raw = px.imshow(corr_raw, text_auto=".2f",
                                              color_continuous_scale="RdBu_r", aspect="auto",
                                              title="Correlation Heatmap — Top 20 Numeric Features")
                    st.plotly_chart(fig_corr_raw, use_container_width=True)

        # ── helper: sklearn bundle → train/test split ──
        def _sklearn_split(X_arr, y_arr, feature_names_list):
            from sklearn.model_selection import train_test_split
            Xdf = pd.DataFrame(X_arr, columns=feature_names_list)
            ysr = pd.Series(y_arr)
            return train_test_split(Xdf, ysr, test_size=0.2, random_state=42, stratify=ysr)

        if dataset_choice == "Synthetic Dataset":
            samples = st.slider("Dataset Size", 500, 5000, 1000)
            X_train, X_test, y_train, y_test = generate_dataset(samples)
            feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
            X_train = pd.DataFrame(X_train, columns=feature_names)
            X_test  = pd.DataFrame(X_test,  columns=feature_names)

        elif dataset_choice == "Upload CSV (CIC-IDS)":
            uploaded_file = st.file_uploader("Upload CIC-IDS CSV File", type=["csv"])
            if uploaded_file is not None:
                st.markdown("---")
                st.subheader("📋 Uploaded Dataset Preview")
                try:
                    uploaded_file.seek(0)
                    raw_df = pd.read_csv(uploaded_file)
                    uploaded_file.seek(0)
                    _show_raw_preview(raw_df)
                    with st.expander("🏷️ Label / Target Column Distribution", expanded=False):
                        label_col = raw_df.columns[-1]
                        st.caption(f"Detected label column: **{label_col}**")
                        vc = raw_df[label_col].value_counts().reset_index()
                        vc.columns = ["Class", "Count"]
                        vc["Percentage"] = (vc["Count"] / vc["Count"].sum() * 100).round(2)
                        st.dataframe(vc, use_container_width=True)
                        fig_lbl = px.bar(vc, x="Class", y="Count", color="Class",
                                         title="Class Distribution", text="Percentage")
                        fig_lbl.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                        st.plotly_chart(fig_lbl, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not preview dataset: {e}")
                st.markdown("---")
                X_train, X_test, y_train, y_test = load_cic_ids(uploaded_file)
                for attr in ["X_train","X_test","y_train","y_test"]:
                    obj = locals()[attr]
                    if isinstance(obj, np.ndarray):
                        locals()[attr] = pd.DataFrame(obj) if "X" in attr else pd.Series(obj)
                X_train = X_train.reset_index(drop=True) if not isinstance(X_train, np.ndarray) else pd.DataFrame(X_train)
                y_train = y_train.reset_index(drop=True) if not isinstance(y_train, np.ndarray) else pd.Series(y_train)
                X_test  = X_test.reset_index(drop=True)  if not isinstance(X_test,  np.ndarray) else pd.DataFrame(X_test)
                y_test  = y_test.reset_index(drop=True)  if not isinstance(y_test,  np.ndarray) else pd.Series(y_test)
                max_samples = 5000
                if len(X_train) > max_samples:
                    sampled_idx = X_train.sample(max_samples, random_state=42).index
                    X_train = X_train.loc[sampled_idx].reset_index(drop=True)
                    y_train = y_train.loc[sampled_idx].reset_index(drop=True)
                feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
                X_train = pd.DataFrame(X_train.values, columns=feature_names)
                X_test  = pd.DataFrame(X_test.values,  columns=feature_names)
            else:
                st.warning("Please upload a CSV file to continue.")
                st.stop()

        elif dataset_choice == "Custom CSV (any dataset)":
            st.info("Upload **any** CSV. You will choose which column is the target label.")
            custom_file = st.file_uploader("Upload CSV", type=["csv"], key="custom_csv")
            if custom_file is not None:
                st.markdown("---")
                st.subheader("📋 Dataset Preview")
                try:
                    custom_file.seek(0)
                    raw_df = pd.read_csv(custom_file)
                    _show_raw_preview(raw_df)

                    st.markdown("---")
                    st.subheader("⚙️ Dataset Configuration")
                    cc1, cc2, cc3 = st.columns(3)

                    # Target column selector
                    target_col = cc1.selectbox(
                        "🎯 Select Target (label) column",
                        raw_df.columns.tolist(),
                        index=len(raw_df.columns) - 1
                    )

                    # Drop columns selector
                    drop_cols = cc2.multiselect(
                        "🗑️ Columns to drop (e.g. ID, timestamp)",
                        [c for c in raw_df.columns if c != target_col],
                        default=[]
                    )

                    # Test split size
                    test_size = cc3.slider("Test Split (%)", 10, 40, 20) / 100.0

                    # Max samples
                    max_custom = st.slider("Max training samples (0 = use all)", 0, 20000, 5000)

                    st.markdown("---")
                    # Show class distribution
                    with st.expander("🏷️ Target Column Distribution", expanded=True):
                        vc2 = raw_df[target_col].value_counts().reset_index()
                        vc2.columns = ["Class", "Count"]
                        vc2["Percentage"] = (vc2["Count"] / vc2["Count"].sum() * 100).round(2)
                        st.dataframe(vc2, use_container_width=True)
                        fig_vc2 = px.bar(vc2, x="Class", y="Count", color="Class",
                                         title=f"Class Distribution — {target_col}", text="Percentage")
                        fig_vc2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                        st.plotly_chart(fig_vc2, use_container_width=True)

                    # Process
                    proc_df = raw_df.drop(columns=drop_cols + [target_col], errors="ignore")
                    # Drop non-numeric columns (encode or skip)
                    non_num = proc_df.select_dtypes(exclude=np.number).columns.tolist()
                    if non_num:
                        st.warning(f"Dropping non-numeric columns: {non_num}. Encode them first if needed.")
                        proc_df = proc_df.select_dtypes(include=np.number)

                    # Fill NaN
                    proc_df = proc_df.fillna(proc_df.median(numeric_only=True))

                    X_all = proc_df
                    y_all = raw_df[target_col]

                    # Encode target if string
                    if y_all.dtype == object or str(y_all.dtype) == "category":
                        _le_custom = LabelEncoder()
                        y_all = pd.Series(_le_custom.fit_transform(y_all))

                    from sklearn.model_selection import train_test_split as _tts
                    X_train, X_test, y_train, y_test = _tts(
                        X_all, y_all,
                        test_size=test_size, random_state=42,
                        stratify=y_all if y_all.nunique() < 50 else None
                    )
                    X_train = X_train.reset_index(drop=True)
                    X_test  = X_test.reset_index(drop=True)
                    y_train = y_train.reset_index(drop=True)
                    y_test  = y_test.reset_index(drop=True)

                    if max_custom > 0 and len(X_train) > max_custom:
                        idx = X_train.sample(max_custom, random_state=42).index
                        X_train = X_train.loc[idx].reset_index(drop=True)
                        y_train = y_train.loc[idx].reset_index(drop=True)

                    feat_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
                    X_train.columns = feat_names
                    X_test.columns  = feat_names

                    st.success(f"✅ Dataset ready — {len(X_train):,} train / {len(X_test):,} test samples, "
                               f"{X_train.shape[1]} features, {y_all.nunique()} classes.")

                except Exception as e:
                    st.error(f"Could not process dataset: {e}")
                    st.stop()
            else:
                st.warning("Please upload a CSV file to continue.")
                st.stop()

        elif dataset_choice == "sklearn — Iris":
            from sklearn.datasets import load_iris
            _d = load_iris()
            X_train, X_test, y_train, y_test = _sklearn_split(_d.data, _d.target, list(_d.feature_names))
            st.info(f"✅ Iris dataset loaded — {len(_d.target)} samples, 4 features, 3 classes.")

        elif dataset_choice == "sklearn — Wine":
            from sklearn.datasets import load_wine
            _d = load_wine()
            X_train, X_test, y_train, y_test = _sklearn_split(_d.data, _d.target, list(_d.feature_names))
            st.info(f"✅ Wine dataset loaded — {len(_d.target)} samples, 13 features, 3 classes.")

        elif dataset_choice == "sklearn — Breast Cancer":
            from sklearn.datasets import load_breast_cancer
            _d = load_breast_cancer()
            X_train, X_test, y_train, y_test = _sklearn_split(_d.data, _d.target, list(_d.feature_names))
            st.info(f"✅ Breast Cancer dataset loaded — {len(_d.target)} samples, 30 features, 2 classes.")

        elif dataset_choice == "sklearn — Digits":
            from sklearn.datasets import load_digits
            _d = load_digits()
            feat_names = [f"pixel_{i}" for i in range(_d.data.shape[1])]
            X_train, X_test, y_train, y_test = _sklearn_split(_d.data, _d.target, feat_names)
            st.info(f"✅ Digits dataset loaded — {len(_d.target)} samples, 64 features, 10 classes.")

        elif dataset_choice == "sklearn — MNIST (subset)":
            from sklearn.datasets import fetch_openml
            with st.spinner("Fetching MNIST subset (2000 samples)..."):
                try:
                    _mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
                    _idx   = np.random.RandomState(42).choice(len(_mnist.target), 2000, replace=False)
                    _X     = _mnist.data[_idx].astype(float)
                    _y     = _mnist.target[_idx].astype(int)
                    feat_names = [f"pixel_{i}" for i in range(_X.shape[1])]
                    X_train, X_test, y_train, y_test = _sklearn_split(_X, _y, feat_names)
                    st.info(f"✅ MNIST subset loaded — 2000 samples, 784 features, 10 classes.")
                except Exception as e:
                    st.error(f"MNIST fetch failed: {e}. Try another dataset.")
                    st.stop()

        # =====================================================
        # DATASET OVERVIEW
        # =====================================================
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Training Samples", len(X_train))
        col2.metric("Testing Samples", len(X_test))
        col3.metric("Features", X_train.shape[1])

        # =====================================================
        # RUN ML EXPERIMENT
        # =====================================================
        if st.button("Run ML Analysis"):

            import psutil
            import os
            import tracemalloc
            from sklearn.metrics import (
                precision_score, recall_score, matthews_corrcoef,
                balanced_accuracy_score, cohen_kappa_score
            )

            # =====================================================
            # ENCODE LABELS
            # =====================================================
            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train)
            y_test_enc  = le.transform(y_test)

            # =====================================================
            # RUN UCF EXPERIMENT (multi-run averages)
            # =====================================================
            with st.spinner("Running ML experiment..."):
                results = run_ml_experiment(
                    models[model_choice],
                    X_train, X_test,
                    y_train_enc, y_test_enc,
                    runs
                )

            # =====================================================
            # TRAIN FINAL MODEL + CAPTURE ALL SYSTEM METRICS
            # =====================================================
            try:
                import threading as _threading
                proc = psutil.Process(os.getpid())

                # --- Snapshot before ---
                proc.cpu_percent(interval=None)   # prime the counter (first call is always 0)
                ram_before_mb = proc.memory_info().rss / 1024 / 1024

                # --- Background CPU sampler: blocks 100 ms per sample for accurate reading ---
                _cpu_samples  = []
                _stop_sampler = _threading.Event()

                def _cpu_sampler():
                    # cpu_percent(interval=0.1) blocks internally for 100ms and returns
                    # the actual CPU % used during that window — no busy-wait needed
                    while not _stop_sampler.is_set():
                        try:
                            _cpu_samples.append(proc.cpu_percent(interval=0.1))
                        except Exception:
                            pass

                _t = _threading.Thread(target=_cpu_sampler, daemon=True)
                _t.start()

                # --- tracemalloc for precise heap delta ---
                tracemalloc.start()

                start_time    = time.time()
                model         = models[model_choice]()
                model.fit(X_train, y_train_enc)
                training_time = time.time() - start_time

                # Stop CPU sampler and collect results
                _stop_sampler.set()
                _t.join(timeout=0.5)
                cpu_during    = float(np.mean(_cpu_samples)) if _cpu_samples else 0.0
                cpu_peak      = float(np.max(_cpu_samples))  if _cpu_samples else 0.0

                current_mem, peak_mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                peak_heap_mb  = peak_mem / 1024 / 1024

                # --- Snapshot after ---
                ram_after_mb  = proc.memory_info().rss / 1024 / 1024
                ram_delta_mb  = ram_after_mb - ram_before_mb

                # cpu_after kept for backward-compat with saved results_df column
                cpu_after     = cpu_during

                # --- Inference timing ---
                infer_start    = time.time()
                y_pred         = model.predict(X_test)
                inference_time = time.time() - infer_start

            except Exception as e:
                st.error(f"Model failed: {e}")
                st.stop()

            # =====================================================
            # CLASSIFICATION METRICS
            # =====================================================
            n_classes = len(np.unique(y_test_enc))
            avg_mode  = "binary" if n_classes == 2 else "weighted"

            f1        = f1_score(y_test_enc, y_pred, average="weighted")
            precision = precision_score(y_test_enc, y_pred, average=avg_mode, zero_division=0)
            recall    = recall_score(y_test_enc, y_pred, average=avg_mode, zero_division=0)
            mcc       = matthews_corrcoef(y_test_enc, y_pred)
            bal_acc   = balanced_accuracy_score(y_test_enc, y_pred)
            kappa     = cohen_kappa_score(y_test_enc, y_pred)

            # =====================================================
            # UCF DERIVED METRICS
            # Keys from experiment_runner.py:
            #   pS, pD, UCF_mean, TC_mean, R_mean,
            #   Time_mean, Time_std, Memory_mean,
            #   Accuracy_mean, F1_mean, Efficiency_Score
            # =====================================================
            epsilon       = 1e-9
            pS            = results.get("pS", 0)
            pD            = results.get("pD", 0)
            time_ft       = results.get("Time_mean", 0)
            time_std      = results.get("Time_std", 0)
            tc_ad         = results.get("TC_mean", 0)
            remainder_r   = results.get("R_mean", 0)
            memory_gm     = results.get("Memory_mean", 0)
            accuracy_mean = results.get("Accuracy_mean", 0)
            f1_mean       = results.get("F1_mean", 0)

            # f(u) already computed by UCFEngine across all runs — use directly
            fu = results.get("UCF_mean", 0)

            # Efficiency Score: computed here directly so it is never 0
            # Formula: balanced score of Accuracy + F1 normalised by UCF complexity cost
            # = (Accuracy + F1) / 2  ×  (pS + pD) / 2  /  (TC + R + ε)
            # Falls back to accuracy/fu if UCF components are zero
            if tc_ad > 0 or remainder_r > 0:
                efficiency = (
                    ((accuracy_mean + f1_mean) / 2.0) *
                    ((pS + pD) / 2.0) /
                    (tc_ad + remainder_r + epsilon)
                )
            elif fu > 0:
                efficiency = accuracy_mean / (fu + epsilon)
            else:
                efficiency = accuracy_mean  # plain accuracy as last resort

            # Real model size via pickle serialisation (getsizeof only gives Python shell ~56B)
            try:
                import pickle as _pkl
                model_size_mb = len(_pkl.dumps(model)) / 1024 / 1024
            except Exception:
                model_size_mb = 0.0

            space_complexity_mb = peak_heap_mb + model_size_mb

            # Throughput: samples predicted per second
            throughput = len(X_test) / (inference_time + epsilon)

            # =====================================================
            # ── DISPLAY: UCF CORE METRICS ──
            # =====================================================
            st.markdown("---")
            st.subheader("🔬 UCF Core Metrics")

            st.caption(
                "UCF Formula: **f(u) = UCFEngine(pS, pD, TC, R)** averaged over "
                f"{runs} runs  |  Time std = {time_std:.6f}s"
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("p(S)  — Stability",        f"{pS:.4f}",
                      help="Probability of stable behaviour across runs")
            c2.metric("p(D)  — Determinism",       f"{pD:.4f}",
                      help="Probability of deterministic output")
            c3.metric("f(u)  — UCF Score",         f"{fu:.6f}",
                      help="Unified Complexity Function score (lower = more complex)")
            c4.metric("Efficiency Score",           f"{efficiency:.4f}",
                      help="Accuracy / UCF_mean — higher is better")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Time f(t) mean (s)",         f"{time_ft:.6f}",
                      help=f"Avg over {runs} runs  |  std = {time_std:.6f}s")
            c6.metric("TC(A,D) — Total Complexity", f"{tc_ad:.6f}",
                      help="TC = f(time, memory) from UCFEngine")
            c7.metric("Remainder R",                f"{remainder_r:.6f}",
                      help="R = UCFEngine.compute_R(pS, pD, memory)")
            c8.metric("Memory g(M) (MB)",           f"{memory_gm:.4f}",
                      help="Peak heap MB averaged over runs (tracemalloc)")

            # =====================================================
            # ── DISPLAY: CLASSIFICATION METRICS ──
            # =====================================================
            st.markdown("---")
            st.subheader("📊 Classification Metrics")

            st.caption(
                f"Single-run scores (final model)  |  "
                f"Multi-run avg F1 from UCF runner = **{f1_mean:.4f}**"
            )

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Accuracy",           f"{accuracy_mean:.4f}",
                      help="Multi-run average from experiment runner")
            d2.metric("F1 (weighted)",      f"{f1:.4f}",
                      help=f"Final model  |  Runner avg = {f1_mean:.4f}")
            d3.metric("Precision",          f"{precision:.4f}")
            d4.metric("Recall",             f"{recall:.4f}")

            d5, d6, d7 = st.columns(3)
            d5.metric("MCC",                f"{mcc:.4f}",
                      help="Matthews Correlation Coefficient: ±1 range, robust for imbalanced classes")
            d6.metric("Balanced Accuracy",  f"{bal_acc:.4f}",
                      help="Average recall per class — better than accuracy for imbalanced data")
            d7.metric("Cohen's Kappa",      f"{kappa:.4f}",
                      help=">0.8 = excellent, 0.6–0.8 = good, <0.4 = poor agreement")

            # =====================================================
            # ── DISPLAY: SYSTEM / RUNTIME METRICS ──
            # =====================================================
            st.markdown("---")
            st.subheader("⚙️ System & Runtime Metrics")
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Training Time (s)",      f"{training_time:.4f}")
            e2.metric("Inference Time (s)",     f"{inference_time:.6f}")
            e3.metric("Throughput (samples/s)", f"{throughput:.1f}")
            e4.metric("CPU Avg During Train (%)", f"{cpu_during:.1f}",
                      help="Average CPU % sampled every 100 ms during model.fit()")

            e5, e6, e7, e8 = st.columns(4)
            e5.metric("RAM Before (MB)",        f"{ram_before_mb:.2f}")
            e6.metric("RAM After (MB)",         f"{ram_after_mb:.2f}")
            e7.metric("RAM Delta (MB)",         f"{ram_delta_mb:.2f}",
                      help="Extra RAM consumed by training")
            e8.metric("Peak Heap (MB)",         f"{peak_heap_mb:.4f}",
                      help="Peak heap allocation during training (tracemalloc)")

            # CPU usage bar chart (avg vs peak)
            cpu_chart_df = pd.DataFrame({
                "Metric": ["CPU Avg During Train (%)", "CPU Peak During Train (%)"],
                "Value":  [cpu_during, cpu_peak]
            })
            fig_cpu_bar = px.bar(cpu_chart_df, x="Metric", y="Value",
                                  color="Value", color_continuous_scale="reds",
                                  range_y=[0, max(100, cpu_peak + 5)],
                                  title="CPU Usage During Training (sampled every 100 ms)",
                                  text="Value")
            fig_cpu_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig_cpu_bar, use_container_width=True)

            # =====================================================
            # ── DISPLAY: SPACE COMPLEXITY ──
            # =====================================================
            st.markdown("---")
            st.subheader("🗂️ Space Complexity")
            f1c, f2c, f3c = st.columns(3)
            f1c.metric("Model Object Size (MB)",    f"{model_size_mb:.4f}")
            f2c.metric("Peak Heap During Fit (MB)", f"{peak_heap_mb:.4f}")
            f3c.metric("Space Complexity Total (MB)",f"{space_complexity_mb:.4f}",
                       help="Peak heap + model object size — practical space estimate")

            # =====================================================
            # ── DISPLAY: CHARTS ──
            # =====================================================

            # Metrics bar chart
            st.markdown("---")
            st.subheader("📈 Metrics at a Glance")
            glance_data = pd.DataFrame({
                "Metric": ["Accuracy", "F1", "Precision", "Recall", "Balanced Acc", "MCC (norm)", "Kappa"],
                "Value":  [
                    accuracy_mean, f1, precision, recall, bal_acc,
                    (mcc + 1) / 2,   # normalise MCC to 0-1 for the chart
                    max(kappa, 0)
                ]
            })
            fig_glance = px.bar(
                glance_data, x="Metric", y="Value",
                color="Value", color_continuous_scale="teal",
                range_y=[0, 1],
                title=f"{model_choice} — Classification Metrics Overview"
            )
            st.plotly_chart(fig_glance, use_container_width=True)

            # Runtime breakdown pie
            st.subheader("⏱️ Runtime Breakdown")
            runtime_df = pd.DataFrame({
                "Phase":   ["Training", "Inference"],
                "Time (s)":[training_time, inference_time]
            })
            fig_pie = px.pie(
                runtime_df, names="Phase", values="Time (s)",
                title="Training vs Inference Time",
                color_discrete_sequence=["#636EFA", "#EF553B"]
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # Memory waterfall
            st.subheader("🧠 Memory Usage Profile")
            mem_data = pd.DataFrame({
                "Stage":  ["RAM Before", "Peak Heap", "RAM After", "Model Size"],
                "MB":     [ram_before_mb, peak_heap_mb, ram_after_mb, model_size_mb]
            })
            fig_mem = px.bar(
                mem_data, x="Stage", y="MB",
                color="MB", color_continuous_scale="oranges",
                title="Memory Usage Across Stages"
            )
            st.plotly_chart(fig_mem, use_container_width=True)

            # =====================================================
            # CONFUSION MATRIX
            # =====================================================
            st.markdown("---")
            st.subheader("Confusion Matrix")
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                confusion_matrix(y_test_enc, y_pred),
                annot=True, fmt="d", cmap="Blues", ax=ax_cm
            )
            st.pyplot(fig_cm)
            plt.close(fig_cm)

            # =====================================================
            # ROC CURVE (Binary Only)
            # =====================================================
            if n_classes == 2 and hasattr(model, "predict_proba"):
                st.subheader("ROC Curve")
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test_enc, y_prob)
                roc_auc = auc(fpr, tpr)
                fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
                ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                ax_roc.plot([0, 1], [0, 1], linestyle="--")
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.legend()
                st.pyplot(fig_roc)
                plt.close(fig_roc)
            else:
                roc_auc = 0.0

            # =====================================================
            # SAVE RESULTS (all metrics — keyed to match runner output)
            # =====================================================
            new_row = pd.DataFrame([{
                "Model":               model_choice,
                # --- UCF metrics (keys match experiment_runner.py summary) ---
                "pS":                  pS,
                "pD":                  pD,
                "UCF_mean":            fu,           # = results["UCF_mean"]
                "Time_mean":           time_ft,
                "Time_std":            time_std,
                "TC_mean":             tc_ad,
                "R_mean":              remainder_r,
                "Memory_mean":         memory_gm,
                "Efficiency_Score":    efficiency,
                # --- Classification (runner averages + final-model singles) ---
                "Accuracy_mean":       accuracy_mean,
                "F1_mean":             f1_mean,      # multi-run avg from runner
                "F1_Score":            f1,           # final model (weighted)
                "Precision":           precision,
                "Recall":              recall,
                "MCC":                 mcc,
                "Balanced_Accuracy":   bal_acc,
                "Cohen_Kappa":         kappa,
                "ROC_AUC":             roc_auc,
                # --- Runtime (measured live in this run) ---
                "Training_Time":       training_time,
                "Inference_Time":      inference_time,
                "Throughput":          throughput,
                "CPU_Usage_pct":       cpu_during,
                "CPU_Peak_pct":        cpu_peak,
                # --- Memory / space ---
                "RAM_Before_MB":       ram_before_mb,
                "RAM_After_MB":        ram_after_mb,
                "RAM_Delta_MB":        ram_delta_mb,
                "Peak_Heap_MB":        peak_heap_mb,
                "Model_Size_MB":       model_size_mb,
                "Space_Complexity_MB": space_complexity_mb,
            }])

            st.session_state.results_df = pd.concat(
                [st.session_state.results_df, new_row],
                ignore_index=True
            )

            # Save for XAI page
            st.session_state.X_train       = X_train
            st.session_state.X_test        = X_test
            st.session_state.y_train       = y_train_enc
            st.session_state.y_test        = y_test_enc
            st.session_state.label_encoder = le

            st.session_state.ml_done = True
            st.success("✅ Analysis complete — scroll up to review all metrics.")

    # =====================================================
    # 3️⃣ UCF DASHBOARD
