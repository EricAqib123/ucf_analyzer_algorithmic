import numpy as np
import tracemalloc
from profiler import time_profiler
from profiler.step_counter import StepCounter
from ucf.ucf_engine import UCFEngine
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

# ------------------------------
# Classical Algorithm Experiment
# ------------------------------
def run_experiment(algorithm, data):

    counter = StepCounter()
    ucf = UCFEngine()

    result, exec_time = time_profiler.measure_time(algorithm, data.copy(), counter)

    tracemalloc.start()
    algorithm(data.copy(), counter)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    memory_usage = peak / (1024 * 1024)

    steps = counter.get_steps()

    pS  = ucf.compute_pS(steps, exec_time)
    pD  = ucf.compute_pD(exec_time, pS)
    f_u = ucf.compute_fu(exec_time, pS, pD)
    TC  = ucf.compute_TC(exec_time, memory_usage)
    R   = ucf.compute_R(pS, pD, memory_usage)

    return {
        "Time (f(t))":           exec_time,
        "Memory (g(M))":         memory_usage,
        "Steps (S)":             steps,
        "p(S)":                  pS,
        "p(D)":                  pD,
        "f(u)":                  f_u,
        "Total Complexity (TC)": TC,
        "Remainder (R)":         R
    }


# ------------------------------
# Machine Learning Experiment
# ------------------------------
def run_ml_experiment(model_func, X_train, X_test, y_train, y_test, runs=10):

    ucf     = UCFEngine()
    results = []

    for _ in range(runs):

        model = model_func()

        tracemalloc.start()

        # Train
        _, train_time = time_profiler.measure_time(model.fit, X_train, y_train)

        # Predict
        y_pred, predict_time = time_profiler.measure_time(model.predict, X_test)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        memory_usage = peak / (1024 * 1024)
        total_time   = train_time + predict_time
        steps        = X_train.shape[0]

        # Accuracy metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1       = f1_score(y_test, y_pred, average="weighted")

        # UCF Calculations
        pS  = ucf.compute_pS(steps, total_time)
        pD  = ucf.compute_pD(total_time, pS)
        f_u = ucf.compute_fu(total_time, pS, pD)
        TC  = ucf.compute_TC(total_time, memory_usage)
        R   = ucf.compute_R(pS, pD, memory_usage)

        results.append({
            "Time":     total_time,
            "Memory":   memory_usage,
            "pS":       pS,
            "pD":       pD,
            "f(u)":     f_u,
            "TC":       TC,
            "R":        R,
            "Accuracy": accuracy,
            "F1":       f1
        })

    # ── Summary statistics ──
    ucf_vals      = [r["f(u)"]     for r in results]
    accuracy_vals = [r["Accuracy"] for r in results]
    f1_vals       = [r["F1"]       for r in results]
    tc_vals       = [r["TC"]       for r in results]
    r_vals        = [r["R"]        for r in results]
    pS_vals       = [r["pS"]       for r in results]
    pD_vals       = [r["pD"]       for r in results]

    ucf_mean      = float(np.mean(ucf_vals))
    accuracy_mean = float(np.mean(accuracy_vals))
    f1_mean       = float(np.mean(f1_vals))
    tc_mean       = float(np.mean(tc_vals))
    r_mean        = float(np.mean(r_vals))
    pS_mean       = float(np.mean(pS_vals))
    pD_mean       = float(np.mean(pD_vals))

    summary = {
        "Time_mean":     float(np.mean([r["Time"]   for r in results])),
        "Time_std":      float(np.std( [r["Time"]   for r in results])),
        "Memory_mean":   float(np.mean([r["Memory"] for r in results])),
        "pS":            pS_mean,
        "pD":            pD_mean,
        "UCF_mean":      ucf_mean,
        "TC_mean":       tc_mean,
        "R_mean":        r_mean,
        "Accuracy_mean": accuracy_mean,
        "F1_mean":       f1_mean,
    }

    # ── Efficiency Score ──
    # Computed from values guaranteed to be non-zero after any successful run:
    # pS, pD (UCF stability/determinism), Accuracy, F1 (classification quality)
    # and Time (runtime cost — always > 0).
    #
    # Formula:
    #   Efficiency = (Accuracy × F1 × pS × pD) / (Time_mean + ε)
    #
    # Interpretation: high accuracy+F1 with high UCF stability at low runtime = high score.
    # Normalised to a comparable scale across models since all inputs are bounded.

    epsilon = 1e-9
    time_mean = summary["Time_mean"]

    efficiency = (accuracy_mean * f1_mean * pS_mean * pD_mean) / (time_mean + epsilon)

    summary["Efficiency_Score"] = float(efficiency)

    return summary