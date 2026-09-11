# 🧠 AI-Based Algorithmic Analyzer — Unified Complexity Function (UCF)

<p align="center">
  <img src="extracted/ucf_analyzer_algorithmic/img/main.png"
       alt="AI-Based Algorithmic Analyzer — Unified Complexity Function"
       width="420">
</p>

<p align="center">
  <strong>Empirical analysis of algorithms and machine-learning models through performance, complexity, and explainability.</strong>
</p>

<p align="center">
  A Streamlit-based research platform combining empirical execution profiling,
  the Unified Complexity Function (UCF), classical complexity analysis,
  machine-learning evaluation, statistical analysis, and Explainable AI.
</p>

<p align="center">

<img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python">

<img src="https://img.shields.io/badge/Streamlit-Application-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">

<img src="https://img.shields.io/badge/UCF-Experimental Framework-8A6A52" alt="UCF">

<img src="https://img.shields.io/badge/Machine%20Learning-13%20Models-1A8763" alt="Machine Learning">

<img src="https://img.shields.io/badge/XAI-SHAP%20%7C%20LIME-B4790E" alt="Explainable AI">

<img src="https://img.shields.io/badge/Research-Academic-6A5ACD" alt="Academic Research">

</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-key-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-ucf-framework">UCF</a> •
  <a href="#-machine-learning-analysis">ML Analysis</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-research-context">Research</a>
</p>

---

## 📌 Overview

**AI-Based Algorithmic Analyzer (UCF)** is an academic research and experimental software platform designed to investigate the practical computational behavior of classical algorithms and machine-learning models.

The system combines:

* Theoretical complexity analysis
* Empirical execution profiling
* Unified Complexity Function (UCF)
* Machine-learning evaluation
* CPU and memory profiling
* Statistical analysis
* Comparative visualization
* Model comparison
* AutoML-oriented model selection
* Explainable AI using SHAP and LIME

The objective is to move beyond evaluating an algorithm or model using a single metric such as accuracy or Big-O notation.

Instead, the platform investigates computational behavior through multiple measurable dimensions, including:

> **Runtime + CPU + Memory + Execution Steps + Stability + Determinism + Predictive Performance + Explainability**

The project is intended primarily for **academic experimentation, master's-level research, algorithm benchmarking, machine-learning experimentation, and empirical computational analysis**.

---

## 🎓 Research Context

This project was developed as part of master's-level research into empirical computational complexity and machine-learning model behavior.

### Research Title

**A Unified Complexity Function Framework for Empirical Analysis of Machine Learning Algorithms with Explainable AI**

### Researcher

**Aqib Ali Buriro**

Master of Data Science
Mehran University of Engineering and Technology
Jamshoro, Sindh, Pakistan

### Supervisors

**Dr. Bushra Naz**
**Dr. Sammer Zai**

---

# 🚀 Key Features

## 1. Classical Algorithm Analysis

The platform provides an experimental environment for analysing classical algorithms using practical system measurements.

Current workflow includes:

* Algorithm selection
* Input-size configuration
* Runtime measurement
* CPU monitoring
* RAM monitoring
* Peak memory/heap measurement
* Execution-step measurement
* UCF computation
* Big-O reference
* Complexity visualization
* Experiment result analysis

### Current classical algorithm

* Bubble Sort

The architecture is designed so that additional algorithms can be integrated into the `algorithms/` module.

---

## 2. Machine-Learning Analysis

The ML module allows different machine-learning models to be compared using both predictive performance and computational behavior.

### Supported Models

| Model                           | Category          |
| ------------------------------- | ----------------- |
| Decision Tree                   | Tree-based        |
| Random Forest                   | Ensemble          |
| Logistic Regression             | Linear            |
| K-Nearest Neighbors             | Instance-based    |
| Support Vector Machine          | Kernel-based      |
| Gradient Boosting               | Ensemble          |
| XGBoost                         | Gradient boosting |
| LightGBM                        | Gradient boosting |
| AdaBoost                        | Ensemble          |
| Extra Trees                     | Ensemble          |
| Linear Discriminant Analysis    | Statistical       |
| Quadratic Discriminant Analysis | Statistical       |
| MLP Neural Network              | Neural network    |

This provides a **13-model comparative ML analysis environment**.

---

# 📊 Performance Metrics

The platform can evaluate models and experiments across multiple dimensions.

## Computational Metrics

* Execution time
* Training time
* Inference time
* CPU utilization
* Peak CPU
* RAM before execution
* RAM after execution
* RAM delta
* Peak heap
* Throughput
* Model size
* Practical space-complexity estimates
* Execution steps

## Classification Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Matthews Correlation Coefficient (MCC)
* Balanced Accuracy
* Cohen's Kappa
* ROC-AUC where applicable

## UCF Metrics

* `p(S)`
* `p(D)`
* `f(t)`
* `g(M)`
* `f(u)`
* Total Complexity (TC)
* Complexity Remainder (R)
* Execution steps

---

# 🧮 Unified Complexity Function (UCF)

The central research concept implemented by this project is the **Unified Complexity Function**.

The current formulation is:

```text
f(u) = f(t) + p(s) - p(d)
```

Where:

| Symbol | Interpretation                              |
| ------ | ------------------------------------------- |
| `f(u)` | Unified Complexity Function output          |
| `f(t)` | Execution-time component                    |
| `p(s)` | Stability/sequence-related component        |
| `p(d)` | Determinism/complexity-difference component |

The exact interpretation and calculation of these components depend on the implementation and experimental methodology used by the project.

---

# ⚖️ UCF vs Big-O

UCF and Big-O should not be interpreted as competing replacements for one another.

They provide different perspectives.

```text
                    Algorithm / Model
                           │
             ┌─────────────┴─────────────┐
             │                           │
       Theoretical                  Empirical
        Analysis                    Execution
             │                           │
          Big-O              Time / CPU / Memory / Steps
             │                           │
             │                           ▼
             │                          UCF
             │                           │
             └─────────────┬─────────────┘
                           ▼
                  Practical Interpretation
```

### Big-O

Big-O primarily describes the asymptotic growth of computational resources as input size increases.

### UCF

UCF is an empirical framework implemented by this project for organizing experimentally observed computational behavior using the project's defined measurements.

### Important Research Note

> **UCF does not replace formal computational-complexity analysis or mathematical proofs of Big-O complexity.**

Instead, it is intended to provide an additional empirical perspective.

Experimental UCF values are dependent on factors such as:

* Hardware
* Operating system
* Python/runtime environment
* Dataset
* Implementation
* System load
* Measurement methodology
* Number of experimental repetitions

---

# 🔬 How the Analyzer Works

The general experimental pipeline is:

```text
Problem / Dataset
       │
       ▼
Algorithm / ML Model
       │
       ▼
Experiment Runner
       │
       ▼
Performance Profiler
       │
       ├── Runtime
       ├── CPU
       ├── Memory
       ├── Heap
       └── Execution Steps
       │
       ▼
UCF Computation
       │
       ├── Time Component
       ├── Stability
       ├── Determinism
       └── Complexity Measures
       │
       ▼
ML Evaluation
       │
       ▼
Statistical Analysis
       │
       ▼
Visualization
       │
       ▼
Research Results
```

---

# 🧪 Experimental Workflow

A typical experiment follows this process:

### Step 1 — Define the experiment

Select:

* Algorithm
* ML model
* Dataset
* Input size
* Number of repetitions
* Experimental configuration

### Step 2 — Configure the dataset

Configure:

* Target column
* Features
* Columns to remove
* Train/test split
* Maximum training samples
* Preprocessing

### Step 3 — Execute

Run the selected algorithm or ML model.

### Step 4 — Profile

Measure:

* Runtime
* CPU
* Memory
* Heap
* Execution steps
* Throughput

### Step 5 — Calculate UCF

Apply the project's UCF formulation to the experimental measurements.

### Step 6 — Evaluate ML performance

Where applicable, calculate:

* Accuracy
* Precision
* Recall
* F1
* MCC
* Balanced Accuracy
* Kappa
* ROC-AUC

### Step 7 — Analyse statistically

Compare repeated experiments and model behavior.

### Step 8 — Visualize

Generate:

* Comparative charts
* Distributions
* Heatmaps
* Model comparisons
* Complexity visualizations

### Step 9 — Interpret

Compare empirical observations with:

* Theoretical complexity
* Predictive performance
* Computational resource usage
* Explainability results

---

# 🤖 Machine-Learning Analysis

The ML workflow is:

```text
Dataset
   │
   ▼
Preprocessing
   │
   ▼
Train/Test Split
   │
   ▼
Model Training
   │
   ▼
Performance Profiling
   │
   ▼
Prediction
   │
   ▼
Evaluation Metrics
   │
   ▼
UCF / Complexity Analysis
   │
   ▼
Model Comparison
```

This allows a model to be evaluated not only according to predictive accuracy but also according to its computational behavior.

For example:

```text
Model A
Accuracy:       99%
Training Time:  Low
Memory:         Low
UCF:            ...
        │
        ▼
Practical Evaluation
```

versus:

```text
Model B
Accuracy:       99.2%
Training Time:  High
Memory:         High
UCF:            ...
        │
        ▼
Practical Evaluation
```

This supports a more multidimensional interpretation of model performance.

---

# 📚 Dataset Support

The project is designed to work with several dataset sources.

### Built-in / Supported Sources

* Synthetic datasets
* CIC-IDS CSV data
* Custom CSV datasets
* Scikit-learn Iris
* Scikit-learn Wine
* Scikit-learn Breast Cancer
* Scikit-learn Digits
* MNIST subset

### Custom CSV Configuration

Custom datasets can be configured using:

* Target column
* Columns to drop
* Train/test split
* Maximum training samples
* Dataset preprocessing
* Feature configuration

Large datasets should be configured according to available system memory and computational resources.

---

# 🧠 Explainable AI

The project includes an `xai/` component for model interpretation.

The system is designed to support:

* **SHAP**
* **LIME**

The general XAI workflow is:

```text
Trained Model
      │
      ▼
XAI Engine
      │
      ▼
Feature Contributions
      │
      ├───────────────┐
      ▼               ▼
Global Explanation  Local Explanation
      │               │
      └───────┬───────┘
              ▼
        Interpretation
```

### Why XAI?

Predictive performance alone does not explain why a model produces a particular prediction.

Explainability can provide information about:

* Important features
* Feature contributions
* Global model behavior
* Individual predictions
* Model decision patterns

XAI results should always be interpreted in relation to the selected model, dataset, preprocessing pipeline, and explanation method.

---

# 📈 Statistical Analysis

The platform is designed for empirical comparison across:

* Algorithms
* ML models
* Datasets
* Input sizes
* Repeated experiments
* Computational metrics

Statistical functionality includes or is intended to support methods such as:

* Welch's t-test
* Mann-Whitney U test
* One-way ANOVA
* Kruskal-Wallis test
* Levene's test
* Correlation analysis
* Model-wise averages
* Distribution analysis
* Heatmaps
* Box plots
* Comparative bar charts

### Statistical caution

Statistical significance should not automatically be interpreted as practical significance.

The appropriate statistical test depends on:

* Experimental design
* Sample size
* Distributional assumptions
* Independence
* Variance characteristics
* Number of repetitions

---

# 🖥️ Application Modules

The Streamlit application is organized around dedicated analysis modules.

| Module                  | Purpose                              |
| ----------------------- | ------------------------------------ |
| Home                    | Application overview                 |
| Classical Algorithm     | Classical algorithm profiling        |
| Machine Learning        | ML model experimentation             |
| UCF Dashboard           | UCF-based analysis                   |
| Model Comparison        | Compare ML models                    |
| Performance Metrics     | Computational and predictive metrics |
| Explainable AI          | SHAP/LIME analysis                   |
| UCF vs Big-O            | Empirical vs theoretical comparison  |
| AutoML Selection        | Model selection and comparison       |
| Advanced Visualizations | Advanced analytical visualization    |
| UCF Formula Lab         | UCF formula experimentation          |
| About                   | Project and research information     |

---

# 🏗️ Architecture

The project follows a modular architecture separating computational logic from the Streamlit presentation layer.

```text
┌───────────────────────────────┐
│       Streamlit UI            │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│          GUI Pages            │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Experiment / Service Logic    │
└───────────────┬───────────────┘
                │
        ┌───────┴────────┐
        ▼                ▼
┌─────────────┐   ┌──────────────┐
│ Algorithms  │   │ ML Models    │
└──────┬──────┘   └──────┬───────┘
       │                 │
       └────────┬────────┘
                ▼
       ┌─────────────────┐
       │ Performance     │
       │ Profiler        │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │ UCF Engine      │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │ Results         │
       │ & Visualization │
       └─────────────────┘
```

---

# 📁 Project Structure

The main application is organized into modular components:

```text
ucf_analyzer_algorithmic/
│
├── .devcontainer/
│
├── .github/
│   └── workflows/
│
├── extracted/
│   └── ucf_analyzer_algorithmic/
│       │
│       ├── app.py
│       ├── main.py
│       ├── README.md
│       ├── requirements.txt
│       ├── .env.example
│       ├── .gitignore
│       │
│       ├── algorithms/
│       ├── experiments/
│       ├── profiler/
│       ├── ucf/
│       ├── xai/
│       ├── ml_models/
│       ├── dataset/
│       ├── img/
│       │
│       └── gui/
│           ├── layout.py
│           ├── state.py
│           ├── utils.py
│           ├── components.py
│           │
│           └── pages/
│               ├── home.py
│               ├── classical_algorithm.py
│               ├── machine_learning.py
│               ├── ucf_dashboard.py
│               ├── model_comparison.py
│               ├── performance_metrics.py
│               ├── explainable_ai.py
│               ├── ucf_vs_bigo.py
│               ├── automl_selection.py
│               ├── advanced_visualizations.py
│               ├── ucf_formula_lab.py
│               └── about.py
│
├── LICENSE.md
├── README.md
└── ucf_analyzer_algorithmic.rar
```

---

# 🧩 Module Responsibilities

| Module         | Responsibility                                   |
| -------------- | ------------------------------------------------ |
| `app.py`       | Main Streamlit application entry point           |
| `algorithms/`  | Classical algorithms and related components      |
| `experiments/` | Experiment execution and repeated runs           |
| `profiler/`    | Runtime, CPU, RAM, heap, and execution profiling |
| `ucf/`         | UCF calculations and complexity analysis         |
| `ml_models/`   | Machine-learning model implementations           |
| `xai/`         | SHAP/LIME and explainability functionality       |
| `dataset/`     | Dataset loading and preparation                  |
| `gui/`         | Streamlit presentation layer                     |
| `gui/pages/`   | Individual application pages                     |
| `img/`         | Screenshots, logos, diagrams, and visual assets  |

---

# ⚙️ Requirements

Recommended environment:

* Python 3.x
* pip
* Git
* Virtual environment
* Windows, Linux, or macOS

The exact Python package requirements are defined in:

```text
requirements.txt
```

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/EricAqib123/ucf_analyzer_algorithmic.git
```

Move into the repository:

```bash
cd ucf_analyzer_algorithmic
```

---

## 2. Create a virtual environment

### Windows

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

From the project root:

```bash
python -m streamlit run app.py
```

If your local project configuration uses `streamlit_app.py` as the entry point:

```bash
streamlit run streamlit_app.py
```

The application should normally be available at:

```text
http://localhost:8501
```

> **Important:** Start Streamlit from the project root rather than from inside the `gui/` directory.

---

# 🔐 Environment Configuration

If environment variables are required, create a `.env` file in the project root.

Example:

```env
APP_ENV=development
DEBUG=false

OPENAI_API_KEY=
GROQ_API_KEY=
```

Only configure variables actually required by your installation.

### Security

Never commit real API keys, passwords, or credentials.

❌ Do not do this:

```python
api_key = "secret-value"
```

✅ Use environment variables:

```python
import os

api_key = os.getenv("OPENAI_API_KEY")
```

Keep `.env` excluded from Git.

Use `.env.example` for safe placeholders.

---

# 🧪 Reproducible Research

Because this project is intended for academic experimentation, experiment configuration should be recorded carefully.

A recommended experiment record contains:

```text
Experiment ID
Date
Dataset
Dataset Size
Algorithm / Model
Input Size
Number of Runs
Train/Test Split
Preprocessing
Execution Time
CPU Usage
Memory Usage
Peak Heap
UCF Metrics
Accuracy
Precision
Recall
F1-score
Other Evaluation Metrics
Hardware
Operating System
Python Version
Library Versions
Experiment Configuration
```

For meaningful comparisons, keep the following consistent whenever appropriate:

* Dataset version
* Preprocessing
* Train/test split
* Number of repetitions
* Hardware
* Python version
* Library versions
* Experimental configuration

---

# 🔁 Recommended Experiment Design

Repeated experiments are important because runtime and system-level measurements can vary due to:

* Operating-system scheduling
* Background processes
* CPU frequency changes
* Available memory
* Cache behavior
* Dataset size
* Hardware differences
* System load

A recommended experimental structure is:

```text
Experiment
│
├── Configuration
│
├── Warm-up / Preparation
│
├── Repeated Executions
│
├── Raw Measurements
│
├── Aggregated Statistics
│
├── UCF Computation
│
├── Statistical Tests
│
└── Visualization
```

---

# 🔍 Verify the Installation

After installation:

```bash
python -m streamlit run app.py
```

You can also verify the major Python modules:

```bash
python -c "import algorithms, experiments, profiler, ucf"
```

If the command completes without an import error, the major project packages are accessible from the current environment.

---

# 🛠️ Troubleshooting

## `ModuleNotFoundError: No module named 'algorithms'`

Make sure you are running the application from the project root.

Check that the module exists:

```text
algorithms/
    __init__.py
```

Then run:

```bash
python -m streamlit run app.py
```

Avoid launching the application from inside:

```text
gui/
```

---

## Dependency Errors

Activate your virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If a package fails to install, check compatibility between the package version and your Python version.

---

## Dataset Errors

For custom datasets, check:

* File format
* Encoding
* Target column
* Missing values
* Feature types
* Dataset size
* Available memory

Make sure the configured target column exists in the CSV file.

---

## MNIST Download Errors

Depending on the dataset implementation, MNIST may require internet connectivity.

If MNIST cannot be downloaded, use another dataset such as:

* Iris
* Wine
* Breast Cancer
* Digits
* Synthetic datasets
* CIC-IDS
* Custom CSV

---

## XAI Errors

Verify that:

1. The required XAI packages are installed.
2. The selected model is supported by the explanation method.
3. The input data has the expected format.
4. Preprocessing is compatible with the selected XAI method.

SHAP and LIME may require different model interfaces and data preparation.

---

## Inconsistent Runtime or Memory Measurements

System-level measurements can vary significantly depending on the execution environment.

For more reliable experiments:

* Close unnecessary applications.
* Use repeated runs.
* Keep hardware conditions consistent.
* Record system specifications.
* Record Python/library versions.
* Avoid comparing experiments performed under substantially different system loads.

---

# 👨‍💻 Development Guidelines

The preferred architecture separates computational responsibilities from the Streamlit interface.

```text
Streamlit UI
     ↓
GUI Page
     ↓
Experiment / Service Logic
     ↓
Algorithm / ML Model
     ↓
Profiler / UCF
     ↓
Results
     ↓
Visualization
```

Avoid placing large computational functions directly inside Streamlit page files.

This makes the project easier to:

* Test
* Extend
* Maintain
* Benchmark
* Reuse
* Debug

---

# ➕ Adding a New Algorithm or ML Model

To add a new algorithm or model:

1. Implement it in the appropriate module.
2. Test it independently.
3. Connect it to the experiment runner.
4. Add it to the relevant UI selection.
5. Verify profiling.
6. Verify UCF calculations.
7. Add appropriate evaluation metrics.
8. Verify visualizations.
9. Confirm result storage.
10. Test the complete workflow through Streamlit.

---

# 🔬 Research Framework

The project connects four complementary perspectives:

```text
┌──────────────────────────────┐
│ Theoretical Complexity       │
└──────────────┬───────────────┘
               │
               +
               ▼
┌──────────────────────────────┐
│ Empirical Execution Behavior │
└──────────────┬───────────────┘
               │
               +
               ▼
┌──────────────────────────────┐
│ Machine Learning Performance │
└──────────────┬───────────────┘
               │
               +
               ▼
┌──────────────────────────────┐
│ Explainable AI               │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Practical Algorithm Analysis │
└──────────────────────────────┘
```

The goal is not simply to identify the algorithm or model with the highest score.

Instead, the system investigates:

> **How do different computational approaches behave under measurable experimental conditions?**

---

# ⚠️ Research Limitations

Results generated by this platform should be interpreted as experimental evidence.

Important limitations include:

* Runtime varies with system scheduling.
* CPU measurements depend on hardware and system load.
* Memory measurements depend on the execution environment.
* Dataset characteristics affect ML performance.
* Different models may require different preprocessing.
* UCF values depend on the defined experimental methodology.
* Big-O and empirical measurements represent different perspectives.
* XAI explanations depend on the model, dataset, preprocessing, and explanation technique.
* A single experiment is generally insufficient for a reliable performance conclusion.
* Results from one hardware environment may not generalize to another.
* Large datasets may substantially increase memory and execution requirements.

### Important distinction

**UCF is an empirical framework implemented by this research project.**

It should not be presented as a universal replacement for formal computational complexity theory.

---

# 🚧 Future Development

Potential future improvements include:

* [ ] Additional classical algorithms
* [ ] Additional ML models
* [ ] Deep-learning models
* [ ] GPU profiling
* [ ] Larger automated benchmarking
* [ ] Experiment-history database
* [ ] Automated PDF reports
* [ ] Automated Excel reports
* [ ] Additional XAI methods
* [ ] Cloud-based experiments
* [ ] Distributed benchmarking
* [ ] Expanded AutoML functionality
* [ ] Additional statistical validation
* [ ] Hardware-aware benchmarking
* [ ] Versioned datasets
* [ ] Reproducible experiment packages
* [ ] Automated research-report generation

---

# 📋 Quick Start

```text
1. Clone the repository
        ↓
2. Open terminal in project root
        ↓
3. Create virtual environment
        ↓
4. Activate environment
        ↓
5. Install requirements
        ↓
6. Start Streamlit
        ↓
7. Select analysis module
        ↓
8. Configure experiment
        ↓
9. Run analysis
        ↓
10. Inspect performance
        ↓
11. Inspect UCF
        ↓
12. Compare models / algorithms
        ↓
13. Review statistical results
        ↓
14. Review XAI results
```

---

# 📖 Research Use

This repository is intended to support academic experimentation involving:

* Algorithm analysis
* Computational complexity
* Machine-learning benchmarking
* Empirical performance analysis
* Explainable AI
* Statistical model comparison
* Resource profiling
* Reproducible experimentation

Researchers using the framework should report sufficient experimental details to allow results to be interpreted and reproduced.

---

# 📄 Citation

If you use this software, UCF framework, methodology, or experimental approach in academic research, please cite the associated research work.

### Research Article

**A Unified Complexity Function Framework for Empirical Analysis of Machine Learning Algorithms with Explainable AI**

**Author:** Aqib Ali Buriro

Publication information and DOI can be added here once the associated research article is formally published.

Example:

```bibtex
@article{buriro_ucf_ml_xai,
  title   = {A Unified Complexity Function Framework for Empirical Analysis of Machine Learning Algorithms with Explainable AI},
  author  = {Buriro, Aqib Ali},
  year    = {2026},
  note    = {Research software repository}
}
```

---

# 👤 Author

### Aqib Ali Buriro

**Master of Data Science**
**Mehran University of Engineering and Technology**
Jamshoro, Sindh, Pakistan

GitHub:

https://github.com/EricAqib123

---

# 👨‍🏫 Supervisors

**Dr. Bushra Naz**

**Dr. Sammer Zai**

Mehran University of Engineering and Technology
Jamshoro, Sindh, Pakistan

---

# 📜 License

See [`LICENSE.md`](LICENSE.md) for the applicable license and usage terms.

---

# ⭐ Acknowledgement

This project was developed as part of academic research into empirical computational complexity, machine-learning performance, and Explainable AI.

The project aims to provide a practical experimental environment where theoretical analysis and real-world computational measurements can be studied together.

---

<p align="center">
  <strong>AI-Based Algorithmic Analyzer — UCF</strong>
  <br>
  Empirical Complexity • Machine Learning • Performance Profiling • Explainable AI
</p>

<p align="center">
  Made for research, experimentation, and reproducible computational analysis.
</p>
