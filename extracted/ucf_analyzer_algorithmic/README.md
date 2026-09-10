# AI-Based Algorithmic Analyzer (UCF)

## Overview

**AI-Based Algorithmic Analyzer (UCF)** is a Streamlit-based academic
research and experimental platform for analyzing the practical
performance and computational complexity of classical algorithms and
machine-learning models.

The system uses the **Unified Complexity Function (UCF)** approach and
combines theoretical complexity analysis with empirical execution
profiling.

### Main measurements

-   Execution time
-   CPU utilization
-   Memory usage
-   Runtime behavior
-   Algorithm/model steps
-   Stability and determinism
-   UCF score
-   Total complexity
-   Remainder
-   Accuracy, precision, recall and F1-score
-   MCC, Balanced Accuracy and Cohen's Kappa
-   ROC-AUC where applicable
-   Throughput
-   Model size
-   Practical space complexity
-   Explainable AI results

## UCF Model

The project uses the following UCF formulation:

``` text
f(u) = f(t) + p(s) - p(d)
```

Where:

-   `f(u)` = Unified Complexity Function output
-   `f(t)` = execution-time component
-   `p(s)` = stability/sequence-related component
-   `p(d)` = determinism/complexity-difference component

UCF results should be interpreted according to the implementation and
experimental methodology of this project. UCF complements rather than
replaces formal asymptotic analysis such as Big-O.

## Main Features

### Classical Algorithm Analysis

The application provides practical profiling of classical algorithms,
including:

-   Algorithm selection
-   Input-size selection
-   Execution-time measurement
-   CPU monitoring
-   RAM monitoring
-   Peak heap measurement
-   Step counting
-   UCF calculation
-   Big-O reference
-   Complexity visualization
-   Experiment-result storage

The current implementation includes Bubble Sort.

### Machine Learning Analysis

Supported models include:

1.  Decision Tree
2.  Random Forest
3.  Logistic Regression
4.  K-Nearest Neighbors
5.  Support Vector Machine
6.  Gradient Boosting
7.  XGBoost
8.  LightGBM
9.  AdaBoost
10. Extra Trees
11. Linear Discriminant Analysis
12. Quadratic Discriminant Analysis
13. MLP Neural Network

### Dataset Support

The application supports:

-   Synthetic datasets
-   CIC-IDS CSV data
-   Custom CSV datasets
-   Scikit-learn Iris
-   Scikit-learn Wine
-   Scikit-learn Breast Cancer
-   Scikit-learn Digits
-   MNIST subset

For custom CSV files, the application can configure the target column,
columns to drop, test split, and maximum training samples.

## Application Pages

The organized application contains pages for:

``` text
Home
Classical Algorithm
Machine Learning
UCF Dashboard
Model Comparison
Performance Metrics
Explainable AI
UCF VS BigO Analysis
AutoML Selection
Advanced Visualizations
UCF Formula Lab
About
```

## Project Structure

``` text
AI-Based-Algorithmic-Analyzer/
│
├── app.py
├── main.py
├── README.md
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
│
├── algorithms/
├── experiments/
├── profiler/
├── ucf/
├── xai/
├── ml_models/
├── dataset/
├── img/
│
├── gui/
│   ├── __init__.py
│   ├── layout.py
│   ├── state.py
│   ├── utils.py
│   ├── components.py
│   └── pages/
│       ├── __init__.py
│       ├── home.py
│       ├── classical_algorithm.py
│       ├── machine_learning.py
│       ├── ucf_dashboard.py
│       ├── model_comparison.py
│       ├── performance_metrics.py
│       ├── explainable_ai.py
│       ├── ucf_vs_bigo.py
│       ├── automl_selection.py
│       ├── advanced_visualizations.py
│       ├── ucf_formula_lab.py
│       └── about.py
│
└── ...
```

## Module Responsibilities

### `app.py`

Main Streamlit entry point. It should primarily initialize the
application, configure navigation, and call the selected page.

### `algorithms/`

Contains classical algorithms, ML model definitions, and dataset-loading
components.

### `experiments/`

Contains experiment execution and repeated-run logic.

### `profiler/`

Contains practical execution profiling such as time, CPU, and memory
measurements.

### `ucf/`

Contains UCF calculation and related complexity logic.

### `xai/`

Contains Explainable AI functionality such as SHAP/LIME integrations.

### `gui/`

Contains Streamlit presentation code. Common layout, state, utilities,
components, and individual pages are separated here.

## Environment Configuration

Create a `.env` file in the project root:

``` env
APP_ENV=development
DEBUG=false

OPENAI_API_KEY=
GROQ_API_KEY=
```

Only configure variables that are actually required by the project.

**Never commit real API keys to Git.**

The `.env.example` file should contain placeholder values and can be
committed safely.

## Installation

### 1. Create a virtual environment

Windows:

``` bash
python -m venv .venv
```

Activate it:

``` bash
.venv\Scriptsctivate
```

### 2. Install dependencies

``` bash
pip install -r requirements.txt
```

### 3. Run the application

Run from the **project root**:

``` bash
streamlit run streamlit_app.py
```

You can also use:

``` bash
python -m streamlit run app.py
```

Do not run the application from inside the `gui` directory.

## Experiment Workflow

The intended research workflow is:

``` text
Problem Definition
        ↓
Literature Review
        ↓
UCF Model Design
        ↓
System Development
        ↓
Dataset Integration
        ↓
Algorithm / Model Execution
        ↓
Performance Profiling
        ↓
UCF Computation
        ↓
Statistical Analysis
        ↓
Visualization
        ↓
Results & Evaluation
```

## Performance Metrics

### UCF metrics

  Metric   Meaning
  -------- -----------------------------
  `p(S)`   Stability-related measure
  `p(D)`   Determinism-related measure
  `f(u)`   UCF score
  `TC`     Total complexity
  `R`      Complexity remainder
  `f(t)`   Time component
  `g(M)`   Memory component
  Steps    Execution steps

### ML metrics

-   Accuracy
-   F1-score
-   Precision
-   Recall
-   MCC
-   Balanced Accuracy
-   Cohen's Kappa
-   ROC-AUC where supported

### System metrics

-   Training time
-   Inference time
-   Throughput
-   Average CPU
-   Peak CPU
-   RAM before/after
-   RAM delta
-   Peak heap
-   Model size
-   Space-complexity estimate

## Statistical Analysis

The UCF Dashboard can be used for empirical model comparison and may
include:

-   Welch's t-test
-   Mann-Whitney U test
-   One-way ANOVA
-   Kruskal-Wallis test
-   Levene's test
-   Correlation analysis
-   Model-wise averages
-   Heatmaps
-   Distribution box plots
-   Comparative bar charts

Statistical conclusions should be based on sufficient repeated
experiments and the assumptions of the selected statistical test.

## Explainable AI

The XAI component is designed to support model interpretation using
techniques such as:

-   SHAP
-   LIME

The purpose is to help investigate feature contributions and make model
behavior more understandable.

## Reproducibility

For thesis experiments, record:

``` text
Experiment ID
Date
Dataset
Dataset size
Algorithm / Model
Number of runs
Train/Test split
Input size
Execution time
CPU usage
Memory usage
UCF metrics
Accuracy
F1-score
Other evaluation metrics
Hardware
Software environment
```

Use the same preprocessing, dataset split, hardware conditions, and
experiment settings when comparing models.

## Troubleshooting

### `ModuleNotFoundError: No module named 'algorithms'`

Make sure you are running Streamlit from the project root and that:

``` text
algorithms/
    __init__.py
```

exists.

Run:

``` bash
streamlit run streamlit_app.py
```

Do not run the application from inside `gui/`.

### Dependency errors

Activate the virtual environment and run:

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Dataset errors

Check the file format, encoding, target column, missing values, and
available memory.

### MNIST download errors

The MNIST option may require internet access. If it fails, use Iris,
Wine, Breast Cancer, Digits, Synthetic Dataset, or another available
dataset.

### XAI errors

Verify that the required XAI packages are installed and that the
selected model is compatible with the explanation method.

## Security

Do not hard-code credentials:

``` python
api_key = "secret-value"
```

Use environment variables instead:

``` python
import os

api_key = os.getenv("OPENAI_API_KEY")
```

Keep `.env` excluded by `.gitignore`.

## Development Guidelines

Keep responsibilities separated:

``` text
Streamlit UI
    ↓
GUI page
    ↓
Experiment/service logic
    ↓
Algorithm or ML model
    ↓
Profiler / UCF
    ↓
Results
    ↓
Visualization
```

Avoid placing large computational functions directly inside Streamlit
page files.

When adding a new algorithm or model:

1.  Implement it in the appropriate algorithm/model module.
2.  Test it independently.
3.  Connect it to the experiment runner.
4.  Add it to the appropriate UI selection.
5.  Verify profiling and UCF results.
6.  Verify charts and saved results.

## Research Context

This system is intended to support master's-level research into
practical computational complexity and algorithm/model performance.

The platform is designed to help connect:

``` text
Theoretical Complexity
        +
Empirical Execution Behavior
        +
Machine Learning Performance
        +
Explainability
        ↓
Practical Algorithm Analysis
```

## Limitations

-   Execution time can vary with operating-system and background
    processes.
-   CPU and memory measurements are hardware/system dependent.
-   Dataset characteristics can strongly affect ML performance.
-   Different algorithms may require different preprocessing.
-   UCF is an empirical framework implemented by this project and should
    be interpreted within its defined methodology.
-   Big-O and empirical measurements describe different aspects of
    algorithm behavior.
-   XAI explanations depend on the selected method and model.

## Future Development

Possible future improvements include:

-   Additional classical algorithms
-   Additional ML/deep-learning models
-   GPU profiling
-   Larger automated benchmarking
-   Experiment-history database
-   Automated PDF/Excel reports
-   More XAI methods
-   Cloud/distributed experiments
-   Expanded AutoML functionality
-   Additional statistical validation

## Author

**Aqib Ali Buriro**
Master of Data Science
Mehran University of Engineering and Technology\
Jamshoro, Sindh, Pakistan

### Supervisors

-   Dr. Bushra Naz
-   Dr. Sammer Zai

## Quick Start

``` bash
cd AI-Based-Algorithmic-Analyzer
python -m venv .venv
.venv\Scriptsctivate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then select an analysis module from the Streamlit sidebar.
#   u c f _ a n a l y z e r _ a l g o r i t h m i c  
 #   u c f _ a n a l y z e r _ a l g o r i t h m i c  
 