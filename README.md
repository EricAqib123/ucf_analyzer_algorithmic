AI-Based Algorithmic Analyzer (UCF)
Overview

AI-Based Algorithmic Analyzer (UCF) is a Streamlit-based academic research and experimental platform for analyzing the practical performance and computational complexity of classical algorithms and machine-learning models.

The system uses the Unified Complexity Function (UCF) approach and combines theoretical complexity analysis with empirical execution profiling.

Main Measurements
Execution time
CPU utilization
Memory usage
Runtime behavior
Algorithm/model execution steps
Stability and determinism
UCF score
Total complexity
Remainder/residual complexity
Accuracy
Precision
Recall
F1-score
Matthews Correlation Coefficient (MCC)
Balanced Accuracy
Cohen's Kappa
ROC-AUC where applicable
Throughput
Model size
Practical space-complexity estimate
Explainable AI results
UCF Model

The project uses the following UCF formulation:

f(u) = f(t) + p(s) - p(d)


Where:

f(u) = Unified Complexity Function output
f(t) = execution-time component
p(s) = stability/sequence-related component
p(d) = determinism/complexity-difference component

UCF results should be interpreted according to the implementation and experimental methodology of this project. UCF complements rather than replaces formal asymptotic analysis such as Big-O.

Main Features
Classical Algorithm Analysis

The application provides practical profiling of classical algorithms, including:

Algorithm selection
Input-size selection
Execution-time measurement
CPU monitoring
RAM monitoring
Peak heap measurement
Step counting
UCF calculation
Big-O reference
Complexity visualization
Experiment-result storage

The current implementation includes Bubble Sort.

Machine Learning Analysis

Supported models include:

Decision Tree
Random Forest
Logistic Regression
K-Nearest Neighbors
Support Vector Machine
Gradient Boosting
XGBoost
LightGBM
AdaBoost
Extra Trees
Linear Discriminant Analysis
Quadratic Discriminant Analysis
MLP Neural Network
Dataset Support

The application supports:

Synthetic datasets
CIC-IDS CSV data
Custom CSV datasets
Scikit-learn Iris
Scikit-learn Wine
Scikit-learn Breast Cancer
Scikit-learn Digits
MNIST subset

For custom CSV files, the application can configure:

Target column
Columns to drop
Test split
Maximum training samples
Application Pages

The organized application contains the following pages:

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

Project Structure
AI-Based-Algorithmic-Analyzer/
│
├── app.py
├── main.py
├── README.md
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
├── LICENSE
│
├── algorithms/
│   ├── __init__.py
│   └── ...
│
├── experiments/
│   ├── __init__.py
│   └── ...
│
├── profiler/
│   ├── __init__.py
│   └── ...
│
├── ucf/
│   ├── __init__.py
│   └── ...
│
├── xai/
│   ├── __init__.py
│   └── ...
│
├── dataset/
│   └── ...
│
├── img/
│   ├── logo.png
│   └── logo1.png
│
├── gui/
│   ├── __init__.py
│   ├── layout.py
│   ├── state.py
│   ├── utils.py
│   ├── components.py
│   │
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

Module Responsibilities
app.py

Main Streamlit entry point. It initializes the application, configures navigation, and calls the selected page.

algorithms/

Contains classical algorithms, machine-learning model definitions, and algorithm-related components.

experiments/

Contains experiment execution, repeated-run logic, and experimental workflow components.

profiler/

Contains practical execution profiling such as execution time, CPU utilization, memory usage, and related measurements.

ucf/

Contains UCF calculation and related computational-complexity logic.

xai/

Contains Explainable AI functionality such as SHAP and LIME integrations.

dataset/

Contains dataset loading, generation, preprocessing, and dataset-related utilities.

gui/

Contains Streamlit presentation code. Common layout, state management, utilities, components, and individual application pages are separated here.

Environment Configuration

Create a .env file in the project root if environment variables are required:

APP_ENV=development
DEBUG=false

OPENAI_API_KEY=
GROQ_API_KEY=


Only configure variables that are actually required by the project.

Never commit real API keys or credentials to GitHub.

The .env.example file should contain placeholder values and can be committed safely.

The .env file should be excluded through .gitignore:

.env

Installation
1. Clone the Repository
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd AI-Based-Algorithmic-Analyzer

2. Create a Virtual Environment
Windows
python -m venv .venv


Activate it:

.venv\Scripts\activate

Linux/macOS
python3 -m venv .venv


Activate it:

source .venv/bin/activate

3. Install Dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

4. Run the Application

Run the application from the project root:

python -m streamlit run app.py


You can also use:

streamlit run app.py


Do not run the application from inside the gui directory.

Experiment Workflow

The intended research workflow is:

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

Performance Metrics
UCF Metrics
Metric	Meaning
p(S)	Stability-related measure
p(D)	Determinism-related measure
f(u)	UCF score
TC	Total complexity
R	Complexity remainder
f(t)	Time component
g(M)	Memory component
Steps	Execution-step measurement
ML Metrics
Accuracy
F1-score
Precision
Recall
Matthews Correlation Coefficient (MCC)
Balanced Accuracy
Cohen's Kappa
ROC-AUC where supported
System Metrics
Training time
Inference time
Execution time
Throughput
Average CPU utilization
Peak CPU utilization
RAM before execution
RAM after execution
RAM delta
Peak heap usage
Model size
Practical space-complexity estimate
Statistical Analysis

The UCF Dashboard can be used for empirical model comparison and may include:

Welch's t-test
Mann-Whitney U test
One-way ANOVA
Kruskal-Wallis test
Levene's test
Correlation analysis
Model-wise averages
Heatmaps
Distribution box plots
Comparative bar charts

Statistical conclusions should be based on sufficient repeated experiments and the assumptions of the selected statistical test.

Explainable AI

The XAI component is designed to support model interpretation using techniques such as:

SHAP
LIME

The purpose is to investigate feature contributions and make machine-learning model behavior more understandable.

Reproducibility

For thesis experiments, record:

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
Precision
Recall
F1-score
Other evaluation metrics
Hardware
Operating system
Python version
Software environment
Experiment configuration


Use the same preprocessing, dataset split, hardware conditions, software environment, and experiment settings when comparing models.

Troubleshooting
ModuleNotFoundError: No module named 'algorithms'

Make sure you are running Streamlit from the project root and that:

algorithms/
    __init__.py


exists.

Run:

python -m streamlit run app.py


Do not run the application from inside gui/.

Dependency Errors

Activate the virtual environment and run:

python -m pip install --upgrade pip
pip install -r requirements.txt

Dataset Errors

Check the following:

File format
File encoding
Target column
Missing values
Feature types
Available memory
Dataset size

For custom CSV files, verify that the selected target column exists.

MNIST Download Errors

The MNIST option may require internet access.

If it fails, use:

Iris
Wine
Breast Cancer
Digits
Synthetic Dataset
Another available local dataset
XAI Errors

Verify that:

Required XAI packages are installed.
The selected model is compatible with the explanation method.
The dataset contains valid feature values.
The selected XAI method supports the model type.
Security

Do not hard-code credentials:

api_key = "secret-value"


Use environment variables instead:

import os

api_key = os.getenv("OPENAI_API_KEY")


Keep .env excluded from Git:

.env


Never upload API keys, passwords, tokens, or other private credentials to GitHub.

Development Guidelines

Keep responsibilities separated:

Streamlit UI
    ↓
GUI Page
    ↓
Experiment / Service Logic
    ↓
Algorithm or ML Model
    ↓
Profiler / UCF
    ↓
Results
    ↓
Visualization


Avoid placing large computational functions directly inside Streamlit page files.

When adding a new algorithm or model:

Implement it in the appropriate algorithm/model module.
Test it independently.
Connect it to the experiment runner.
Add it to the appropriate UI selection.
Verify profiling and UCF results.
Verify statistical calculations where applicable.
Verify charts and visualizations.
Verify saved experiment results.
Update the documentation.
Research Context

This system is intended to support master's-level research into practical computational complexity and algorithm/model performance.

The platform is designed to help connect:

Theoretical Complexity
        +
Empirical Execution Behavior
        +
Machine Learning Performance
        +
Explainability
        ↓
Practical Algorithm Analysis


The system provides an experimental environment for studying the relationship between theoretical complexity and observed computational behavior.

Limitations
Execution time can vary with operating-system activity and background processes.
CPU measurements are hardware and system dependent.
Memory measurements are hardware and system dependent.
Dataset characteristics can strongly affect machine-learning performance.
Different algorithms and models may require different preprocessing.
Different hardware configurations may produce different empirical results.
UCF is an empirical framework implemented by this project and should be interpreted within its defined methodology.
UCF does not replace formal asymptotic complexity analysis.
Big-O and empirical measurements describe different aspects of algorithm behavior.
XAI explanations depend on the selected model, dataset, and explanation method.
Statistical conclusions depend on the number and quality of repeated experiments.
Future Development

Possible future improvements include:

Additional classical algorithms
Additional machine-learning models
Deep-learning models
GPU profiling
Larger automated benchmarking
Experiment-history database
Automated PDF reports
Automated Excel/CSV reports
More XAI methods
Cloud/distributed experiments
Expanded AutoML functionality
Additional statistical validation
Reproducible experiment configuration
Automated experiment pipelines
Author

Aqib Ali Buriro

Master of Data Science
Mehran University of Engineering and Technology
Jamshoro, Sindh, Pakistan

Supervisors
Dr. Bushra Naz
Dr. Sammer Zai
License

This project is licensed under the MIT License.

See the LICENSE file for the complete license text.

Citation

If this project contributes to academic research, please cite the associated thesis, research paper, or repository.

Aqib Ali Buriro.
AI-Based Algorithmic Analyzer (UCF).
Mehran University of Engineering and Technology.
Jamshoro, Sindh, Pakistan.


A formal BibTeX citation can be added when the associated research publication or thesis citation details are finalized.

Acknowledgments

This project was developed as part of master's-level research in Data Science at Mehran University of Engineering and Technology, Jamshoro.

Special acknowledgment is given to the project supervisors:

Dr. Bushra Naz
Dr. Sammer Zai
Quick Start
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd AI-Based-Algorithmic-Analyzer

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt

python -m streamlit run app.py


Then open the Streamlit application and select an analysis module from the sidebar.
