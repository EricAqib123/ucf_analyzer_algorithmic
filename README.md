AI-Based Algorithmic Analyzer (UCF)

An experimental Streamlit platform for evaluating classical algorithms and machine-learning models using empirical performance profiling, machine-learning evaluation metrics, explainable AI, statistical analysis, and the project's Unified Complexity Function (UCF) framework.

Research project: Master's research in Data Science
Institution: Mehran University of Engineering and Technology, Jamshoro, Sindh, Pakistan

Overview

The AI-Based Algorithmic Analyzer (UCF) is a research-oriented software platform designed to investigate the relationship between theoretical computational complexity and observed execution behavior.

The system combines:

Theoretical complexity references such as Big-O
Empirical execution profiling
CPU and memory measurements
Execution-step analysis
Stability and determinism measurements
UCF-based complexity scoring
Machine-learning performance evaluation
Statistical analysis
Explainable AI
Interactive Streamlit visualizations

The objective is not to replace formal complexity analysis, but to provide an experimental framework for studying how algorithms and machine-learning models behave under practical execution conditions.

Research Concept

The project uses the following UCF formulation:

f(u) = f(t) + p(s) - p(d)


where:

Symbol	Description
f(u)	Unified Complexity Function output
f(t)	Execution-time component
p(s)	Stability/sequence-related component
p(d)	Determinism/complexity-difference component

The implementation also includes experimental UCF variants for investigating relationships between probability-related factors, total complexity, and residual complexity.

UCF is an empirical framework implemented by this project. Its results should therefore be interpreted according to the project's experimental methodology and should not be treated as a replacement for formal asymptotic analysis.

Key Features
Classical Algorithm Analysis

The classical algorithm module supports practical analysis of algorithms using measurements such as:

Execution time
CPU utilization
Memory usage
Peak heap usage
Execution steps
Input-size variation
UCF score
Total complexity
Remainder/residual complexity
Big-O reference complexity
Performance visualization
Repeated experiments

The current implementation includes:

Bubble Sort

Additional algorithms can be integrated through the algorithm and experiment layers.

Machine Learning Analysis

The platform currently supports:

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

The application supports several dataset sources, including:

Synthetic datasets
CIC-IDS CSV datasets
Custom CSV datasets
Iris
Wine
Breast Cancer
Digits
MNIST subset

For custom CSV datasets, the application can support configuration of:

Target column
Columns to remove
Train/test split
Maximum training samples
Dataset preprocessing
Application Modules

The Streamlit application contains the following modules:

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

UCF Dashboard

Provides experimental summaries and comparative analysis of UCF-related measurements.

Model Comparison

Allows different machine-learning models to be compared using performance and computational metrics.

Performance Metrics

Provides evaluation metrics such as:

Accuracy
Precision
Recall
F1-score
Matthews Correlation Coefficient
Balanced Accuracy
Cohen's Kappa
ROC-AUC where applicable
Explainable AI

Provides model-interpretability functionality using techniques such as:

SHAP
LIME
UCF vs Big-O

Provides a comparison between empirical UCF measurements and traditional asymptotic complexity references.

AutoML Selection

Provides experimental model-selection functionality for comparing candidate machine-learning models.

Advanced Visualizations

Provides interactive charts and comparative visualizations for experimental results.

UCF Formula Lab

Provides an interactive environment for experimenting with UCF parameters, sensitivity analysis, Monte Carlo simulations, and UCF formula variants.

System Metrics

The platform can collect or calculate metrics including:

Category	Metrics
Runtime	Training time, inference time, execution time
CPU	Average CPU, peak CPU
Memory	RAM before/after, RAM delta, peak heap
Complexity	UCF, total complexity, remainder
Execution	Steps, throughput
Model	Model size, practical space-complexity estimate
Classification	Accuracy, precision, recall, F1
Statistical	MCC, Balanced Accuracy, Cohen's Kappa
ROC	ROC-AUC where supported
Statistical Analysis

The platform is designed to support empirical statistical analysis of experimental results.

Depending on the experiment, available analyses may include:

Welch's t-test
Mann-Whitney U test
One-way ANOVA
Kruskal-Wallis test
Levene's test
Correlation analysis
Model-wise aggregation
Distribution analysis
Box plots
Heatmaps
Comparative bar charts

Statistical conclusions should be based on an appropriate number of repeated experiments and the assumptions of the selected statistical method.

Experimental Workflow

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
Results and Evaluation

Reproducibility

For research experiments, record the following information:

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
UCF measurements
Accuracy
F1-score
Other evaluation metrics
Hardware
Operating system
Python version
Package versions
Experiment configuration


For meaningful comparisons, keep the following conditions consistent where appropriate:

Dataset
Preprocessing
Train/test split
Random seeds
Hardware
Software environment
Number of repetitions
Experimental configuration
Project Architecture

The project follows a layered architecture:

Streamlit Application
        │
        ▼
GUI / Presentation Layer
        │
        ▼
Experiment / Service Layer
        │
        ├───────────────┐
        ▼               ▼
Algorithms / ML       Profiler
        │               │
        └───────┬───────┘
                ▼
             UCF
                │
                ▼
             Results
                │
                ▼
        Visualization / XAI


The goal is to keep the user interface separate from computational and experimental logic.

Project Structure
AI-Based-Algorithmic-Analyzer/
│
├── app.py
├── main.py
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── algorithms/
│   ├── __init__.py
│   ├── ...
│
├── experiments/
│   ├── __init__.py
│   ├── ...
│
├── profiler/
│   ├── __init__.py
│   ├── ...
│
├── ucf/
│   ├── __init__.py
│   ├── ...
│
├── xai/
│   ├── __init__.py
│   ├── ...
│
├── dataset/
│   ├── ...
│
├── img/
│   ├── logo.png
│   └── logo1.png
│
└── gui/
    ├── __init__.py
    ├── layout.py
    ├── state.py
    ├── utils.py
    ├── components.py
    │
    └── pages/
        ├── __init__.py
        ├── home.py
        ├── classical_algorithm.py
        ├── machine_learning.py
        ├── ucf_dashboard.py
        ├── model_comparison.py
        ├── performance_metrics.py
        ├── explainable_ai.py
        ├── ucf_vs_bigo.py
        ├── automl_selection.py
        ├── advanced_visualizations.py
        ├── ucf_formula_lab.py
        └── about.py


The exact structure may evolve as the research implementation develops.

Installation
Prerequisites

Recommended:

Python 3.10+
Git
pip
Virtual environment

Clone the repository:

git clone <YOUR-REPOSITORY-URL>
cd AI-Based-Algorithmic-Analyzer


Create a virtual environment:

Windows
python -m venv .venv
.venv\Scripts\activate

Linux / macOS
python3 -m venv .venv
source .venv/bin/activate


Install dependencies:

python -m pip install --upgrade pip
pip install -r requirements.txt

Running the Application

Run the application from the project root.

If app.py is the Streamlit entry point:

python -m streamlit run app.py


Alternatively:

streamlit run app.py


The application will provide a local Streamlit URL, normally:

http://localhost:8501


Use the actual entry-point filename present in your repository. Keep the README consistent with the deployed application.

Environment Variables

Create a local .env file if the project requires environment variables:

APP_ENV=development
DEBUG=false

OPENAI_API_KEY=
GROQ_API_KEY=


Only configure variables required by your implementation.

Security

Never commit API keys or other credentials.

Do not do this:

api_key = "your-secret-key"


Instead:

import os

api_key = os.getenv("OPENAI_API_KEY")


Add .env to .gitignore:

.env


The repository may safely contain:

.env.example


with placeholder values.

Troubleshooting
ModuleNotFoundError

If you see:

ModuleNotFoundError: No module named 'algorithms'


make sure you are running the application from the project root.

Verify that the required package directories contain __init__.py where appropriate.

Run:

python -m streamlit run app.py


rather than launching the application from inside the gui directory.

Dependency Errors

Activate the virtual environment and run:

python -m pip install --upgrade pip
pip install -r requirements.txt


If necessary, recreate the environment:

rm -rf .venv
python -m venv .venv


On Windows, remove .venv manually or use:

Remove-Item -Recurse -Force .venv

Dataset Errors

Check:

File format
Encoding
Target column
Missing values
Feature types
Dataset size
Available memory

For custom datasets, verify that the selected target column exists.

MNIST Download Problems

The MNIST option may require internet access.

If MNIST cannot be downloaded, use another available dataset such as:

Iris
Wine
Breast Cancer
Digits
Synthetic Dataset
Explainable AI Errors

If SHAP or LIME functionality fails:

Verify that the required packages are installed.
Verify that the selected model is supported.
Check the dataset's feature representation.
Check compatibility between the XAI method and model type.
Development Guidelines

Keep computational responsibilities outside Streamlit page modules whenever practical.

Recommended flow:

GUI
 ↓
Page
 ↓
Experiment / Service
 ↓
Algorithm / ML Model
 ↓
Profiler / UCF
 ↓
Results
 ↓
Visualization


When adding a new algorithm or machine-learning model:

Implement the algorithm/model in the appropriate module.
Test it independently.
Integrate it with the experiment runner.
Add it to the appropriate UI selection.
Verify profiling measurements.
Verify UCF calculations.
Verify stored results.
Verify visualizations.
Add appropriate documentation.
Limitations

The results generated by this platform depend on the experimental environment.

Important limitations include:

Execution time varies with operating-system activity and background processes.
CPU measurements depend on hardware and system load.
Memory measurements are system dependent.
Dataset characteristics can substantially affect ML performance.
Algorithms may require different preprocessing strategies.
Different hardware configurations can produce different empirical results.
UCF is an empirical framework implemented by this project.
UCF should not be interpreted as a replacement for formal asymptotic complexity.
Big-O and empirical measurements describe different aspects of computational behavior.
XAI results depend on the selected model, dataset, and explanation technique.
Research and Academic Use

This project is intended to support master's-level research into:

Practical computational complexity
Algorithm benchmarking
Machine-learning model performance
Empirical execution analysis
Statistical comparison
Explainable AI
Complexity-aware model evaluation

The software should be considered an experimental research platform rather than a production benchmarking standard.

When reporting experimental findings, document the hardware, software environment, datasets, preprocessing, number of repetitions, and configuration used.

Future Development

Potential future improvements include:

Additional classical algorithms
Additional machine-learning models
Deep-learning models
GPU profiling
Automated large-scale benchmarking
Experiment-history database
Automated PDF reports
Excel/CSV experiment reports
Additional XAI methods
Distributed experiments
Cloud-based benchmarking
Expanded AutoML capabilities
Additional statistical validation
Reproducible experiment configuration files
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

If this project contributes to academic research, please cite the associated thesis, paper, or repository.

Aqib Ali Buriro.
AI-Based Algorithmic Analyzer (UCF).
Mehran University of Engineering and Technology.


A formal BibTeX citation can be added when the associated research paper or thesis citation details are finalized.

Acknowledgments

This project was developed as part of master's-level research in Data Science at Mehran University of Engineering and Technology, Jamshoro.

Quick Start
git clone <YOUR-REPOSITORY-URL>
cd AI-Based-Algorithmic-Analyzer

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt

python -m streamlit run app.py


Then open the Streamlit application and select an analysis module from the sidebar.
