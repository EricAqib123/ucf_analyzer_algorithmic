<p align="center"> <img src="extracted/ucf_analyzer_algorithmic/img/main.png" alt="AI-Based Algorithmic Analyzer — Unified Complexity Function" width="300"> </p> <p align="center"> <strong>Analyze algorithms and machine-learning models through practical performance, complexity, and explainability.</strong><br> Combine empirical execution profiling with the Unified Complexity Function (UCF), classical complexity analysis, ML evaluation, and Explainable AI. </p> <p align="center"> <img src="https://img.shields.io/badge/Streamlit-UI-C4522A" alt="Streamlit"> <img src="https://img.shields.io/badge/Python-3.x-3776AB" alt="Python"> <img src="https://img.shields.io/badge/UCF-Experimental-8A6A52" alt="UCF"> <img src="https://img.shields.io/badge/ML-13%20Models-1A8763" alt="Machine learning models"> <img src="https://img.shields.io/badge/XAI-SHAP%20%7C%20LIME-B4790E" alt="Explainable AI"> </p> <p align="center"> <a href="#quick-start">Quick start</a> · <a href="#how-it-works">How it works</a> · <a href="#ucf-model">UCF model</a> · <a href="#machine-learning-analysis">ML analysis</a> · <a href="#project-map">Project map</a> · <a href="#research-context">Research context</a> </p>

Introduction
AI-Based Algorithmic Analyzer (UCF) is a Streamlit-based academic research and experimental platform for studying the practical performance and computational behavior of classical algorithms and machine-learning models.

The system combines theoretical complexity analysis with empirical execution profiling. It measures execution time, CPU utilization, memory consumption, execution steps, model performance, and other practical characteristics, then applies the project's Unified Complexity Function (UCF) framework to organize these observations into an experimental complexity analysis.

The platform also provides statistical analysis, comparative visualizations, AutoML-oriented model selection, and Explainable AI capabilities using techniques such as SHAP and LIME.

Research note: UCF is an empirical framework implemented by this project. It complements formal complexity analysis such as Big-O rather than replacing it.

Created by Aqib Ali Buriro
Master of Data Science
Mehran University of Engineering and Technology, Jamshoro, Sindh, Pakistan

Supervisors
Dr. Bushra Naz
Dr. Sammer Zai
What you can analyze
The platform is designed to answer questions such as:

Analyze	Measure
Classical algorithms	Runtime, CPU, memory, steps, UCF, Big-O reference
Machine-learning models	Training/inference performance and predictive metrics
Datasets	Dataset size, preprocessing, train/test configuration, throughput
Computational behavior	Stability, determinism, complexity, and remainder
Model quality	Accuracy, precision, recall, F1, MCC, ROC-AUC and more
Explainability	Feature contributions using SHAP/LIME
Statistical behavior	Significance tests, distributions, correlations, and comparisons
Model selection	Comparative performance and AutoML-oriented selection
Experiments	Repeated runs and reproducible experiment records

Main measurements
The analyzer can collect and display:

Execution time
CPU utilization
Peak CPU
Memory usage
RAM before and after execution
RAM delta
Peak heap usage
Runtime behavior
Execution steps
Stability
Determinism
UCF score
Total complexity
Complexity remainder
Time component
Memory component
Accuracy
Precision
Recall
F1-score
MCC
Balanced Accuracy
Cohen's Kappa
ROC-AUC where applicable
Training time
Inference time
Throughput
Model size
Practical space-complexity estimates
Explainable AI results
How it works
The intended analysis pipeline connects theoretical concepts with empirical experimentation:

    A[Problem / Dataset] --> B[Algorithm or ML Model]
    B --> C[Experiment Runner]
    C --> D[Performance Profiler]
    D --> E[Time / CPU / Memory / Steps]
    E --> F[UCF Computation]
    B --> G[ML Evaluation]
    F --> H[Statistical Analysis]
    G --> H
    H --> I[Visualization]
    I --> J[Research Results]
    classDef input fill:#F3D9C2,stroke:#C4522A,color:#241A12;
    classDef process fill:#F3E4D4,stroke:#B4790E,color:#241A12;
    class A,B input;
    class C,D,E,F,G,H,I,J process;

Research workflow
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

Define the experiment. Select an algorithm, ML model, dataset, or input size.
Configure the experiment. Choose preprocessing, train/test split, number of runs, and other parameters.
Execute the workload. Run the selected algorithm or model.
Profile execution. Capture time, CPU, RAM, heap, throughput, and other practical measurements.
Calculate UCF metrics. Apply the project's UCF formulation to the experimental results.
Evaluate ML performance. Calculate predictive metrics where applicable.
Analyze statistically. Compare repeated experiments and model behavior.
Visualize the results. Use dashboards, charts, distributions, and comparison views.
Interpret the findings. Compare empirical behavior with theoretical complexity and model performance.
UCF Model
The project uses the following Unified Complexity Function formulation:

f(u) = f(t) + p(s) - p(d)

Where:

Symbol	Meaning
f(u)	Unified Complexity Function output
f(t)	Execution-time component
p(s)	Stability/sequence-related component
p(d)	Determinism/complexity-difference component

The exact interpretation of these components depends on the implementation and experimental methodology used by the project.

UCF and Big-O
UCF and Big-O describe different aspects of algorithm behavior.

flowchart TD
    A[Algorithm / Model] --> B[Formal Analysis]
    A --> C[Empirical Execution]
    B --> D[Big-O / Theoretical Complexity]
    C --> E[Time / CPU / Memory / Steps]
    E --> F[UCF Analysis]
    D --> G[Compare Perspectives]
    F --> G
    G --> H[Practical Interpretation]

Big-O focuses on asymptotic growth, while UCF is intended to capture experimentally observed behavior using the project's defined measurements.

Therefore, a UCF result should not be interpreted as a replacement for a formal proof of computational complexity.

Classical Algorithm Analysis
The classical-analysis module provides practical profiling of algorithms.

Current functionality includes:

Algorithm selection
Input-size selection
Execution-time measurement
CPU monitoring
RAM monitoring
Peak heap measurement
Execution-step counting
UCF calculation
Big-O reference
Complexity visualization
Experiment-result storage
The current implementation includes:

Bubble Sort
Bubble Sort is included as the initial classical algorithm for demonstrating the complete profiling and UCF workflow.

A typical experiment can compare:

Input Size
    ↓
Bubble Sort
    ↓
Execution Time
    ↓
CPU / Memory
    ↓
Step Count
    ↓
UCF
    ↓
Visualization

Additional classical algorithms can be added through the algorithms/ module.

Machine Learning Analysis
The machine-learning component supports a range of classical ML models and neural-network-based classification.

Supported models
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
The models can be compared using both predictive performance and practical computational behavior.

For example:

Dataset
   ↓
Preprocessing
   ↓
Train / Test Split
   ↓
Model Training
   ↓
Performance Profiling
   ↓
Prediction
   ↓
Evaluation Metrics
   ↓
UCF / Complexity Analysis
   ↓
Model Comparison

Dataset Support
The application supports multiple dataset sources.

Included / supported datasets
Synthetic datasets
CIC-IDS CSV data
Custom CSV datasets
Scikit-learn Iris
Scikit-learn Wine
Scikit-learn Breast Cancer
Scikit-learn Digits
MNIST subset
Custom CSV datasets
For custom datasets, the application can configure:

Target column
Columns to drop
Test split
Maximum training samples
Dataset preprocessing
When using large datasets, experiment configuration should consider available system memory and execution time.

Application Pages
The Streamlit application is organized into dedicated analysis pages:

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

Dashboard overview
The dashboard provides a central location for inspecting experimental results and comparing algorithms or models.

Typical analysis flow:

Select Experiment
      ↓
Inspect Performance
      ↓
Inspect UCF
      ↓
Compare Models
      ↓
Review Statistical Results
      ↓
Inspect Explainability
      ↓
Interpret Results

Performance Metrics
UCF metrics
Metric	Meaning
p(S)	Stability-related measure
p(D)	Determinism-related measure
f(u)	UCF score
TC	Total complexity
R	Complexity remainder
f(t)	Time component
g(M)	Memory component
Steps	Execution-step measurement

Machine-learning metrics
The ML analysis module can calculate:

Accuracy
Precision
Recall
F1-score
MCC
Balanced Accuracy
Cohen's Kappa
ROC-AUC where supported
System metrics
Practical computational behavior can be analyzed using:

Training time
Inference time
Throughput
Average CPU
Peak CPU
RAM before execution
RAM after execution
RAM delta
Peak heap
Model size
Practical space-complexity estimate
Statistical Analysis
The UCF Dashboard can be used for empirical comparison across algorithms, models, datasets, and repeated experiments.

Supported or planned statistical analysis includes:

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
Statistical tests should be selected according to the experimental design and assumptions of the individual test.

Research caution: Statistical significance should not be confused with practical significance. Conclusions should be based on sufficient repeated experiments and appropriate assumptions.

Explainable AI
The xai/ component provides model-interpretation functionality.

The project is designed to support techniques such as:

SHAP
LIME
The goal is to investigate feature contributions and make machine-learning behavior easier to understand.

flowchart LR
    A[Trained Model] --> B[XAI Engine]
    B --> C[Feature Contributions]
    C --> D[Global Explanation]
    C --> E[Local Explanation]
    D --> F[Interpretation]
    E --> F

XAI explanations depend on the selected model, dataset, preprocessing pipeline, and explanation method.

Project Map
The repository is organized into separate layers for the user interface, experiments, algorithms, profiling, UCF calculations, machine learning, datasets, and explainability.

extracted/ucf_analyzer_algorithmic/
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
Module	Responsibility
app.py	Main Streamlit entry point and application navigation
algorithms/	Classical algorithms, model definitions, and related analysis components
experiments/	Experiment execution and repeated-run logic
profiler/	Time, CPU, memory, heap, and execution profiling
ucf/	UCF calculation and complexity-related logic
xai/	SHAP/LIME and other explainability functionality
ml_models/	Machine-learning model implementations
dataset/	Dataset loading and preparation
gui/	Streamlit presentation layer
gui/pages/	Individual application pages
img/	Logos, screenshots, diagrams, and other visual assets

app.py
The primary Streamlit entry point should initialize the application, configure navigation, and load the selected page.

algorithms/
Contains classical algorithm implementations and related algorithm/model components.

experiments/
Contains experiment execution, repeated-run handling, and experimental workflow logic.

profiler/
Contains practical measurements such as execution time, CPU utilization, RAM usage, and peak memory.

ucf/
Contains the Unified Complexity Function implementation and related complexity calculations.

xai/
Contains Explainable AI functionality, including SHAP/LIME integrations where configured.

gui/
Contains the Streamlit presentation layer, including common layout, state management, reusable components, utilities, and individual pages.

Environment Configuration
Create a .env file in the project root when environment variables are required:

APP_ENV=development
DEBUG=false

OPENAI_API_KEY=
GROQ_API_KEY=

Only configure variables that are actually required by your installation.

Never commit real API keys or credentials to Git.

The .env.example file should contain placeholders only and can safely be committed.

Quick Start
Run the application from the project root.

Windows PowerShell
Create the virtual environment:

python -m venv .venv

Activate it:

.\.venv\Scripts\Activate.ps1

Install dependencies:

python -m pip install --upgrade pip
pip install -r requirements.txt

Start Streamlit:

streamlit run streamlit_app.py

If the project uses app.py as the actual entry point, use:

python -m streamlit run app.py

macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py

Open the Streamlit address shown in the terminal, typically:

http://localhost:8501

Important: Run Streamlit from the project root. Do not start the application from inside the gui/ directory.

Reproducible Research
For thesis experiments, record the configuration and environment used for every experiment.

A recommended experiment record contains:

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
Peak heap
UCF metrics
Accuracy
Precision
Recall
F1-score
Other evaluation metrics
Hardware
Operating system
Python version
Library versions
Experiment settings

For meaningful comparisons, keep the following consistent where appropriate:

Dataset version
Preprocessing
Train/test split
Number of repetitions
Hardware
Software environment
Experiment configuration
Experiment Design
Repeated experiments are especially important for runtime and system-level measurements because execution can be affected by:

Operating-system scheduling
Background processes
CPU frequency changes
Available memory
Dataset size
Cache behavior
Hardware differences
A recommended structure is:

Experiment
    ├── Configuration
    ├── Warm-up / preparation
    ├── Repeated executions
    ├── Raw measurements
    ├── Aggregated statistics
    ├── UCF computation
    ├── Statistical tests
    └── Visualization

Verify the Installation
After installing the dependencies, verify that the application can start successfully:

python -m streamlit run app.py

If your repository uses streamlit_app.py as the primary entry point:

streamlit run streamlit_app.py

You can also verify that Python can import the main project modules:

python -c "import algorithms, experiments, profiler, ucf"

Troubleshooting
ModuleNotFoundError: No module named 'algorithms'
Make sure Streamlit is being launched from the project root.

Verify that the package exists:

algorithms/
    __init__.py

Then run:

streamlit run app.py

Do not launch the application from inside gui/.

Dependency errors
Activate the virtual environment and update the package installer:

python -m pip install --upgrade pip
pip install -r requirements.txt

If a particular package fails, verify that your Python version is compatible with the pinned dependencies.

Dataset errors
Check:

File format
File encoding
Target-column name
Missing values
Feature types
Available memory
Dataset size
For custom CSV files, confirm that the configured target column actually exists.

MNIST download errors
The MNIST option may require internet access depending on the dataset-loading implementation.

If MNIST cannot be downloaded, use another available dataset such as:

Iris
Wine
Breast Cancer
Digits
Synthetic Dataset
CIC-IDS
Custom CSV
XAI errors
Verify that the required XAI dependencies are installed and that the selected model is compatible with the chosen explanation method.

Different SHAP and LIME approaches may require different model interfaces or data formats.

Incorrect runtime or memory measurements
System-level measurements are sensitive to the execution environment.

For more reliable experiments:

Close unnecessary applications.
Use repeated runs.
Keep hardware conditions consistent.
Record the machine configuration.
Avoid comparing measurements collected under substantially different system loads.
Security
Never hard-code credentials:

api_key = "secret-value"

Use environment variables instead:

import os

api_key = os.getenv("OPENAI_API_KEY")

Keep .env excluded from Git:

.env

Use .env.example for safe placeholder configuration.

Never commit:

API keys
Passwords
Private datasets
Sensitive experiment data
Personal information
Private credentials
Development Guidelines
Keep computational responsibilities separate from Streamlit presentation code.

The preferred architecture is:

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

Adding a new algorithm or model
When adding a new algorithm or machine-learning model:

Implement it in the appropriate algorithm/model module.
Test it independently.
Connect it to the experiment runner.
Add it to the appropriate UI selection.
Verify profiling and UCF calculations.
Add the relevant performance metrics.
Verify charts and visualizations.
Confirm that experiment results are stored correctly.
Test the complete workflow from the Streamlit interface.
Research Context
This system is intended to support master's-level research into practical computational complexity and algorithm/model performance.

The platform connects four complementary perspectives:

Theoretical Complexity
        +
Empirical Execution Behavior
        +
Machine Learning Performance
        +
Explainability
        ↓
Practical Algorithm Analysis

The objective is not simply to determine which algorithm is "best," but to investigate how different computational approaches behave under measurable experimental conditions.

Limitations
Execution time can vary because of operating-system scheduling and background processes.
CPU and memory measurements are hardware and system dependent.
Dataset characteristics can strongly affect machine-learning performance.
Different algorithms and models may require different preprocessing.
UCF is an empirical framework implemented by this project and should be interpreted within its defined methodology.
Big-O and empirical measurements describe different aspects of computational behavior.
XAI explanations depend on the selected model, data, preprocessing, and explanation technique.
A single experiment is generally insufficient to establish a reliable performance conclusion.
Results from one hardware environment may not generalize to another.
Large datasets can substantially increase memory and execution requirements.
Interpret results as experimental evidence, not universal guarantees.

Future Development
Possible future improvements include:

Additional classical algorithms
Additional machine-learning models
Deep-learning models
GPU profiling
Larger automated benchmarking
Experiment-history database
Automated PDF reports
Automated Excel reports
More Explainable AI methods
Cloud-based experiments
Distributed experiments
Expanded AutoML functionality
Additional statistical validation
Hardware-aware benchmarking
Experiment reproducibility packages
Versioned datasets and experiment configurations
Quick Reference
1. Clone / open the project
2. Open a terminal in the project root
3. Create a virtual environment
4. Activate the environment
5. Install requirements
6. Start Streamlit
7. Select an analysis module
8. Configure the experiment
9. Run the analysis
10. Inspect UCF and performance results
11. Compare algorithms/models
12. Review statistical and XAI results

Windows
cd AI-Based-Algorithmic-Analyzer

python -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

pip install -r requirements.txt

streamlit run app.py

If the repository's configured entry point is streamlit_app.py:

streamlit run streamlit_app.py

Then select an analysis module from the Streamlit sidebar.

Author
Aqib Ali Buriro
Master of Data Science
Mehran University of Engineering and Technology
Jamshoro, Sindh, Pakistan

Supervisors
Dr. Bushra Naz
Dr. Sammer Zai
