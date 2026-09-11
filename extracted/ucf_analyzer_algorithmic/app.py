import warnings
import streamlit as st

from gui.layout import (
configure_page,
render_header,
render_sidebar,
apply_global_style,
)

from gui.state import init_session_state

from gui.pages import home
from gui.pages import classical_algorithm
from gui.pages import machine_learning
from gui.pages import ucf_dashboard
from gui.pages import model_comparison
from gui.pages import performance_metrics
from gui.pages import ucf_vs_bigo
from gui.pages import explainable_ai
from gui.pages import about
from gui.pages import automl_selection
from gui.pages import advanced_visualizations
from gui.pages import ucf_formula_lab

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROUTES = {
"Home": home.render,
"Classical Algorithm": classical_algorithm.render,
"Machine Learning": machine_learning.render,
"UCF Dashboard": ucf_dashboard.render,
"Model Comparison": model_comparison.render,
"Performance Metrics": performance_metrics.render,
"Explainable AI": explainable_ai.render,
"UCF VS BigO Analysis": ucf_vs_bigo.render,
"AutoML Selection": automl_selection.render,
"Advanced Visualizations": advanced_visualizations.render,
"UCF Formula Lab": ucf_formula_lab.render,
"About": about.render,
}

def main():
    configure_page()
    apply_global_style()
    init_session_state()
    render_header()

```
page = render_sidebar()

if page in ROUTES:
    try:
        ROUTES[page]()
    except Exception as error:
        st.error("An error occurred while loading this page.")
        st.exception(error)
else:
    st.error("Unknown page selected: " + str(page))
```

if **name** == "**main**":
main()
