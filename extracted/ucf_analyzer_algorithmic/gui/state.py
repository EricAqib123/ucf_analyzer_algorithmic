import pandas as pd
import streamlit as st


def init_session_state():
    defaults = {
        "results_df": pd.DataFrame(),
        "sorting_done": False,
        "ml_done": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
