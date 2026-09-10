import numpy as np
import streamlit as st


def colored_metric(label, value):
    if not st.session_state.results_df.empty:
        numeric_vals = st.session_state.results_df.select_dtypes(include=np.number)
        mean_val = numeric_vals.values.mean() if not numeric_vals.empty else 0
        color = "green" if value < mean_val else "red"
    else:
        color = "green"
    st.markdown(
        f"""
        <div style="padding:10px; border-radius:10px; background-color:{color}; color:white">
        <h4>{label}</h4>
        <h3>{value:.4f}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
