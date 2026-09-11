import streamlit as st
from pathlib import Path

# Project root:
#extracted/ucf_analyzer_algorithmic/

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMG_DIR = PROJECT_ROOT / "img"

LOGO_PATH = IMG_DIR / "logo.png"
LOGO1_PATH2 = IMG_DIR / "logo1.png"

PAGES = (
    "Home",
    "Classical Algorithm",
    "Machine Learning",
    "UCF Dashboard",
    "Model Comparison",
    "Performance Metrics",
    "Explainable AI",
    "UCF VS BigO Analysis",
    "AutoML Selection",
    "Advanced Visualizations",
    "UCF Formula Lab", 
    "About",
)

def configure_page():
    st.set_page_config(
    page_title="AI-Based Algorithmic Analyzer (UCF Model)",
    page_icon=str(LOGO1_PATH2) if LOGO1_PATH2.exists() else "🧠",
    layout="wide",
    )

def render_header():
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=80)
    with col2:
        st.markdown(
            "<h1 style='text-align:center;'>"
            "AI-Based Algorithmic Analyzer(UCF)"
            "</h1>",
             unsafe_allow_html=True,
        )

        st.markdown(
            "<h4 style='text-align:center;'>"
            "Mehran University of Engineering & Technology"
            "</h4>",
             unsafe_allow_html=True,
        )

        st.markdown(
            "<h5 style='text-align:center;'>"
            "Jamshoro, Sindh, Pakistan"
            "</h5>",
            unsafe_allow_html=True,
        )
    with col3:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=80)
            st.markdown("---")
            
def render_sidebar():
    if LOGO1_PATH.exists():
        st.sidebar.image(str(LOGO1_PATH2), width=100)
        st.sidebar.title("AI Algorithm Analyzer")
        return st.sidebar.radio("Select Phase:",PAGES,)

def apply_global_style():
    st.markdown(
        """ <style>
        .sidebar .sidebar-content {
        background-color: #f5f5f5;
        }
        h1, h2, h3 {
        color: #003366;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
