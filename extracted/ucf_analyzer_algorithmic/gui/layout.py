import streamlit as st
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

# layout.py is inside:
# project_root/gui/layout.py
#
# Therefore:
# parent= gui
# parent.parent = project_root

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMG_DIR = PROJECT_ROOT / "img"

UNVI_PATH = IMG_DIR / "logo.png"
SIDEBAR_PATH = IMG_DIR / "main.png"


# ============================================================
# PAGES
# ============================================================

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


# ============================================================
# PAGE CONFIGURATION
# ============================================================

def configure_page():
    st.set_page_config(
        page_title="AI-Based Algorithmic Analyzer (UCF Model)",
        page_icon=str(UNVI_PATH) if UNVI_PATH.exists() else "🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ============================================================
# HEADER
# ============================================================

def render_header():
    col1, col2, col3 = st.columns([1, 4, 1])

    with col1:
        if UNVI_PATH.exists():
            st.image(str(UNVI_PATH), width=80)

    with col2:
        st.markdown(
            """
            <h1 style="text-align:center;">
                AI-Based Algorithmic Analyzer (UCF)
            </h1>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <h4 style="text-align:center;">
                Mehran University of Engineering & Technology
            </h4>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <h5 style="text-align:center;">
                Jamshoro, Sindh, Pakistan
            </h5>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=80)


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():

    # Sidebar should ALWAYS be created.
    # The logo is optional.

    if SIDEBAR_PATH.exists():
        st.sidebar.image(str(SIDEBAR_PATH), width=100)

    st.sidebar.title("AI Algorithm Analyzer")

    page = st.sidebar.radio(
        "Select Page:",
        PAGES,
        index=0,
    )

    return page


# ============================================================
# GLOBAL CSS
# ============================================================

def apply_global_style():
    st.markdown(
        """
        <style>

        h1, h2, h3 {
            color: #003366;
        }

        [data-testid="stSidebar"] {
            background-color: #f5f5f5;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )
