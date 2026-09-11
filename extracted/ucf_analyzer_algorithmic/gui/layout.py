from pathlib import Path

import streamlit as st

# ---------------------------------------------------------

# PROJECT PATHS

# ---------------------------------------------------------

PROJECT_ROOT = Path(**file**).resolve().parent.parent

IMG_DIR = PROJECT_ROOT / "img"

LOGO_PATH = IMG_DIR / "logo.png"

# ---------------------------------------------------------

# PAGE NAMES

# ---------------------------------------------------------

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

# ---------------------------------------------------------

# PAGE CONFIGURATION

# ---------------------------------------------------------

def configure_page():
"""
Configure Streamlit page settings.
"""

```
# Use the logo only if it actually exists.
if LOGO_PATH.exists():
    page_icon = str(LOGO_PATH)
else:
    page_icon = "🧠"

st.set_page_config(
    page_title="AI-Based Algorithmic Analyzer (UCF Model)",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)
```

# ---------------------------------------------------------

# HEADER

# ---------------------------------------------------------

def render_header():
"""
Render the application header.
"""

```
col1, col2, col3 = st.columns([1, 4, 1])

with col1:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=80)

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

st.markdown("---")
```

# ---------------------------------------------------------

# SIDEBAR

# ---------------------------------------------------------

def render_sidebar():
"""
Render sidebar navigation.
"""

```
if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH), width=100)

st.sidebar.title("AI Algorithm Analyzer")

selected_page = st.sidebar.radio(
    "Select Phase:",
    PAGES,
)

return selected_page
```

# ---------------------------------------------------------

# GLOBAL CSS

# ---------------------------------------------------------

def apply_global_style():
"""
Apply application-wide CSS.
"""

```
st.markdown(
    """
    <style>

    [data-testid="stSidebar"] {
        background-color: #f5f5f5;
    }

    h1,
    h2,
    h3 {
        color: #003366;
    }

    .stButton > button {
        border-radius: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)
```
