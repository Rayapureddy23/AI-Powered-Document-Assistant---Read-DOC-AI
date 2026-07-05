"""
ui.py — Shared theme and sidebar status.
Purple professional theme. Shows ONLY the document uploaded in the
current session — nothing is restored from previous runs.
"""

import os
import streamlit as st

PRIMARY = "#5C2D91"   # deep purple
ACCENT  = "#7C3AED"
LIGHT   = "#F7F5FB"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background: {LIGHT}; }}
.main .block-container {{
    background: #FFFFFF;
    border-radius: 14px;
    padding: 2.2rem 2.8rem;
    margin-top: 1rem;
    box-shadow: 0 1px 3px rgba(92, 45, 145, 0.10);
    max-width: 1100px;
}}
h1 {{ font-size: 1.6rem; font-weight: 700; color: #2D1B4E; }}
h2 {{ font-size: 1.15rem; font-weight: 600; color: #2D1B4E;
     border-bottom: 2px solid #EDE9F7; padding-bottom: 6px; margin-top: 1.6rem; }}
h3 {{ font-size: 1.0rem; font-weight: 600; color: #3B2A5E; }}
[data-testid="stMetricValue"] {{ font-size: 1.5rem; color: {PRIMARY}; }}
.stButton > button[kind="primary"] {{
    background: {PRIMARY}; border-color: {PRIMARY};
    border-radius: 8px; font-weight: 600;
}}
.stButton > button[kind="primary"]:hover {{ background: {ACCENT}; border-color: {ACCENT}; }}
[data-testid="stSidebar"] {{ background: #FBFAFD; border-right: 1px solid #EDE9F7; }}
</style>
"""


def doc_status_sidebar():
    """Current-session document status only. No restore from disk."""
    with st.sidebar:
        st.markdown("#### 📄 Document")
        paths = st.session_state.get("file_paths") or []
        if paths:
            for p in paths:
                st.markdown(f"✅ **{os.path.basename(p)}**")
            built = sorted(st.session_state.get("built_sizes", set()))
            if built:
                st.caption("Indexes: " + ", ".join(str(b) for b in built) + " chars")
        else:
            st.caption("No document uploaded — go to the **app** page.")
        st.divider()


def apply_theme(title: str, subtitle: str = "", show_status: bool = True):
    st.markdown(CSS, unsafe_allow_html=True)
    if show_status:
        doc_status_sidebar()
    st.title(title)
    if subtitle:
        st.caption(subtitle)
    st.divider()