"""
ui.py — Shared theme and sidebar status.
Purple professional theme. The sidebar status block appears on EVERY
page so the uploaded document and index state are always visible.
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


def _doc_status_sidebar():
    """Always-visible status of document, indexes and LLM — on every page."""
    with st.sidebar:
        st.markdown("#### 📄 Document status")
        paths = st.session_state.get("file_paths") or _restore_uploads()
        if paths:
            for p in paths:
                st.markdown(f"✅ **{os.path.basename(p)}**")
        else:
            st.caption("No document uploaded yet — go to the **app** page.")

        built = sorted(st.session_state.get("built_sizes", set()))
        if built:
            st.markdown("**Indexes built:** " + ", ".join(f"`{b}`" for b in built))
        else:
            st.caption("No indexes built yet.")
        st.divider()


def _restore_uploads():
    """After a full restart, re-discover previously uploaded files on disk."""
    up_dir = os.path.join("data", "uploads")
    if os.path.isdir(up_dir):
        paths = [os.path.abspath(os.path.join(up_dir, f))
                 for f in sorted(os.listdir(up_dir))
                 if f.lower().endswith((".pdf", ".html", ".htm", ".txt"))]
        if paths:
            st.session_state["file_paths"] = paths
            return paths
    return []


def apply_theme(title: str, subtitle: str = "", show_status: bool = True):
    st.markdown(CSS, unsafe_allow_html=True)
    if show_status:
        _doc_status_sidebar()
    st.title(title)
    if subtitle:
        st.caption(subtitle)
    st.divider()
