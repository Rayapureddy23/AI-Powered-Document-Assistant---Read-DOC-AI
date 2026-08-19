"""
ui.py — Shared theme, sidebar status, and PDF preview helpers.
"""

import base64
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


def get_previewable_paths(paths):
    """Return only the PDF files from a list of uploaded document paths."""
    if not paths:
        return []
    return [p for p in paths if isinstance(p, str) and os.path.splitext(p)[1].lower() == ".pdf"]


def build_pdf_preview_html(file_path: str) -> str:
    """Create an HTML snippet that embeds a PDF for inline preview in Streamlit."""
    if not os.path.exists(file_path):
        return "<div style='padding:12px;color:#6B7280'>PDF file not found.</div>"

    with open(file_path, "rb") as handle:
        pdf_bytes = handle.read()

    pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")
    return f"""
    <div style="width:100%;border:1px solid #E5E7EB;border-radius:12px;overflow:hidden;">
      <iframe src="data:application/pdf;base64,{pdf_b64}" width="100%" height="520px" type="application/pdf" style="border:none;"></iframe>
    </div>
    """


def apply_theme(title: str, subtitle: str = "", show_status: bool = True):
    st.markdown(CSS, unsafe_allow_html=True)
    if show_status:
        doc_status_sidebar()
    st.title(title)
    if subtitle:
        st.caption(subtitle)
    st.divider()