"""
streamlit_app.py — Codexa Multi-Page Streamlit App Entrypoint.

Delegates execution to Home.py so that both `streamlit run Home.py`
and `streamlit run streamlit_app.py` (used by Streamlit Cloud) load
the new multi-page Codexa Streamlit UI.
"""

import os
import runpy

home_path = os.path.join(os.path.dirname(__file__), "Home.py")
runpy.run_path(home_path, run_name="__main__")
