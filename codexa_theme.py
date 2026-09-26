"""
codexa_theme.py

Shared visual theme, CSS, and small HTML-snippet helpers used by every page
of the Codexa Streamlit app, so the app matches the original Codexa
HTML/CSS design system (light mode, cyan/purple gradient brand, Outfit /
Inter / JetBrains Mono type, panel & card system, status pills, checks
grid, etc.)
"""

import streamlit as st

VIZ_PAGES = [
    {
        "key": "correspondence",
        "label": "Correspondences Plot",
        "short": "Line-by-line vectors connecting every matched keypoint between the two frames.",
    },
    {
        "key": "overlay",
        "label": "Registration Overlay",
        "short": "Image B warped and blended over Image A in false color to check alignment.",
    },
    {
        "key": "checkerboard",
        "label": "Checkerboard Comparison",
        "short": "Alternating tiles from A and warped B — seams reveal any residual misalignment.",
    },
    {
        "key": "cmap_a",
        "label": "Image A Evidence Map",
        "short": "Heatmap of match confidence across Image A's surface.",
    },
    {
        "key": "cmap_b",
        "label": "Image B Evidence Map",
        "short": "Heatmap of match confidence across Image B's surface.",
    },
]

STATS_KEY = "lunar_session_stats"
RESULT_KEY = "lunar_last_result"
VIZ_KEY = "lunar_last_viz"


def init_session_state():
    if STATS_KEY not in st.session_state:
        st.session_state[STATS_KEY] = {"tested": 0, "accepted": 0, "rejected": 0}
    if RESULT_KEY not in st.session_state:
        st.session_state[RESULT_KEY] = None
    if VIZ_KEY not in st.session_state:
        st.session_state[VIZ_KEY] = {}


def bump_stats(matched: bool):
    s = st.session_state[STATS_KEY]
    s["tested"] += 1
    if matched:
        s["accepted"] += 1
    else:
        s["rejected"] += 1

def inject_css():
    css_content = """
    :root {
        --bg: #f5f7fb;
        --bg-topo: #eef1f7;
        --surface: #ffffff;
        --surface-alt: #fbfcfe;
        --border: #dde3ee;
        --border-strong: #c7d0e0;
        --text-main: #111726;
        --text-muted: #5c6579;
        --text-faint: #8b93a3;

        --accent-cyan: #0891b2;
        --accent-cyan-soft: #e0f4f8;
        --accent-purple: #6d28d9;
        --accent-purple-soft: #efe8fd;
        --accent-emerald: #059669;
        --accent-emerald-soft: #e2f6ee;
        --accent-amber: #b45309;
        --accent-amber-soft: #fdf1e2;
        --accent-rose: #e11d48;
        --accent-rose-soft: #fde6ea;

        --font-heading: 'Outfit', sans-serif;
        --font-body: 'Inter', sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
        --radius: 14px;
        --shadow-md: 0 8px 24px rgba(17, 23, 38, .07);
    }

    html, body, [class*="css"] { font-family: var(--font-body); color: var(--text-main); }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        overflow-x: hidden !important;
    }

    .main .block-container, [data-testid="stMainBlockContainer"] {
        max-width: 1200px !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        margin: 0 auto !important;
    }

    .stApp, [data-testid="stAppViewContainer"] {
        background-color: var(--bg) !important;
        background-image: linear-gradient(var(--bg-topo) 1px, transparent 1px),
                           linear-gradient(90deg, var(--bg-topo) 1px, transparent 1px) !important;
        background-size: 42px 42px !important;
        color: var(--text-main) !important;
    }

    [data-testid="stHeader"] { background: transparent !important; }

    [data-testid="stSidebar"] {
        background-color: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }

    h1, h2, h3, h4, .codexa-heading {
        font-family: var(--font-heading) !important;
        color: var(--text-main) !important;
        letter-spacing: -0.01em;
    }

    .codexa-topnav {
        display: flex; justify-content: space-between; align-items: center;
        padding: .5rem 1rem 1.25rem; flex-wrap: wrap; gap: .8rem;
        border-bottom: 1px solid var(--border); margin-bottom: 1.5rem;
    }
    .codexa-brand { display: flex; align-items: center; gap: .7rem; text-decoration: none !important; color: inherit !important; }
    .codexa-brand:hover { text-decoration: none !important; }
    .moon-icon {
        width: 36px; height: 36px; border-radius: 50%; flex-shrink: 0;
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 4px 14px rgba(8, 145, 178, .35); font-size: 16px;
    }
    .codexa-brand-name { font-family: var(--font-heading); font-weight: 700; font-size: 1rem; color: var(--text-main) !important; }
    .codexa-brand-sub { font-size: .68rem; color: var(--text-muted) !important; }

    .nav-links { display: flex; align-items: center; gap: 1.5rem; }
    .nav-link { color: var(--text-muted) !important; text-decoration: none !important; font-size: .88rem; font-weight: 500; transition: color .2s; cursor: pointer; }
    .nav-link:hover { color: var(--accent-cyan) !important; text-decoration: none !important; }
    .nav-link.current { color: var(--accent-cyan) !important; font-weight: 700; border-bottom: 2px solid var(--accent-cyan); padding-bottom: 2px; }

    .header-right { display: flex; align-items: center; gap: .8rem; flex-wrap: wrap; }
    .pill { display: inline-flex; align-items: center; gap: .4rem; padding: .32rem .75rem; border-radius: 9999px;
        font-size: .7rem; font-weight: 600; font-family: var(--font-mono); letter-spacing: .02em; border: 1px solid transparent; }
    .pill-sih { background: var(--accent-amber-soft); color: var(--accent-amber); border-color: #f3d9b3; }
    .pill-live { background: var(--accent-emerald-soft); color: var(--accent-emerald); border-color: #bfe8d6; }
    
    .status-badge {
        display: inline-flex; align-items: center; gap: .5rem; padding: .4rem .85rem; border-radius: 9999px;
        font-size: .78rem; font-weight: 500; background: var(--accent-emerald-soft); border: 1px solid #bfe8d6; color: var(--accent-emerald);
    }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; box-shadow: 0 0 6px currentColor; }
    .session-counter-badge {
        display: inline-flex; align-items: center; gap: .6rem; padding: .4rem .85rem; border-radius: 9999px;
        font-size: .76rem; font-family: var(--font-mono); background: var(--accent-cyan-soft); border: 1px solid #bfe1ea; color: var(--accent-cyan);
    }

    .codexa-panel {
        background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
        box-shadow: var(--shadow-md); padding: 1.5rem 1.75rem; margin-bottom: 1.4rem;
    }
    .codexa-card {
        background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
        box-shadow: 0 2px 8px rgba(17, 23, 38, .05); padding: 1.2rem 1.4rem;
    }

    .results-banner { padding: 1rem 1.5rem; border-radius: 12px; margin-bottom: 1.2rem; border: 1px solid transparent; }
    .results-banner.success { background: var(--accent-emerald-soft); border-color: #bfe8d6; color: #046b4c; }
    .results-banner.rejected { background: var(--accent-rose-soft); border-color: #f3b9c6; color: #a3123a; }
    .results-banner.uncertain { background: var(--accent-amber-soft); border-color: #f3d9b3; color: #8a4306; }
    .banner-title { font-family: var(--font-heading); font-size: 1.2rem; font-weight: 700; margin-bottom: .3rem; }

    .rejection-box { background: var(--accent-rose-soft); border: 1px solid #f3b9c6; border-radius: 12px; padding: 1.1rem 1.3rem; margin-bottom: 1.2rem; }
    .rejection-title { font-family: var(--font-heading); font-weight: 700; color: #a3123a; margin-bottom: .4rem; }
    .rejection-reason { font-weight: 600; color: #8a0e30; margin-bottom: .3rem; }
    .rejection-note { font-size: .78rem; color: #a3536b; font-style: italic; }

    .metric-card { padding: 1rem 1.15rem; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); box-shadow: var(--shadow-md); }
    .metric-label { font-size: .78rem; color: var(--text-muted); margin-bottom: .35rem; }
    .metric-value { font-family: var(--font-heading); font-size: 1.5rem; font-weight: 700; color: var(--text-main); }
    .metric-sub { font-size: .72rem; color: var(--accent-cyan); margin-top: .2rem; font-family: var(--font-mono); }

    .acceptance-decision-badge { font-family: var(--font-heading); font-size: 1.05rem; font-weight: 800; padding: .4rem 1.1rem; border-radius: 8px; letter-spacing: .04em; display: inline-block; }
    .acceptance-decision-badge.accepted { background: var(--accent-emerald-soft); border: 1px solid #bfe8d6; color: #046b4c; }
    .acceptance-decision-badge.rejected { background: var(--accent-rose-soft); border: 1px solid #f3b9c6; color: #a3123a; }
    .score-value { font-family: var(--font-heading); font-size: 1.3rem; font-weight: 700; color: var(--accent-cyan); }
    .score-subtext { font-size: .68rem; color: var(--text-faint); font-style: italic; }

    .check-card { background: var(--surface-alt); border: 1px solid var(--border); border-radius: 10px; padding: .9rem 1rem; margin-bottom: .7rem; }
    .check-card.pass { border-left: 4px solid var(--accent-emerald); }
    .check-card.fail { border-left: 4px solid var(--accent-rose); }
    .check-name { font-weight: 600; font-size: .86rem; color: var(--text-main); }
    .check-status-pill { font-family: var(--font-mono); font-size: .65rem; font-weight: 700; padding: .18rem .45rem; border-radius: 4px; float: right; }
    .check-status-pill.pass { background: var(--accent-emerald-soft); color: var(--accent-emerald); }
    .check-status-pill.fail { background: var(--accent-rose-soft); color: var(--accent-rose); }
    .check-val-row { font-family: var(--font-mono); font-size: .82rem; color: var(--accent-cyan); margin-top: .3rem; }
    .check-reason { font-size: .73rem; color: var(--text-muted); margin-top: .2rem; }

    .flow-step { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: .7rem 1rem; height: 100%; }
    .flow-step-num { font-family: var(--font-mono); font-size: .65rem; color: var(--accent-purple); text-transform: uppercase; }
    .flow-step-heading { font-weight: 600; font-size: .82rem; margin: .15rem 0; color: var(--text-main); }
    .flow-step-desc { font-size: .72rem; color: var(--text-muted); }

    .transform-values div { font-family: var(--font-mono); font-size: .85rem; color: var(--text-muted); margin-bottom: .5rem; }
    .transform-values span { color: var(--accent-cyan); }

    .hero-title { font-family: var(--font-heading); font-weight: 700; font-size: clamp(2rem, 4vw, 3rem); line-height: 1.1; color: var(--text-main); }
    .hero-title .accent { color: var(--accent-cyan); }
    .lede { color: var(--text-muted); font-size: 1.02rem; line-height: 1.6; max-width: 56ch; }

    /* Streamlit widget restyling */
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)) !important;
        border: none !important; font-family: var(--font-heading) !important; font-weight: 600 !important;
        box-shadow: 0 6px 18px rgba(8, 145, 178, .28) !important;
        color: #ffffff !important;
        border-radius: 11px !important;
        padding: .85rem 1.7rem !important;
    }
    .stButton>button[kind="primary"]:hover {
        box-shadow: 0 8px 22px rgba(8, 145, 178, .4) !important;
        color: #ffffff !important;
    }

    /* Secondary / default buttons — fix solid black box issue */
    .stButton>button:not([kind="primary"]),
    button[kind="secondary"],
    [data-testid="stBaseButton-secondary"],
    div[data-testid="stFileUploader"] button,
    button[data-baseweb="button"] {
        background: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        font-family: var(--font-body) !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    }
    .stButton>button:not([kind="primary"]):hover,
    button[kind="secondary"]:hover,
    [data-testid="stBaseButton-secondary"]:hover,
    div[data-testid="stFileUploader"] button:hover {
        background: #f8fafc !important;
        color: #0284c7 !important;
        border-color: #0284c7 !important;
    }

    /* File uploader high contrast text */
    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed var(--border-strong) !important; background: var(--surface-alt) !important;
        border-radius: 12px !important; padding: 1.5rem !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent-cyan) !important; background: var(--accent-cyan-soft) !important; }
    [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploaderDropzoneInstructions"] *,
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span {
        color: #334155 !important;
        font-weight: 500 !important;
    }

    .session-counter-badge {
        display: inline-flex; align-items: center; gap: .6rem; padding: .45rem .9rem; border-radius: 9999px;
        font-size: .78rem; font-family: var(--font-mono); background: var(--accent-cyan-soft); border: 1px solid #bfe1ea; color: var(--accent-cyan);
        white-space: nowrap; line-height: 1.4;
    }

    [data-testid="stMetricValue"] { font-family: var(--font-heading) !important; color: var(--accent-cyan) !important; }
    [data-testid="stMetricLabel"] { font-family: var(--font-body) !important; color: var(--text-muted) !important; }
    div[data-testid="stExpander"] { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; }

    /* Image display sizing & centering for UI */
    [data-testid="stImage"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin-bottom: 0.5rem !important;
    }
    [data-testid="stImage"] img {
        max-height: 450px !important;
        width: auto !important;
        object-fit: contain !important;
        border-radius: 8px !important;
        border: 1px solid var(--border) !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08) !important;
    }
    div[data-testid="stExpander"] [data-testid="stImage"] img {
        max-height: 85vh !important;
    }
    """
    clean_css = " ".join(css_content.split())
    st.markdown(f"<style>{clean_css}</style>", unsafe_allow_html=True)


def render_topnav(current: str = "Home"):
    c_home = "current" if current == "Home" else ""
    c_engine = "current" if current == "Match Engine" else ""
    c_results = "current" if current == "Results" else ""
    
    html = f"""<div class="codexa-topnav">
  <a href="./" target="_self" class="codexa-brand">
    <div class="moon-icon">
      <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="#ffffff" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>
    </div>
    <div>
      <div class="codexa-brand-name">Codexa</div>
      <div class="codexa-brand-sub">Lunar Match &amp; Register</div>
    </div>
  </a>
  <div class="nav-links">
    <a href="./" target="_self" class="nav-link {c_home}">About the model</a>
    <a href="./Match_Engine" target="_self" class="nav-link {c_engine}">Match Engine</a>
    <a href="./Results" target="_self" class="nav-link {c_results}">Last results ↗</a>
  </div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)


def render_session_badge():
    s = st.session_state[STATS_KEY]
    html = f"""<div class="header-right">
  <div class="session-counter-badge">
    <b>TEST SESSION</b>&nbsp;&nbsp;Pairs Tested: {s['tested']} | Accepted: {s['accepted']} | Rejected: {s['rejected']}
  </div>
  <div class="status-badge">
    <div class="status-dot"></div>
    <span>API Ready &amp; Model Loaded</span>
  </div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)


def panel_open():
    if hasattr(st, "html"):
        st.html('<div class="codexa-panel">')
    else:
        st.markdown('<div class="codexa-panel">', unsafe_allow_html=True)


def panel_close():
    if hasattr(st, "html"):
        st.html('</div>')
    else:
        st.markdown('</div>', unsafe_allow_html=True)