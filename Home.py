"""
Home.py — Codexa Streamlit app entry point.
Mirrors the original index.html "About the model" landing page.
"""

import streamlit as st
from codexa_theme import inject_css, render_topnav, init_session_state

st.set_page_config(
    page_title="Codexa | Cross-Sensor Lunar Location Matching Engine",
    page_icon="🌕",
    layout="wide",
)

init_session_state()
inject_css()
render_topnav("Home")

# ---------------- Hero ----------------
col_l, col_r = st.columns([1.1, 0.9], gap="large")
with col_l:
    st.markdown(
        """
        <div class="hero-title">Two lunar photos.<br><span class="accent">One surface location.</span></div>
        <p class="lede">A hybrid computer-vision engine that decides whether two orbital images —
        captured by different sensors, missions, and sun angles — show the same patch of the Moon,
        then registers them pixel-for-pixel.</p>
        """,
        unsafe_allow_html=True,
    )
    b1, b2 = st.columns([1, 1])
    with b1:
        if st.button("🔍  Try the Matching Engine", type="primary", use_container_width=True):
            st.switch_page("pages/1_Match_Engine.py")
    with b2:
        st.page_link("pages/1_Match_Engine.py", label="See how it works ↓")
with col_r:
    st.markdown(
        """
        <div class="codexa-panel" style="text-align:center;background:radial-gradient(circle at 30% 25%,#efe8fd,transparent 55%),
        radial-gradient(circle at 75% 75%,#e0f4f8,transparent 55%);">
        <div style="font-size:5rem;">🌑🪐</div>
        <div style="color:var(--text-muted);font-size:.85rem;">Crater constellations + learned features, fused and RANSAC-verified.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ---------------- Architecture ----------------
st.markdown('<div style="color:var(--accent-cyan);font-family:var(--font-mono);font-size:.76rem;">DUAL-BRANCH FEATURE EXTRACTION</div>', unsafe_allow_html=True)
st.markdown('<h2 class="codexa-heading">A crater map and a neural network, cross-checking each other</h2>', unsafe_allow_html=True)
st.markdown(
    '<p style="color:var(--text-muted);max-width:70ch;">Craters are permanent, sensor-independent landmarks — so a classical '
    "geometry engine reads them the way a cartographer would, while a learned transformer reads pixel-level texture that "
    "craters alone can miss. Their candidate matches are fused before anything is trusted.</p>",
    unsafe_allow_html=True,
)

c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown(
        """
        <div class="codexa-card">
          <span class="branch-tag classical">Branch 1 · Classical</span>
          <h3 class="codexa-heading">Crater Constellation Engine</h3>
          <ol style="color:var(--text-muted);font-size:.86rem;line-height:1.8;padding-left:1.1rem;">
            <li><b>Crater detection</b> — circular rims found via Hough transforms / a trained detector.</li>
            <li><b>Delaunay graph</b> — detected centers are wired into a spatial triangulation.</li>
            <li><b>Invariant descriptors</b> — distance ratios, radius ratios, interior angles that survive scale and rotation.</li>
            <li><b>Constellation matching</b> — crater groups in A are matched to crater groups in B by invariant similarity.</li>
          </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
        <div class="codexa-card">
          <span class="branch-tag learned">Branch 2 · Learned</span>
          <h3 class="codexa-heading">LoFTR Feature Engine</h3>
          <ol style="color:var(--text-muted);font-size:.86rem;line-height:1.8;padding-left:1.1rem;">
            <li><b>Detector-free transformer</b> — self- and cross-attention match regions directly, no keypoint step needed.</li>
            <li><b>Tiling &amp; windowing</b> — large frames split into overlapping 640×640 tiles so memory stays bounded.</li>
            <li><b>Cross-tile dedup</b> — correspondences repeated across tile borders are merged into one.</li>
            <li><b>Low-texture resilience</b> — effective even over flat, feature-poor regolith where craters are sparse.</li>
          </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<div style="text-align:center;color:var(--text-muted);font-family:var(--font-mono);font-size:.78rem;margin:1.2rem 0 2rem;">'
    "— Correspondence fusion — deduplicated within 4.0 px, confidence-balanced across both branches —</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# ---------------- Pipeline ----------------
st.markdown('<div style="color:var(--accent-cyan);font-family:var(--font-mono);font-size:.76rem;">END-TO-END FLOW</div>', unsafe_allow_html=True)
st.markdown('<h2 class="codexa-heading">Six stages, from raw frames to a registered pair</h2>', unsafe_allow_html=True)

stages = [
    ("Preprocess", "Normalize illumination & scale across OHRC / TMC-2 / LROC inputs"),
    ("Detect / Match", "Run both branches to generate candidate correspondences"),
    ("Fuse", "Merge & deduplicate correspondences from both engines"),
    ("Verify", "RANSAC fits a 2D similarity transform, isolates inliers"),
    ("Register", "Warp & align image B into image A's coordinate space"),
    ("Evaluate", "Score quality & render the explainability visualizations"),
]
cols = st.columns(6)
for i, (title, desc) in enumerate(stages):
    with cols[i]:
        st.markdown(
            f"""<div class="stage-num">{i+1}</div>
            <div class="stage-title2">{title}</div>
            <div class="stage-desc">{desc}</div>""",
            unsafe_allow_html=True,
        )

st.markdown("---")

# ---------------- Acceptance ----------------
st.markdown('<div style="color:var(--accent-cyan);font-family:var(--font-mono);font-size:.76rem;">ZERO FALSE-POSITIVE TARGET</div>', unsafe_allow_html=True)
st.markdown('<h2 class="codexa-heading">A match is only accepted if all six checks pass</h2>', unsafe_allow_html=True)
st.markdown(
    '<p style="color:var(--text-muted);max-width:70ch;">Repetitive crater terrain makes it easy for a matcher to be '
    "confidently wrong. The Match Acceptance Engine treats every check as mandatory rather than advisory — one failure "
    "is enough to reject the pair.</p>",
    unsafe_allow_html=True,
)

st.table(
    {
        "Check": ["Minimum Inliers", "Minimum Inlier Ratio", "Maximum RMSE", "Spatial Coverage", "Pipeline Confidence", "Transform Sanity"],
        "Threshold": ["≥ 10 pts", "≥ 20%", "≤ 3.0 px", "≥ 15% grid", "≥ 50%", "finite, s > 0"],
        "What it proves": [
            "Enough geometric evidence exists",
            "Not just an accidental point cluster",
            "High spatial accuracy, low alignment error",
            "Matches span the surface, not one corner",
            "Overall statistical certainty of the match",
            "Mathematically valid transform (no NaN)",
        ],
    }
)

qcols = st.columns(5)
qbands = [
    ("Excellent", "RMSE < 1.0px · ≥30 inliers · ≥40% coverage"),
    ("Good", "RMSE < 2.0px · ≥15 inliers · ≥25% coverage"),
    ("Fair", "RMSE < 3.0px · ≥10 inliers · ≥15% coverage"),
    ("Poor", "Low coverage or high reprojection error"),
    ("Failed", "Unmatched or non-convergent pair"),
]
for i, (label, desc) in enumerate(qbands):
    with qcols[i]:
        st.markdown(f'<div class="qband"><div class="qlabel">{label}</div><div class="qdesc">{desc}</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ---------------- Feasibility ----------------
st.markdown('<div style="color:var(--accent-cyan);font-family:var(--font-mono);font-size:.76rem;">FEASIBILITY &amp; VIABILITY</div>', unsafe_allow_html=True)
st.markdown('<h2 class="codexa-heading">Built to survive real lunar imagery, not just clean demos</h2>', unsafe_allow_html=True)

f1, f2, f3 = st.columns(3)
with f1:
    st.markdown(
        """<div class="codexa-card" style="background:var(--accent-emerald-soft);border-color:#bfe8d6;">
        <h4 style="color:#046b4c;">Feasibility</h4>
        <ul style="font-size:.84rem;color:var(--text-muted);line-height:1.7;">
        <li>Python + open-source CV / deep-learning stack</li>
        <li>Modular branches — classical and learned engines run independently</li>
        <li>Works with OHRC, TMC-2, and IIRS inputs</li></ul></div>""",
        unsafe_allow_html=True,
    )
with f2:
    st.markdown(
        """<div class="codexa-card" style="background:var(--accent-amber-soft);border-color:#f3d9b3;">
        <h4 style="color:#8a4306;">Potential Challenges</h4>
        <ul style="font-size:.84rem;color:var(--text-muted);line-height:1.7;">
        <li>Illumination variation across sun angles</li>
        <li>Viewpoint &amp; scale variation between missions</li>
        <li>Multi-modal appearance differences</li>
        <li>Match clustering on repetitive terrain</li></ul></div>""",
        unsafe_allow_html=True,
    )
with f3:
    st.markdown(
        """<div class="codexa-card" style="background:var(--accent-cyan-soft);border-color:#bfe1ea;">
        <h4 style="color:var(--accent-cyan);">Mitigation Strategies</h4>
        <ul style="font-size:.84rem;color:var(--text-muted);line-height:1.7;">
        <li>Illumination &amp; scale normalization in preprocessing</li>
        <li>Multi-scale tiled matching</li>
        <li>Hybrid correspondence fusion across both branches</li>
        <li>RANSAC + the 6-check acceptance engine</li></ul></div>""",
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="codexa-panel" style="text-align:center;margin-top:2rem;
    background:linear-gradient(135deg,var(--accent-cyan-soft),var(--accent-purple-soft));">
      <h2 class="codexa-heading">Ready to test a pair of images?</h2>
      <p style="color:var(--text-muted);">Upload a reference and a query frame — watch the pipeline run,
      then step through every diagnostic view.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
if st.button("🚀  Launch the Matching Engine", type="primary"):
    st.switch_page("pages/1_Match_Engine.py")

st.caption("Codexa — Smart India Hackathon 2026 · Cross-Sensor Lunar Location Matching Engine")