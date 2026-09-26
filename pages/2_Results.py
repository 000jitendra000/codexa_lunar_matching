"""
pages/2_Results.py

Mirrors the original results.html screen: decision banner, metrics grid,
acceptance checks grid, RANSAC→Acceptance→Decision flow, estimated
transform, and the 5-view explainability gallery (correspondences,
registration overlay, checkerboard, and the two confidence maps) with a
pager + full-resolution modal, same as results.html / results-*.html.
"""

import json
import os

import streamlit as st

from codexa_theme import (
    inject_css, render_topnav, render_session_badge, init_session_state,
    RESULT_KEY, VIZ_KEY, VIZ_PAGES,
)

st.set_page_config(page_title="Match Results | Codexa", page_icon="🌕", layout="wide")
init_session_state()
inject_css()
render_topnav("Results")

payload = st.session_state.get(RESULT_KEY)
viz_paths = st.session_state.get(VIZ_KEY, {})

top_l, top_r = st.columns([3, 1])
with top_l:
    st.markdown('<h1 class="codexa-heading">Match Result</h1>', unsafe_allow_html=True)
    if payload:
        st.caption(f"Job ID: {payload.get('jobId','—')}")
    else:
        st.caption("No job loaded yet")
with top_r:
    render_session_badge()

st.markdown("---")

if not payload:
    st.markdown(
        """<div class="codexa-panel" style="text-align:center;padding:3rem 1.5rem;">
        <h3 class="codexa-heading">No match result to show yet</h3>
        <p style="color:var(--text-muted);">Run a pair of images through the engine to see the
        acceptance decision and every diagnostic view.</p></div>""",
        unsafe_allow_html=True,
    )
    if st.button("Go to Match Engine", type="primary"):
        st.switch_page("pages/1_Match_Engine.py")
    st.stop()

res = payload["result"]
acc = res.get("acceptance") or {}
is_accepted = acc.get("accepted", res.get("matched"))

# ---------------- Action bar ----------------
a1, a2 = st.columns([1, 1])
with a2:
    if st.button("🔁  Test Another Pair", use_container_width=True):
        st.switch_page("pages/1_Match_Engine.py")

# ---------------- Banner ----------------
if res.get("matched") is True:
    banner_class, title, sub = "success", "🟢 LOCATION MATCH ACCEPTED", "Acceptance criteria satisfied"
elif acc.get("status") == "REJECTED":
    banner_class, title = "rejected", "🔴 LOCATION MATCH REJECTED"
    sub = acc.get("reason") or "Acceptance criteria not satisfied"
else:
    banner_class, title = "uncertain", "🟡 Match Uncertain / No Consensus"
    sub = "Insufficient spatial inliers for registration"

st.markdown(
    f"""<div class="results-banner {banner_class}">
    <div class="banner-title">{title}</div><div>{sub}</div></div>""",
    unsafe_allow_html=True,
)

if banner_class == "rejected":
    st.markdown(
        f"""<div class="rejection-box">
        <div class="rejection-title">⚠️ Why was this match rejected?</div>
        <div class="rejection-reason">{acc.get('reason') or 'Insufficient geometric evidence.'}</div>
        <div class="rejection-note">RANSAC found a geometrically consistent subset, but the evidence did not
        satisfy the location-match acceptance standard.</div></div>""",
        unsafe_allow_html=True,
    )

# ---------------- Metrics grid ----------------
cov = res.get("coverage")
score = acc.get("acceptance_score")
score_pct = f"{score*100:.1f}%" if score is not None else "N/A"

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.markdown(
        f"""<div class="codexa-card metric-card"><div class="metric-label">Inlier Correspondences</div>
        <div class="metric-value">{res.get('inliers',0)}</div>
        <div class="metric-sub">Total Candidate Ties: {res.get('correspondences',0)}</div></div>""",
        unsafe_allow_html=True,
    )
with m2:
    ir = res.get("inlier_ratio")
    st.markdown(
        f"""<div class="codexa-card metric-card"><div class="metric-label">Inlier Ratio</div>
        <div class="metric-value">{f'{ir*100:.1f}%' if ir is not None else 'N/A'}</div>
        <div class="metric-sub">RANSAC Consensus</div></div>""",
        unsafe_allow_html=True,
    )
with m3:
    rmse = res.get("rmse")
    st.markdown(
        f"""<div class="codexa-card metric-card"><div class="metric-label">Reprojection RMSE</div>
        <div class="metric-value">{f'{rmse:.2f} px' if rmse is not None else 'N/A'}</div>
        <div class="metric-sub">{res.get('quality') or 'N/A'}</div></div>""",
        unsafe_allow_html=True,
    )
with m4:
    conf = res.get("confidence")
    st.markdown(
        f"""<div class="codexa-card metric-card"><div class="metric-label">Match Confidence</div>
        <div class="metric-value">{f'{conf*100:.1f}%' if conf is not None else 'N/A'}</div>
        <div class="metric-sub">Pipeline Certainty</div></div>""",
        unsafe_allow_html=True,
    )
with m5:
    st.markdown(
        f"""<div class="codexa-card metric-card"><div class="metric-label">Spatial Coverage</div>
        <div class="metric-value">{f'{cov*100:.1f}%' if cov is not None else 'N/A'}</div>
        <div class="metric-sub">Inlier grid coverage</div></div>""",
        unsafe_allow_html=True,
    )

st.write("")

# ---------------- Acceptance panel ----------------
badge_cls = "accepted" if is_accepted else "rejected"
h1, h2 = st.columns([2, 1])
with h1:
    st.markdown('<div style="font-size:.8rem;color:var(--text-muted);">Match Acceptance Decision</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="acceptance-decision-badge {badge_cls}">{"ACCEPTED" if is_accepted else "REJECTED"}</div>', unsafe_allow_html=True)
with h2:
    st.markdown(
        f"""<div style="text-align:right;"><div style="font-size:.76rem;color:var(--text-muted);">Evidence Quality</div>
        <div class="score-value">{score_pct}</div>
        <div class="score-subtext">Acceptance Score (deterministic — not a probability)</div></div>""",
        unsafe_allow_html=True,
    )

st.markdown(
    '<div style="font-size:.72rem;color:var(--text-faint);margin:.6rem 0 1rem;">'
    "Standard Thresholds: Inliers ≥10 | Ratio ≥20% | RMSE ≤3.0px | Coverage ≥15% | Conf ≥50%</div>",
    unsafe_allow_html=True,
)

checks = acc.get("checks") or {}
NAME_MAP = {
    "minimum_inliers": "Minimum Inliers", "minimum_inlier_ratio": "Inlier Ratio",
    "maximum_rmse": "Reprojection RMSE", "minimum_coverage": "Spatial Coverage",
    "minimum_confidence": "Match Confidence", "transform_sanity": "Transform Sanity",
}
if not checks:
    st.caption("No individual acceptance check details returned for this run.")
else:
    check_items = list(checks.items())
    cols = st.columns(3)
    for i, (key, chk) in enumerate(check_items):
        name = NAME_MAP.get(key, key.replace("_", " ").title())
        passed = bool(chk.get("passed"))
        val, thresh = chk.get("value"), chk.get("threshold")
        if key == "minimum_inliers":
            val_s, thresh_s = f"{val or 0}", f"{thresh or 10} min"
        elif key == "minimum_inlier_ratio":
            val_s = f"{val*100:.1f}%" if val is not None else "0.0%"
            thresh_s = f"{thresh*100:.1f}% min" if thresh is not None else "20.0% min"
        elif key == "maximum_rmse":
            val_s = f"{val:.2f} px" if val is not None else "N/A"
            thresh_s = f"{thresh:.2f} px max" if thresh is not None else "3.00 px max"
        elif key == "minimum_coverage":
            val_s = f"{val*100:.1f}%" if val is not None else "0.0%"
            thresh_s = f"{thresh*100:.1f}% min" if thresh is not None else "15.0% min"
        elif key == "minimum_confidence":
            val_s = f"{val*100:.1f}%" if val is not None else "0.0%"
            thresh_s = f"{thresh*100:.1f}% min" if thresh is not None else "50.0% min"
        elif key == "transform_sanity":
            val_s = "Finite / positive scale" if val is True else "Invalid transform"
            thresh_s = "Finite + scale > 0"
        else:
            val_s, thresh_s = str(val), str(thresh) if thresh is not None else ""

        with cols[i % 3]:
            reason_html = f'<div class="check-reason">{chk.get("reason")}</div>' if chk.get("reason") else ""
            st.markdown(
                f"""<div class="check-card {'pass' if passed else 'fail'}">
                <span class="check-status-pill {'pass' if passed else 'fail'}">{'PASS' if passed else 'FAIL'}</span>
                <div class="check-name">{name}</div>
                <div class="check-val-row">{val_s} <span style="color:var(--text-faint);">/ {thresh_s}</span></div>
                {reason_html}</div>""",
                unsafe_allow_html=True,
            )

st.write("")

# ---------------- Flow box ----------------
st.markdown('<div style="font-family:var(--font-heading);font-size:.9rem;color:var(--accent-cyan);margin-bottom:.6rem;">How the Decision Works: Raw Geometry vs Location Match Acceptance</div>', unsafe_allow_html=True)
fl1, fl2, fl3 = st.columns(3)
flow = [
    ("Stage 1", "RANSAC Consensus", "Determines whether a geometrically consistent planar transformation matrix can be estimated."),
    ("Stage 2", "Match Acceptance Engine", "Evaluates amount, ratio, coverage, confidence, and transform sanity against hard evidence thresholds."),
    ("Stage 3", "Location Match Decision", "Deterministic ACCEPTED or REJECTED decision for planetary registration."),
]
for col, (num, heading, desc) in zip((fl1, fl2, fl3), flow):
    with col:
        st.markdown(
            f"""<div class="flow-step"><div class="flow-step-num">{num}</div>
            <div class="flow-step-heading">{heading}</div><div class="flow-step-desc">{desc}</div></div>""",
            unsafe_allow_html=True,
        )

st.write("")

# ---------------- Transform box ----------------
tf = res.get("transform") or {}
sanity = checks.get("transform_sanity", {}).get("passed") if checks else None
st.markdown('<div style="font-family:var(--font-heading);font-size:.95rem;margin-bottom:.5rem;">Estimated Planar Similarity Transform (2D)</div>', unsafe_allow_html=True)
scale_v = tf.get("scale")
rot_v = tf.get("rotation_deg")
trans_v = tf.get("translation") or {}
scale_s = f"{scale_v:.4f}" if scale_v is not None else "N/A"
rot_s = f"{rot_v:.2f}°" if rot_v is not None else "N/A"
tx_s = f"{trans_v['x']:.2f} px" if trans_v.get("x") is not None else "N/A"
ty_s = f"{trans_v['y']:.2f} px" if trans_v.get("y") is not None else "N/A"
sanity_s = "VALID" if sanity is True else ("INVALID" if sanity is False else "N/A")

tv1, tv2, tv3, tv4, tv5 = st.columns(5)
tv1.markdown(f"Scale Factor: **{scale_s}**")
tv2.markdown(f"Rotation: **{rot_s}**")
tv3.markdown(f"Translation X: **{tx_s}**")
tv4.markdown(f"Translation Y: **{ty_s}**")
tv5.markdown(f"Sanity: **{sanity_s}**")

st.markdown("---")

# ---------------- Diagnostics copy ----------------
with st.expander("📋 Copy Diagnostics (JSON)"):
    st.code(json.dumps({"job_id": payload.get("jobId"), **res}, indent=2, default=str), language="json")

st.markdown("---")

# ---------------- Explainability gallery ----------------
st.markdown('<div style="font-family:var(--font-heading);font-size:1.1rem;font-weight:600;">Explainable Matching &amp; Registration Assets</div>', unsafe_allow_html=True)
st.caption("Each diagnostic view can be opened full-size and stepped through the set.")

available_keys = [p["key"] for p in VIZ_PAGES if viz_paths.get(p["key"]) and os.path.exists(str(viz_paths.get(p["key"])))]

if "viz_selected" not in st.session_state:
    st.session_state["viz_selected"] = available_keys[0] if available_keys else None

# Thumbnail hub
hub_cols = st.columns(len(VIZ_PAGES))
for i, page in enumerate(VIZ_PAGES):
    path = viz_paths.get(page["key"])
    with hub_cols[i]:
        if path and os.path.exists(str(path)):
            st.image(str(path), use_container_width=True)
            if st.button(page["label"], key=f"hub_{page['key']}", use_container_width=True):
                st.session_state["viz_selected"] = page["key"]
        else:
            st.markdown(
                f"""<div class="codexa-card" style="opacity:.45;text-align:center;padding:1.4rem .6rem;">
                <div style="font-size:.72rem;color:var(--text-faint);">Not available for this run</div></div>""",
                unsafe_allow_html=True,
            )
        st.markdown(f'<div class="viz-hub-name">{page["label"]}</div><div class="viz-hub-desc">{page["short"]}</div>', unsafe_allow_html=True)

st.write("")

# Detail viewer with pager (mirrors results-<view>.html)
sel_key = st.session_state.get("viz_selected")
if sel_key:
    idx = next(i for i, p in enumerate(VIZ_PAGES) if p["key"] == sel_key)
    this_page = VIZ_PAGES[idx]
    prev_page = VIZ_PAGES[(idx - 1) % len(VIZ_PAGES)]
    next_page = VIZ_PAGES[(idx + 1) % len(VIZ_PAGES)]

    st.markdown("---")
    d1, d2, d3 = st.columns([2, 1, 1])
    with d1:
        st.markdown(f'<div class="banner-title" style="font-size:1.1rem;">{this_page["label"]}</div>', unsafe_allow_html=True)
        st.caption(this_page["short"])
    with d2:
        if st.button(f"← {prev_page['label']}", use_container_width=True):
            st.session_state["viz_selected"] = prev_page["key"]
            st.rerun()
    with d3:
        if st.button(f"{next_page['label']} →", use_container_width=True):
            st.session_state["viz_selected"] = next_page["key"]
            st.rerun()

    path = viz_paths.get(sel_key)
    if path and os.path.exists(str(path)):
        st.image(str(path), use_container_width=True)
        with st.expander("🔍 View full resolution"):
            st.image(str(path))
    else:
        st.info("This asset was not returned for the current run.")