"""
pages/1_Match_Engine.py

Codexa Streamlit app — Match Engine page.
Upload Image A (Reference) & Image B (Query), execute HybridMatcher + Registration,
and view live execution progress before jumping to results.
"""

import gc
import logging
import os
import threading
import time
import numpy as np
import streamlit as st

from codexa_theme import (
    inject_css, render_topnav, render_session_badge, init_session_state,
    bump_stats, RESULT_KEY, VIZ_KEY
)
from src.matching.hybrid_matcher import HybridMatcher, HybridMatchResult
from src.registration.registration_engine import RegistrationEngine, RegistrationResult
from src.visualization.match_visualizer import MatchVisualizer, VisualizationResult

logger = logging.getLogger(__name__)


@st.cache_resource
def get_matcher() -> HybridMatcher:
    return HybridMatcher()


def decode_uploaded_image(uploaded_file) -> np.ndarray:
    if uploaded_file is None:
        return None
    try:
        from PIL import Image
        img_obj = Image.open(uploaded_file).convert("L")
        img = np.array(img_obj, dtype=np.uint8)
        uploaded_file.seek(0)
        if img is None or img.size == 0:
            return None
        return img
    except Exception:
        try:
            import cv2
            file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            uploaded_file.seek(0)
            return img
        except Exception as err:
            logger.error("Failed to decode uploaded image: %s", err)
            return None


def format_match_result(hybrid_res: HybridMatchResult, reg_res: RegistrationResult = None):
    acc = hybrid_res.acceptance
    checks_list = []
    if acc and acc.checks:
        for name, check_data in acc.checks.items():
            if isinstance(check_data, dict):
                val = check_data.get("value")
                thresh = check_data.get("threshold", "N/A")
                is_passed = bool(check_data.get("passed", False))
            else:
                val = check_data
                thresh = "N/A"
                is_passed = bool(check_data)
            checks_list.append({
                "Criterion": name.replace("_", " ").title(),
                "Status": "PASS" if is_passed else "FAIL",
                "Measured Value": f"{val:.4f}" if isinstance(val, float) else str(val),
                "Threshold": f"{thresh:.4f}" if isinstance(thresh, float) else str(thresh),
            })

    transform_data = None
    if hybrid_res.transform is not None:
        transform_data = {
            "scale": float(hybrid_res.transform.scale),
            "rotation_deg": float(hybrid_res.transform.rotation_deg),
            "translation_x": float(hybrid_res.transform.translation_x),
            "translation_y": float(hybrid_res.transform.translation_y),
        }

    return {
        "matched": bool(hybrid_res.matched),
        "acceptance_status": acc.status if acc else ("ACCEPTED" if hybrid_res.matched else "REJECTED"),
        "acceptance_score": float(acc.acceptance_score) if acc else (1.0 if hybrid_res.matched else 0.0),
        "acceptance_reason": acc.reason if acc else "Match criteria evaluated",
        "confidence": float(reg_res.confidence) if reg_res else float(hybrid_res.confidence),
        "inliers": int(hybrid_res.num_inliers),
        "correspondences": int(hybrid_res.num_correspondences),
        "inlier_ratio": float(hybrid_res.inlier_ratio),
        "rmse": float(hybrid_res.rmse),
        "spatial_coverage": float(reg_res.coverage) if reg_res else 0.0,
        "quality_category": str(reg_res.quality) if reg_res else "N/A",
        "checks": checks_list,
        "transform": transform_data,
    }


st.set_page_config(page_title="Match Engine | Codexa", page_icon="🌕", layout="wide")
init_session_state()
inject_css()
render_topnav("Match Engine")

top_l, top_r = st.columns([3, 1])
with top_l:
    st.markdown('<h1 class="codexa-heading">Input Lunar Image Pair</h1>', unsafe_allow_html=True)
    st.caption("Upload a reference and a query frame to test for a location match")
with top_r:
    render_session_badge()

st.markdown("---")

st.markdown('<h3 class="codexa-heading">1. Select Orbital Image Pair</h3>', unsafe_allow_html=True)

col_a, col_b = st.columns(2, gap="medium")

with col_a:
    file_a = st.file_uploader(
        "Upload Reference Image A",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="uploader_a",
        help="Select reference lunar image (e.g. Chandrayaan-2 OHRC)",
    )
    if file_a is not None:
        st.image(file_a, caption="Image A Preview", use_container_width=True)

with col_b:
    file_b = st.file_uploader(
        "Upload Query Image B",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="uploader_b",
        help="Select target lunar image to align (e.g. LRO LROC NAC)",
    )
    if file_b is not None:
        st.image(file_b, caption="Image B Preview", use_container_width=True)

both_uploaded = file_a is not None and file_b is not None

st.write("")
match_button = st.button(
    "🚀 Execute Location Match",
    type="primary",
    disabled=not both_uploaded,
    use_container_width=True,
)

if "executing_match" not in st.session_state:
    st.session_state["executing_match"] = False

if match_button and both_uploaded:
    st.session_state["executing_match"] = True

if st.session_state["executing_match"] and both_uploaded:
    with st.status("Running Cross-Sensor Lunar Location Engine...", expanded=True) as status:
        try:
            st.write("📷 Decoding image buffers...")
            img_a = decode_uploaded_image(file_a)
            img_b = decode_uploaded_image(file_b)

            if img_a is None or img_b is None:
                st.error("Failed to decode images. Please ensure valid PNG, JPG, or TIFF files.")
                status.update(label="Decoding Error", state="error", expanded=True)
            else:
                st.write("🧠 Executing Hybrid Matcher (LoFTR + Crater Graph + RANSAC)...")
                matcher = get_matcher()

                t0 = time.time()
                hybrid_res: HybridMatchResult = matcher.match(img_a, img_b)
                st.write(f"✓ Feature matching finished in {time.time() - t0:.2f}s ({hybrid_res.num_inliers} inliers)")

                reg_res: RegistrationResult = None
                if hybrid_res.matched and hybrid_res.transform is not None:
                    st.write("📐 Executing Sub-Pixel Registration Engine...")
                    reg_engine = RegistrationEngine()
                    try:
                        reg_res = reg_engine.register(img_a, img_b, hybrid_res)
                        st.write(f"✓ Registration Quality: {reg_res.quality.value}")
                    except Exception as reg_err:
                        logger.warning("Registration failed: %s", reg_err)

                st.write("🎨 Generating Diagnostic Visualizations...")
                job_id = f"job_{int(time.time())}"
                visualizer = MatchVisualizer()
                viz_res: VisualizationResult = visualizer.generate(
                    image_a=img_a,
                    image_b=img_b,
                    match_result=hybrid_res,
                    registration_result=reg_res,
                    job_id=job_id,
                )

                def safe_float(val, default=0.0):
                    return float(val) if val is not None else default

                def safe_int(val, default=0):
                    return int(val) if val is not None else default

                # Format acceptance checks
                acc_dict = None
                if hybrid_res.acceptance:
                    acc_dict = hybrid_res.acceptance.to_dict()

                # Format transform
                tf_dict = None
                if hybrid_res.transform:
                    tf_dict = {
                        "scale": safe_float(hybrid_res.transform.scale, 1.0),
                        "rotation_deg": safe_float(hybrid_res.transform.rotation_deg, 0.0),
                        "translation": {
                            "x": safe_float(hybrid_res.transform.translation_x, 0.0),
                            "y": safe_float(hybrid_res.transform.translation_y, 0.0),
                        },
                    }

                res_dict = {
                    "matched": bool(hybrid_res.matched),
                    "acceptance": acc_dict,
                    "inliers": safe_int(hybrid_res.num_inliers),
                    "correspondences": safe_int(hybrid_res.num_correspondences),
                    "inlier_ratio": safe_float(hybrid_res.inlier_ratio),
                    "rmse": safe_float(hybrid_res.rmse, 0.0),
                    "confidence": safe_float(hybrid_res.confidence, 0.0),
                    "coverage": safe_float(reg_res.coverage) if reg_res else 0.0,
                    "quality": str(reg_res.quality) if (reg_res and reg_res.quality is not None) else "N/A",
                    "transform": tf_dict,
                    "failure_reason": hybrid_res.acceptance.reason if hybrid_res.acceptance else None,
                }

                viz_map = {
                    "correspondence": viz_res.correspondence_image,
                    "overlay": viz_res.registration_overlay,
                    "checkerboard": viz_res.checkerboard,
                    "cmap_a": viz_res.confidence_map_a,
                    "cmap_b": viz_res.confidence_map_b,
                }

                st.session_state[RESULT_KEY] = {"jobId": job_id, "result": res_dict}
                st.session_state[VIZ_KEY] = viz_map

                bump_stats(bool(hybrid_res.matched))
                status.update(label="Matching Completed Successfully!", state="complete", expanded=False)
                st.success("Matching complete! Switching to results view...")
                time.sleep(0.5)
                st.switch_page("pages/2_Results.py")

        except Exception as exc:
            logger.error("Error during Streamlit match execution: %s", exc, exc_info=True)
            st.error(f"An error occurred during matching: {exc}")
            status.update(label="Execution Error", state="error", expanded=True)

        finally:
            st.session_state["executing_match"] = False
            if "img_a" in locals():
                del img_a
            if "img_b" in locals():
                del img_b
            gc.collect()