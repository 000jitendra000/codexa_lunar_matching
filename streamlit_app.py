"""
streamlit_app.py

Streamlit Demonstration Interface for Cross-Sensor Lunar Location Matching.
Imports and directly invokes the model matching engine (HybridMatcher) using
@st.cache_resource for single-instance model caching and a process-level lock
for multi-user memory safety.
"""

import gc
import logging
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

from configs.default import Config
from src.matching.hybrid_matcher import HybridMatcher, HybridMatchResult
from src.registration.registration_engine import RegistrationEngine, RegistrationResult
from src.utils.memory_debug import get_rss_mb, log_memory_stage
from src.visualization.match_visualizer import MatchVisualizer, VisualizationResult

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Enforce single-concurrency lock process-wide to prevent multi-user memory spikes
EXECUTION_LOCK = threading.Lock()


@st.cache_resource
def get_matcher() -> HybridMatcher:
    """Instantiate and cache a single process-wide HybridMatcher instance.

    Uses @st.cache_resource so PyTorch and LoFTR model weights (~690 MB RSS)
    are initialized exactly once across Streamlit reruns.
    """
    logger.info("Initializing cached HybridMatcher instance for Streamlit...")
    return HybridMatcher()


def decode_uploaded_image(uploaded_file) -> Optional[np.ndarray]:
    """Decode uploaded image bytes into a 2D uint8 NumPy array."""
    if uploaded_file is None:
        return None
    try:
        file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        uploaded_file.seek(0)
        if img is None or img.size == 0:
            return None
        return img
    except Exception as err:
        logger.error("Failed to decode uploaded image: %s", err)
        return None


def format_match_result(
    hybrid_res: HybridMatchResult, reg_res: Optional[RegistrationResult]
) -> Dict[str, Any]:
    """Format matching and registration results into a lightweight serializable dictionary."""
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

            val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
            thresh_str = f"{thresh:.4f}" if isinstance(thresh, float) else str(thresh)

            checks_list.append(
                {
                    "Criterion": name.replace("_", " ").title(),
                    "Status": "PASS" if is_passed else "FAIL",
                    "Measured Value": val_str,
                    "Threshold": thresh_str,
                }
            )

    transform_data = None
    if hybrid_res.transform is not None:
        transform_data = {
            "scale": float(hybrid_res.transform.scale),
            "rotation_deg": float(hybrid_res.transform.rotation_deg),
            "translation_x": float(hybrid_res.transform.translation_x),
            "translation_y": float(hybrid_res.transform.translation_y),
        }

    quality_category = "N/A"
    confidence_val = float(hybrid_res.confidence)
    if reg_res is not None:
        quality_category = str(reg_res.quality)
        confidence_val = float(reg_res.confidence)

    return {
        "matched": bool(hybrid_res.matched),
        "acceptance_status": acc.status if acc else ("ACCEPTED" if hybrid_res.matched else "REJECTED"),
        "acceptance_score": float(acc.acceptance_score) if acc else (1.0 if hybrid_res.matched else 0.0),
        "acceptance_reason": acc.reason if acc else ("Match criteria evaluated"),
        "confidence": confidence_val,
        "inliers": int(hybrid_res.num_inliers),
        "correspondences": int(hybrid_res.num_correspondences),
        "inlier_ratio": float(hybrid_res.inlier_ratio),
        "rmse": float(hybrid_res.rmse),
        "spatial_coverage": float(reg_res.coverage) if reg_res else 0.0,
        "quality_category": quality_category,
        "checks": checks_list,
        "transform": transform_data,
    }


def main():
    st.set_page_config(
        page_title="Cross-Sensor Lunar Location Matching",
        page_icon="🌕",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Application Header
    st.title("🌕 Cross-Sensor Lunar Location Matching Engine")
    st.caption(
        "Automated geometric co-registration and location verification for lunar orbital imagery "
        "(e.g., Chandrayaan-2 OHRC vs. LRO LROC NAC)."
    )

    # Explanation for Judges / Users
    with st.expander("ℹ️ How does this system work?", expanded=False):
        st.markdown(
            """
            This platform determines whether two lunar surface images represent the **exact same physical location** 
            despite drastic differences in resolution (GSD), scale, illumination, sensor orientation, and contrast:
            
            1. **Crater Geometry Graph**: Extracts invariant 3-clique triangles and local crater constellations to build scale- and rotation-robust descriptors.
            2. **Learned LoFTR Transformer Branch**: Extracts dense feature correspondences across texture-rich and complex terrain.
            3. **Correspondence Fusion**: Merges classical crater landmarks and learned correspondences with spatial duplicate resolution and branch balancing.
            4. **RANSAC Geometric Consensus**: Estimates a 4-DOF planar similarity transformation ($p_B \\approx s R(\\theta) p_A + t$) while rejecting outliers.
            5. **Sub-Pixel Image Registration**: Refines tie-points via parabolic peak interpolation and performs pull-based image warping.
            6. **Match Acceptance Engine**: Decouples geometric consensus from location verification using 6 hard evidence criteria (inliers, inlier ratio, RMSE, coverage, confidence, sanity).
            7. **Visualization Suite**: Generates confidence maps, correspondence vectors, checkerboard overlays, and registration alignment.
            """
        )

    # Sidebar Options
    st.sidebar.header("⚙️ Matching Configuration")
    st.sidebar.info(
        "Core algorithms execute with production defaults (LoFTR + Crater Fusion + RANSAC + Match Acceptance)."
    )
    if os.environ.get("LUNAR_MEMORY_DEBUG") == "1":
        st.sidebar.warning(f"Memory Debug Active | RSS: {get_rss_mb():.1f} MB")

    # Image Upload Section
    st.subheader("1. Select Orbital Image Pair")
    col_a, col_b = st.columns(2)

    with col_a:
        file_a = st.file_uploader(
            "Upload Image A (Base / Reference)",
            type=["png", "jpg", "jpeg"],
            key="uploader_a",
            help="Select reference lunar image (e.g. Chandrayaan-2 OHRC)",
        )
        if file_a is not None:
            st.image(file_a, caption="Image A Preview", use_container_width=True)

    with col_b:
        file_b = st.file_uploader(
            "Upload Image B (Search / Target)",
            type=["png", "jpg", "jpeg"],
            key="uploader_b",
            help="Select target lunar image to align (e.g. LRO LROC NAC)",
        )
        if file_b is not None:
            st.image(file_b, caption="Image B Preview", use_container_width=True)

    # Match Execution Button
    both_uploaded = file_a is not None and file_b is not None
    match_button = st.button(
        "🚀 Match Lunar Images",
        type="primary",
        disabled=not both_uploaded,
        use_container_width=True,
    )

    if not both_uploaded:
        st.info("Please upload both Image A and Image B above to execute matching.")

    # Execution Flow
    if match_button and both_uploaded:
        log_memory_stage("process_before_match")
        st.session_state.pop("match_result", None)
        st.session_state.pop("viz_paths", None)

        with st.status("Processing Cross-Sensor Lunar Matching...", expanded=True) as status:
            st.write("🔒 Acquiring execution lock...")
            acquired = EXECUTION_LOCK.acquire(blocking=True)

            try:
                st.write("📷 Decoding image buffers...")
                img_a = decode_uploaded_image(file_a)
                img_b = decode_uploaded_image(file_b)

                if img_a is None or img_b is None:
                    st.error("Error decoding one or both uploaded image files. Please upload valid image files.")
                    status.update(label="Decoding Failed", state="error", expanded=True)
                    return

                log_memory_stage("rss_after_image_loading")

                st.write("🧠 Executing Hybrid Matcher (LoFTR + Crater Graph + RANSAC)...")
                matcher = get_matcher()
                
                start_time = time.time()
                hybrid_res: HybridMatchResult = matcher.match(img_a, img_b)
                st.write(f"✓ Matching finished in {time.time() - start_time:.2f}s (Inliers: {hybrid_res.num_inliers})")
                log_memory_stage("rss_after_loftr")

                # Registration & Quality Evaluation
                reg_res: Optional[RegistrationResult] = None
                if hybrid_res.matched and hybrid_res.transform is not None:
                    st.write("📐 Executing Sub-Pixel Registration Engine...")
                    reg_engine = RegistrationEngine()
                    try:
                        reg_res = reg_engine.register(img_a, img_b, hybrid_res)
                        st.write(f"✓ Registration Quality: {reg_res.quality.value}")
                    except Exception as reg_err:
                        logger.warning("Registration failed: %s", reg_err)
                log_memory_stage("rss_after_registration")

                # Visualization Generation
                st.write("🎨 Generating Visualization Assets...")
                job_id = f"streamlit_{int(time.time())}"
                visualizer = MatchVisualizer()
                viz_res: VisualizationResult = visualizer.generate(
                    image_a=img_a,
                    image_b=img_b,
                    match_result=hybrid_res,
                    registration_result=reg_res,
                    job_id=job_id,
                )
                log_memory_stage("rss_after_visualization")

                # Store lightweight formatted metadata in session state
                result_dict = format_match_result(hybrid_res, reg_res)
                st.session_state["match_result"] = result_dict
                st.session_state["viz_paths"] = {
                    "correspondence": viz_res.correspondence_image,
                    "overlay": viz_res.registration_overlay,
                    "checkerboard": viz_res.checkerboard,
                    "conf_a": viz_res.confidence_map_a,
                    "conf_b": viz_res.confidence_map_b,
                }

                status.update(label="Matching Complete!", state="complete", expanded=False)

            except Exception as exc:
                logger.error("Error during Streamlit match execution: %s", exc, exc_info=True)
                st.error(f"An unexpected error occurred during matching: {exc}")
                status.update(label="Execution Error", state="error", expanded=True)

            finally:
                # Immediate array deallocation & RAM garbage collection
                if 'img_a' in locals():
                    del img_a
                if 'img_b' in locals():
                    del img_b
                gc.collect()
                log_memory_stage("rss_after_complete_job_cleanup")
                EXECUTION_LOCK.release()

    # Render Results if present in session_state
    if "match_result" in st.session_state:
        res = st.session_state["match_result"]
        viz_paths = st.session_state.get("viz_paths", {})

        st.markdown("---")
        st.subheader("2. Location Match Decision")

        # Primary Banner
        if res["matched"]:
            st.success(
                f"### 🟢 LOCATION MATCH ACCEPTED\n"
                f"**Status:** `{res['acceptance_status']}` | **Acceptance Score:** `{res['acceptance_score']:.2f}`"
            )
        else:
            st.error(
                f"### 🔴 LOCATION NOT MATCHED\n"
                f"**Status:** `{res['acceptance_status']}` | **Reason:** {res['acceptance_reason']}"
            )

        # Metrics Dashboard
        st.markdown("#### 📊 Metric Overview")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Match Status", "ACCEPTED" if res["matched"] else "REJECTED")
        m2.metric("Acceptance Score", f"{res['acceptance_score']:.2f}")
        m3.metric("Confidence", f"{res['confidence'] * 100:.1f}%")
        m4.metric("Inliers / Total", f"{res['inliers']} / {res['correspondences']}")
        m5.metric("Inlier Ratio", f"{res['inlier_ratio'] * 100:.1f}%")
        m6.metric("RMSE", f"{res['rmse']:.2f} px")

        # Match Acceptance Details
        with st.expander("🔍 Match Acceptance Details", expanded=not res["matched"]):
            st.write(f"**Diagnostic Reason:** {res['acceptance_reason']}")
            if res["checks"]:
                st.table(res["checks"])
            else:
                st.write("No detailed criteria checks recorded.")

        # Transformation Details
        with st.expander("📐 Estimated Geometric Transformation", expanded=False):
            if res["transform"]:
                t = res["transform"]
                t1, t2, t3, t4 = st.columns(4)
                t1.metric("Scale (s)", f"{t['scale']:.4f}")
                t2.metric("Rotation (θ)", f"{t['rotation_deg']:.2f}°")
                t3.metric("Translation X", f"{t['translation_x']:.2f} px")
                t4.metric("Translation Y", f"{t['translation_y']:.2f} px")
                st.caption(
                    "Represents the 2D similarity mapping $p_B \\approx s R(\\theta) p_A + t$ "
                    "from Image A coordinate space to Image B coordinate space."
                )
            else:
                st.info("No valid geometric transformation matrix estimated.")

        # Visualization Tabs
        st.markdown("---")
        st.subheader("3. Visualization Gallery")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "Correspondences",
                "Registration Overlay",
                "Checkerboard",
                "Confidence Map A",
                "Confidence Map B",
            ]
        )

        with tab1:
            p = viz_paths.get("correspondence")
            if p and os.path.exists(p):
                st.image(p, caption="Feature Correspondences (Crater + Learned Fusion)", use_container_width=True)
            else:
                st.info("Correspondence image not available.")

        with tab2:
            p = viz_paths.get("overlay")
            if p and os.path.exists(p):
                st.image(p, caption="Registration Overlay (Blend)", use_container_width=True)
            else:
                st.info("Registration overlay image not available.")

        with tab3:
            p = viz_paths.get("checkerboard")
            if p and os.path.exists(p):
                st.image(p, caption="Checkerboard Co-Registration Inspection", use_container_width=True)
            else:
                st.info("Checkerboard image not available.")

        with tab4:
            p = viz_paths.get("conf_a")
            if p and os.path.exists(p):
                st.image(p, caption="LoFTR Confidence Map - Image A", use_container_width=True)
            else:
                st.info("Confidence map A not available.")

        with tab5:
            p = viz_paths.get("conf_b")
            if p and os.path.exists(p):
                st.image(p, caption="LoFTR Confidence Map - Image B", use_container_width=True)
            else:
                st.info("Confidence map B not available.")


if __name__ == "__main__":
    main()
