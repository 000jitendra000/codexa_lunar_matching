import os, sys, json, logging
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.default import Config
from src.preprocessing.image_loader import ImageLoader
from src.preprocessing.pipeline import PreprocessingPipeline
from src.matching.classical_baseline import run_classical_baseline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "classical_baseline")
META_PATH = os.path.join(PROJECT_ROOT, "data", "dataset_meta.json")


def _synthetic_pair():
    h, w, block = 512, 512, 40
    board = np.zeros((h, w), dtype=np.uint8)
    for r in range(0, h, block):
        for c in range(0, w, block):
            if (r // block + c // block) % 2 == 0:
                board[r:r+block, c:c+block] = 210
    rng = np.random.default_rng(42)
    noise = rng.integers(0, 25, (h, w), dtype=np.uint8)
    img_a = np.clip(board.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 12.0, 0.92)
    M[0, 2] += 15; M[1, 2] += 8
    img_b = cv2.warpAffine(img_a, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return img_a, img_b


def _save(img, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    if img.dtype == np.float32:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(path, img)
    return path


def _draw_kp(img, kps, name):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    vis = cv2.drawKeypoints(vis, kps, None, color=(0, 200, 0),
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    _save(vis, name)


def _draw_matches_img(ia, ka, ib, kb, matches, name, mx=60):
    vis = cv2.drawMatches(ia, ka, ib, kb, matches[:mx], None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    _save(vis, name)


def _draw_warp(ia, ib, transform, model, name):
    if transform is None:
        return
    h, w = ib.shape[:2]
    warped = cv2.warpAffine(ia, transform, (w, h)) if model == "affine" else cv2.warpPerspective(ia, transform, (w, h))
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[:, :, 1] = warped
    overlay[:, :, 2] = ib
    _save(overlay, name)


def run_exp(raw_a, raw_b, label):
    logger.info("=" * 60)
    logger.info("Classical Baseline Experiment: %s", label)
    cfg_pp = dict(Config.PREPROCESSING); cfg_pp["pyramid_enabled"] = False
    pp = PreprocessingPipeline(cfg_pp)
    img_a = pp.process(raw_a)["image"]; img_b = pp.process(raw_b)["image"]
    if img_a.dtype != np.uint8:
        img_a = (np.clip(img_a, 0, 1) * 255).astype(np.uint8)
    if img_b.dtype != np.uint8:
        img_b = (np.clip(img_b, 0, 1) * 255).astype(np.uint8)
    cfg = Config.CLASSICAL_MATCHING
    for mth in ("sift", "akaze"):
        logger.info("-- Running %s --", mth.upper())
        res = run_classical_baseline(img_a, img_b, cfg, method=mth)
        m = res["metrics"]
        logger.info("  kp_a=%d kp_b=%d candidates=%d inliers=%d ratio=%.2f rmse=%s status=%s elapsed=%ss",
                    m["keypoints_a"], m["keypoints_b"], m["candidate_matches"],
                    m["inlier_count"], m["inlier_ratio"],
                    m.get("reprojection_rmse_px"), m["geometric_status"], m.get("elapsed_sec"))
        pfx = label + "_" + mth
        model = res.get("geo_result", {}).get("model", "affine")
        if res["keypoints_a"]: _draw_kp(img_a, res["keypoints_a"], pfx + "_kp_a.png")
        if res["keypoints_b"]: _draw_kp(img_b, res["keypoints_b"], pfx + "_kp_b.png")
        if res["matches"]: _draw_matches_img(img_a, res["keypoints_a"], img_b, res["keypoints_b"], res["matches"], pfx + "_raw_matches.png")
        if res["inlier_matches"]: _draw_matches_img(img_a, res["keypoints_a"], img_b, res["keypoints_b"], res["inlier_matches"], pfx + "_inliers.png")
        _draw_warp(img_a, img_b, res["transform"], model, pfx + "_warp.png")
    logger.info("Outputs saved to: %s", OUT_DIR)


def main():
    used_real = False
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
        img_map = {x["image_id"]: x for x in meta.get("images", [])}
        for pair in meta.get("pairs", []):
            sm = img_map.get(pair.get("source_id"), {}); rm = img_map.get(pair.get("reference_id"), {})
            sp = sm.get("file_path", ""); rp = rm.get("file_path", "")
            sa = sp if os.path.isabs(sp) else os.path.join(PROJECT_ROOT, sp)
            ra = rp if os.path.isabs(rp) else os.path.join(PROJECT_ROOT, rp)
            try:
                ia = ImageLoader.load_image(sa)["image"]; ib = ImageLoader.load_image(ra)["image"]
                run_exp(ia, ib, label=pair["source_id"] + "_vs_" + pair["reference_id"])
                used_real = True; break
            except FileNotFoundError as exc:
                logger.warning("[MISSING] %s -- Place real imagery in data/raw/.", exc)
    if not used_real:
        logger.info("No real lunar images found. Using SYNTHETIC test pair (software verification only).")
        ia, ib = _synthetic_pair()
        run_exp(ia, ib, "synthetic")


if __name__ == "__main__":
    main()
