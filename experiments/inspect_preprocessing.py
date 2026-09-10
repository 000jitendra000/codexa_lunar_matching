import os, sys, json, logging
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.default import Config
from src.preprocessing.image_loader import ImageLoader
from src.preprocessing.normalize import normalize
from src.preprocessing.clahe import apply_clahe
from src.preprocessing.denoise import denoise
from src.preprocessing.pyramid import build_pyramid
from src.preprocessing.pipeline import _to_grayscale

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "inspect_preprocessing")
META_PATH = os.path.join(PROJECT_ROOT, "data", "dataset_meta.json")


def _save(img, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    save_img = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.dtype == np.float32 else img
    cv2.imwrite(path, save_img)
    logger.info("Saved %s  shape=%s", path, img.shape)


def _synthetic(h=256, w=256):
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, w, dtype=np.float32)
    y = np.linspace(0, 1, h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    base = 0.3 * xx + 0.4 * yy
    base += rng.normal(0, 0.05, (h, w)).astype(np.float32)
    for _ in range(6):
        cx, cy = rng.integers(30, w-30), rng.integers(30, h-30)
        s = rng.integers(10, 30)
        base -= 0.2 * np.exp(-((xx - cx/w)**2 + (yy - cy/h)**2) / (2*(s/w)**2))
    return (np.clip(base, 0, 1) * 255).astype(np.uint8)


def run_inspection(image, label):
    logger.info("=" * 60)
    logger.info("Inspecting: %s  shape=%s  dtype=%s", label, image.shape, image.dtype)
    _save(image, f"{label}_00_raw.png")
    gray = _to_grayscale(image)
    _save(gray, f"{label}_01_gray.png")
    norm = normalize(gray, method="minmax")
    _save(norm, f"{label}_02_normalized.png")
    clahe_r = apply_clahe(gray, clip_limit=2.0)
    _save(clahe_r, f"{label}_03_clahe.png")
    den = denoise(gray, method="gaussian", ksize=3)
    _save(den, f"{label}_04_denoised.png")
    pyr = build_pyramid(clahe_r, num_levels=4)
    for i, lvl in enumerate(pyr):
        _save(lvl, f"{label}_05_pyramid_L{i}.png")
    logger.info("Done. Outputs in: %s", OUTPUT_DIR)


def main():
    logger.info("Phase 3 Preprocessing Inspection")
    used_real = False
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
        for img_entry in meta.get("images", []):
            img_id = img_entry.get("image_id", "unknown")
            fp = img_entry.get("file_path", "")
            abs_p = fp if os.path.isabs(fp) else os.path.join(PROJECT_ROOT, fp)
            try:
                info = ImageLoader.load_image(abs_p)
                logger.info("Using real image: %s", img_id)
                run_inspection(info["image"], label=img_id)
                used_real = True
                break
            except FileNotFoundError:
                logger.warning("[MISSING] %s not found at %s. Place real imagery in data/raw/.", img_id, abs_p)
    if not used_real:
        logger.info("No real images found. Using SYNTHETIC test image (software verification only).")
        run_inspection(_synthetic(), label="synthetic_test")

if __name__ == "__main__":
    main()
