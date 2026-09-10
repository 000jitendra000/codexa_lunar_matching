"""
experiments/inspect_dataset.py

Milestone 2A inspection workflow.
Demonstrates: metadata -> image paths -> ImageLoader -> ImagePair -> summary.
Gracefully reports missing real lunar image files without crashing.
"""

import os
import sys
import json
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.image_loader import ImageLoader
from src.preprocessing.image_pair import ImagePair

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

META_PATH = os.path.join(PROJECT_ROOT, "data", "dataset_meta.json")


def resolve_path(file_path: str) -> str:
    if os.path.isabs(file_path):
        return file_path
    return os.path.join(PROJECT_ROOT, file_path)


def inspect_image(image_id: str, img_meta: dict):
    file_path = resolve_path(img_meta.get("file_path", ""))
    logger.info("---")
    logger.info(f"  Image ID : {image_id}")
    logger.info(f"  Mission  : {img_meta.get('mission', 'unknown')}")
    logger.info(f"  Sensor   : {img_meta.get('sensor', 'unknown')}")
    logger.info(f"  Path     : {file_path}")
    try:
        info = ImageLoader.load_image(file_path)
        logger.info(f"  Size     : {info['width']} x {info['height']} px")
        logger.info(f"  Channels : {info['channels']}")
        return info
    except FileNotFoundError:
        logger.warning(
            f"  [MISSING] File not found: '{file_path}'. "
            "Place real lunar imagery in data/raw/ to load it."
        )
        return None
    except ValueError as exc:
        logger.error(f"  [CORRUPT] Could not read image: {exc}")
        return None


def inspect_dataset(meta_path: str = META_PATH) -> None:
    logger.info("=" * 60)
    logger.info("Dataset Inspection Workflow — Milestone 2A")
    logger.info("=" * 60)

    if not os.path.exists(meta_path):
        logger.error(f"Metadata file not found: {meta_path}")
        return

    with open(meta_path, "r") as f:
        try:
            meta = json.load(f)
        except json.JSONDecodeError as exc:
            logger.error(f"Failed to parse metadata JSON: {exc}")
            return

    images = meta.get("images", [])
    pairs = meta.get("pairs", [])
    logger.info(f"Metadata loaded: {len(images)} image(s), {len(pairs)} pair(s)")

    logger.info("\n--- Individual Image Inspection ---")
    loaded_infos = {}
    for img_entry in images:
        img_id = img_entry.get("image_id", "<no id>")
        result = inspect_image(img_id, img_entry)
        if result is not None:
            loaded_infos[img_id] = result

    logger.info("\n--- Image Pair Inspection ---")
    if not pairs:
        logger.info("No pairs defined in metadata.")
    else:
        for pair_entry in pairs:
            src_id = pair_entry.get("source_id")
            ref_id = pair_entry.get("reference_id")
            label = pair_entry.get("label", "unknown")
            logger.info(f"Pair: {src_id} -> {ref_id}  [label={label}]")
            src_info = loaded_infos.get(src_id)
            ref_info = loaded_infos.get(ref_id)
            if src_info and ref_info:
                pair = ImagePair(src_info, ref_info, label=label)
                logger.info(f"  ImagePair created: {pair.summary()}")
            else:
                missing = [x for x, v in {src_id: src_info, ref_id: ref_info}.items() if v is None]
                logger.warning(
                    f"  ImagePair NOT constructed. Missing file(s): {missing}. "
                    "Add real lunar images to data/raw/ to complete pair creation."
                )

    logger.info("\n" + "=" * 60)
    logger.info("Inspection complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    inspect_dataset()
