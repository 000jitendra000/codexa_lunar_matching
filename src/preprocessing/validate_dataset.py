import os
import json
import logging
import sys

# Ensure src is in the path if running locally
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.preprocessing.image_loader import ImageLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_dataset(meta_path="data/dataset_meta.json"):
    """
    Validates dataset metadata and verifies files.
    """
    if not os.path.exists(meta_path):
        logger.error(f"Metadata file missing at {meta_path}")
        return False
        
    with open(meta_path, 'r') as f:
        try:
            meta = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse metadata JSON: {e}")
            return False

    images = meta.get("images", [])
    pairs = meta.get("pairs", [])
    
    logger.info(f"Validating {len(images)} images and {len(pairs)} pairs from '{meta_path}'...")
    
    image_map = {}
    validation_passed = True
    
    # 1. Validate images
    for img_info in images:
        img_id = img_info.get("image_id")
        
        if img_id in image_map:
            logger.error(f"Duplicate image_id found: {img_id}")
            validation_passed = False
            
        file_path = img_info.get("file_path")
        
        try:
            loaded = ImageLoader.load_image(file_path)
            logger.info(f"Validated image {img_id}: {loaded['width']}x{loaded['height']} ({loaded['channels']} ch) successfully loaded.")
        except FileNotFoundError:
            logger.error(f"Missing image file for '{img_id}' at '{file_path}'. Real lunar imagery likely needs to be downloaded to data/raw/.")
            validation_passed = False
        except Exception as e:
            logger.error(f"Image validation failed for '{img_id}' at '{file_path}': {e}")
            validation_passed = False
            
        image_map[img_id] = img_info
        
    # 2. Validate pairs
    logger.info("Validating pair relationships...")
    for pair in pairs:
        src_id = pair.get("source_id")
        ref_id = pair.get("reference_id")
        
        if src_id not in image_map:
            logger.error(f"Pair source_id '{src_id}' not found in images metadata.")
            validation_passed = False
            
        if ref_id not in image_map:
            logger.error(f"Pair reference_id '{ref_id}' not found in images metadata.")
            validation_passed = False
            
    if validation_passed:
        logger.info("Dataset validation COMPLETE and PASSED.")
    else:
        logger.error("Dataset validation COMPLETE with ERRORS.")
        
    return validation_passed

if __name__ == "__main__":
    success = validate_dataset()
    sys.exit(0 if success else 1)
