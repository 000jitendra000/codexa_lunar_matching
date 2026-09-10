import sys
import logging
import importlib

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

required_packages = [
    "torch",
    "torchvision",
    "opencv-python",  # Imports as cv2
    "numpy",
    "scipy",
    "kornia",
    "networkx",
    "matplotlib",
    "streamlit"
]

module_mapping = {
    "opencv-python": "cv2",
}

def verify_environment():
    logger.info("Starting Environment Verification...")
    
    # Check Python Version
    logger.info(f"Python Version: {sys.version.split(' ')[0]}")
    
    # Check Local Configuration
    try:
        from configs.default import Config
        logger.info(f"Loaded Configuration for: {Config.PROJECT_NAME} v{Config.VERSION}")
    except ImportError as e:
        logger.error(f"Failed to load project configuration: {e}")
        return False
        
    # Check required core dependencies
    missing_packages = []
    
    for pkg in required_packages:
        import_name = module_mapping.get(pkg, pkg)
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, '__version__', 'unknown version')
            logger.info(f"Successfully imported {import_name} (Version: {version})")
        except ImportError:
            logger.error(f"Failed to import {import_name} (Package: {pkg})")
            missing_packages.append(pkg)
            
    if missing_packages:
        logger.error(f"Environment Verification FAILED. Missing packages: {missing_packages}")
        logger.info("Please install them using: pip install -r requirements.txt")
        return False
        
    logger.info("Environment Verification SUCCESSFUL. All systems go!")
    return True

if __name__ == "__main__":
    success = verify_environment()
    sys.exit(0 if success else 1)
