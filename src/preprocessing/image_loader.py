import os
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ImageLoader:
    """
    A class to safely load and inspect lunar images.
    """
    
    @staticmethod
    def load_image(file_path: str):
        """
        Loads an image from the provided file path.
        
        Args:
            file_path (str): The absolute or relative path to the image.
            
        Returns:
            dict: A dictionary containing the image array, width, height, and channels.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is corrupted or cannot be read by OpenCV.
        """
        if not os.path.exists(file_path):
            logger.error(f"Image not found at path: {file_path}")
            raise FileNotFoundError(f"Image not found at path: {file_path}")
            
        # Read the image (unchanged format to support e.g. 16-bit TIFFs in the future)
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            logger.error(f"Image could not be read or is corrupted: {file_path}")
            raise ValueError(f"Failed to read image at path: {file_path}")
            
        # Determine image properties
        if len(img.shape) == 2:
            height, width = img.shape
            channels = 1
        elif len(img.shape) == 3:
            height, width, channels = img.shape
        else:
            logger.error(f"Unexpected image shape {img.shape} for path: {file_path}")
            raise ValueError(f"Unexpected image shape: {img.shape}")
            
        return {
            "image": img,
            "width": width,
            "height": height,
            "channels": channels
        }
