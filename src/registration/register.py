"""
src/registration/register.py

Image registration and warping module for Milestone B (Registration & Quality Engine).
Aligns Image B into the coordinate system of Image A using the verified planar similarity transform.

ARCHITECTURAL RULES & WARPING CONVENTION:
- SimilarityTransform2D represents the forward coordinate mapping from Image A to Image B:
    p_B = T(p_A) = s * R(theta) * p_A + t
- In backward (pull-based) image resampling (e.g. cv2.warpAffine):
    For each destination pixel coordinate p_A in the registered image frame,
    its corresponding source intensity is sampled from p_B = T(p_A) in Image B.
    Therefore, the 2x3 affine matrix passed to cv2.warpAffine maps destination (A)
    to source (B), which is exactly T.to_matrix()[:2, :].
- Produces an aligned registered image and a binary valid_mask (255 where Image B overlap exists).
- Supports 2D single-channel and 3D multi-channel images without color distortion.
"""

from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np

from src.matching.transformation import SimilarityTransform2D


def get_affine_matrix_from_similarity(transform: SimilarityTransform2D) -> np.ndarray:
    """
    Extract the 2x3 affine transformation matrix mapping source (Image B)
    to destination (Image A) coordinates for cv2.warpAffine.
    Since transform T maps Image A -> Image B, warping Image B back into Image A's
    frame requires the inverse transform T^(-1) (mapping B -> A).
    """
    inv_tf = transform.inverse()
    m_3x3 = inv_tf.to_matrix()
    return m_3x3[:2, :].copy()


def register_image(
    image_a: np.ndarray,
    image_b: np.ndarray,
    transform: SimilarityTransform2D,
    output_shape: Optional[Tuple[int, int]] = None,
    interpolation: str = "linear",
    border_mode: str = "constant",
    border_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, SimilarityTransform2D]:
    """
    Warp Image B into the coordinate system of Image A using the similarity transform T (A -> B).

    Args:
        image_a: Reference image (H_A, W_A) or (H_A, W_A, C).
        image_b: Query image to warp (H_B, W_B) or (H_B, W_B, C).
        transform: SimilarityTransform2D mapping Image A coordinates to Image B.
        output_shape: Optional (height, width) for registered output. Defaults to image_a.shape[:2].
        interpolation: Interpolation method: 'linear', 'cubic', or 'nearest'.
        border_mode: Border extrapolation mode: 'constant', 'reflect', or 'replicate'.
        border_value: Value for border pixels outside Image B's domain (default 0.0).

    Returns:
        Tuple of:
            - registered_image: Warped Image B in Image A's coordinate system.
            - valid_mask: uint8 binary mask (255 for valid overlap pixels, 0 for outside).
            - transform_used: The SimilarityTransform2D applied.

    Raises:
        ValueError: If images are invalid or transform parameters are degenerate.
    """
    if not isinstance(transform, SimilarityTransform2D):
        raise TypeError(f"Expected SimilarityTransform2D instance, got {type(transform)}.")

    if image_a.size == 0 or image_b.size == 0:
        raise ValueError("Input images must not be empty.")

    h_a, w_a = image_a.shape[:2]
    h_b, w_b = image_b.shape[:2]

    if output_shape is not None:
        out_h, out_w = int(output_shape[0]), int(output_shape[1])
    else:
        out_h, out_w = h_a, w_a

    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"Invalid output dimensions: ({out_h}, {out_w}).")

    # Map interpolation string to OpenCV flag
    interp_map = {
        "linear": cv2.INTER_LINEAR,
        "bilinear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "bicubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
    }
    cv_interp = interp_map.get(interpolation.lower(), cv2.INTER_LINEAR)

    # Map border mode string to OpenCV flag
    border_map = {
        "constant": cv2.BORDER_CONSTANT,
        "reflect": cv2.BORDER_REFLECT,
        "replicate": cv2.BORDER_REPLICATE,
    }
    cv_border = border_map.get(border_mode.lower(), cv2.BORDER_CONSTANT)

    # Extract 2x3 affine matrix for warping Image B into Image A
    affine_mat = get_affine_matrix_from_similarity(transform)

    # Warp Image B into Image A's frame
    registered_image = cv2.warpAffine(
        image_b,
        affine_mat,
        (out_w, out_h),
        flags=cv_interp,
        borderMode=cv_border,
        borderValue=border_value,
    )

    # Create and warp source coverage mask to identify valid overlap pixels
    source_mask = np.ones((h_b, w_b), dtype=np.uint8) * 255
    valid_mask = cv2.warpAffine(
        source_mask,
        affine_mat,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return registered_image, valid_mask, transform

