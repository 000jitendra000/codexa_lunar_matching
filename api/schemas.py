"""
api/schemas.py

Pydantic schemas for the Lunar Image Matching API.
Ensures strictly validated, JSON-safe data serialization.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check endpoint response schema."""
    status: str = Field("ok", description="Service status")
    service: str = Field("lunar-image-matching-api", description="Service name")
    model_available: bool = Field(True, description="Whether lunar matching engine is initialized")


class JobCreatedResponse(BaseModel):
    """Response returned when a new matching job is enqueued."""
    job_id: str = Field(..., description="Unique UUID identifier for the matching job")
    status: str = Field("queued", description="Initial job status ('queued')")


class TranslationSchema(BaseModel):
    """2D spatial translation parameters."""
    x: float = Field(..., description="Translation along X axis in reference image pixels")
    y: float = Field(..., description="Translation along Y axis in reference image pixels")


class TransformSchema(BaseModel):
    """2D Similarity transformation parameters."""
    scale: float = Field(..., description="Isotropic scale factor")
    rotation_deg: float = Field(..., description="Counter-clockwise rotation angle in degrees")
    translation: TranslationSchema = Field(..., description="Translation offset")


class MatchResultSummary(BaseModel):
    """
    JSON-serializable summary of hybrid matching and registration results.
    """
    matched: bool = Field(..., description="Whether geometrical match and verification succeeded")
    correspondences: int = Field(..., description="Total raw correspondences found across all branches")
    inliers: int = Field(..., description="Number of geometrically verified RANSAC inliers")
    inlier_ratio: float = Field(..., description="Ratio of inliers to raw correspondences in [0.0, 1.0]")
    confidence: float = Field(..., description="Overall registration confidence score in [0.0, 1.0]")
    rmse: Optional[float] = Field(None, description="Reprojection RMSE in pixels (None if unverified/failed)")
    mean_error: Optional[float] = Field(None, description="Mean reprojection error (None if unverified/failed)")
    median_error: Optional[float] = Field(None, description="Median reprojection error (None if unverified/failed)")
    max_error: Optional[float] = Field(None, description="Maximum reprojection error (None if unverified/failed)")
    coverage: float = Field(0.0, description="Spatial grid occupancy coverage in [0.0, 1.0]")
    quality: str = Field("FAILED", description="Quality classification ('EXCELLENT', 'GOOD', 'FAIR', 'POOR', 'FAILED')")
    transform: Optional[TransformSchema] = Field(None, description="Estimated Similarity transformation (None if failed)")
    failure_reason: Optional[str] = Field(None, description="Diagnostic explanation if matched=False")


class ProgressEventSchema(BaseModel):
    """Progress update event structure for status polling and SSE streaming."""
    stage: str = Field(..., description="Current pipeline stage")
    current: int = Field(0, description="Completed items in current stage")
    total: int = Field(1, description="Total items in current stage")
    progress: float = Field(0.0, description="Fractional progress in [0.0, 1.0]")
    message: str = Field("", description="Human-readable progress description")


class VisualizationSchema(BaseModel):
    """Paths / URLs to generated explainability visualization assets."""
    confidence_map_a: Optional[str] = Field(None, description="URL/path to Image A Match Confidence Map")
    confidence_map_b: Optional[str] = Field(None, description="URL/path to Image B Match Confidence Map")
    correspondence_image: Optional[str] = Field(None, description="URL/path to Side-by-Side Correspondence Line Plot")
    registration_overlay: Optional[str] = Field(None, description="URL/path to Registered Alpha Overlay")
    checkerboard: Optional[str] = Field(None, description="URL/path to Registered Checkerboard Comparison")


class JobStatusResponse(BaseModel):
    """Response returned when polling job status."""
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Job status ('queued', 'running', 'completed', 'failed')")
    progress: Optional[ProgressEventSchema] = Field(None, description="Latest progress event")
    result: Optional[MatchResultSummary] = Field(None, description="Final registration result summary (if completed)")
    visualizations: Optional[VisualizationSchema] = Field(None, description="Generated visualization assets (if available)")
    error: Optional[str] = Field(None, description="Safe error message (if failed)")
