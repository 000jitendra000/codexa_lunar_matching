"""
api/main.py

FastAPI Application Entry Point for Cross-Sensor Lunar Location Matching API.

Provides:
- Reusable per-process model singleton lifecycle (LoFTR loaded once at startup).
- In-process thread-safe background job worker execution.
- Real-time Server-Sent Events (SSE) progress streaming.
"""

import os
import sys
import asyncio
import logging
from contextlib import asynccontextmanager

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.matching.hybrid_matcher import HybridMatcher
from src.registration.registration_engine import RegistrationEngine
from api.jobs import JobManager
from api.routes import router, set_job_manager
from api.progress import progress_tracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("lunar_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle context manager.
    Initializes reusable singleton model instances once per process at startup.
    """
    logger.info("Initializing Lunar Matching Engine (HybridMatcher + RegistrationEngine)...")
    
    # Instantiate process-wide singleton models
    matcher = HybridMatcher()
    registration_engine = RegistrationEngine()
    
    # Create JobManager
    job_manager = JobManager(matcher=matcher, registration_engine=registration_engine)
    set_job_manager(job_manager)

    # Wire asyncio event loop for thread-safe SSE progress streaming
    loop = asyncio.get_running_loop()
    progress_tracker.set_loop(loop)

    logger.info("Lunar Matching Engine initialized and ready to process requests.")

    yield

    logger.info("Shutting down Lunar Matching Engine...")


def create_app() -> FastAPI:
    """Factory function creating configured FastAPI application."""
    app = FastAPI(
        title="Cross-Sensor Lunar Location Matching API",
        description=(
            "Clean asynchronous REST API adapter exposing the Lunar Matching Engine. "
            "Supports image pair matching, job status polling, and live progress reporting via Server-Sent Events (SSE)."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # Enable CORS for external frontend integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routes
    app.include_router(router)

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.join(root_dir, "assets")

    # Mount static assets folder for CSS & JS dependencies
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/", include_in_schema=False)
    @app.get("/index", include_in_schema=False)
    @app.get("/index.html", include_in_schema=False)
    async def serve_index():
        return FileResponse(os.path.join(root_dir, "index.html"))

    @app.get("/app", include_in_schema=False)
    @app.get("/app.html", include_in_schema=False)
    async def serve_app():
        return FileResponse(os.path.join(root_dir, "app.html"))

    @app.get("/results", include_in_schema=False)
    @app.get("/results.html", include_in_schema=False)
    @app.get("/Results.html", include_in_schema=False)
    async def serve_results():
        return FileResponse(os.path.join(root_dir, "Results.html"))

    @app.get("/results-correspondence", include_in_schema=False)
    @app.get("/results-correspondence.html", include_in_schema=False)
    async def serve_results_correspondence():
        return FileResponse(os.path.join(root_dir, "results-correspondence.html"))

    @app.get("/results-overlay", include_in_schema=False)
    @app.get("/results-overlay.html", include_in_schema=False)
    async def serve_results_overlay():
        return FileResponse(os.path.join(root_dir, "results-overlay.html"))

    @app.get("/results-checkerboard", include_in_schema=False)
    @app.get("/results-checkerboard.html", include_in_schema=False)
    async def serve_results_checkerboard():
        return FileResponse(os.path.join(root_dir, "results-checkerboard.html"))

    @app.get("/results-cmap-a", include_in_schema=False)
    @app.get("/results-cmap-a.html", include_in_schema=False)
    async def serve_results_cmap_a():
        return FileResponse(os.path.join(root_dir, "results-cmap-a.html"))

    @app.get("/results-cmap-b", include_in_schema=False)
    @app.get("/results-cmap-b.html", include_in_schema=False)
    async def serve_results_cmap_b():
        return FileResponse(os.path.join(root_dir, "results-cmap-b.html"))

    @app.get("/lunar_matching_frontend", include_in_schema=False)
    @app.get("/lunar_matching", include_in_schema=False)
    @app.get("/lunar_match", include_in_schema=False)
    async def serve_lunar_matching_frontend():
        return FileResponse(os.path.join(root_dir, "Lunar_matching_frontend.html"))

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        from fastapi.openapi.utils import get_openapi
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            openapi_version=app.openapi_version,
            description=app.description,
            routes=app.routes,
        )
        # Ensure UploadFile fields (image_a, image_b) in multipart request bodies
        # include format: binary so Swagger UI renders file upload controls ([Choose File])
        schemas = openapi_schema.get("components", {}).get("schemas", {})
        for schema_name, schema_val in schemas.items():
            if "Body_submit_match" in schema_name or "properties" in schema_val:
                props = schema_val.get("properties", {})
                for field_name in ["image_a", "image_b"]:
                    if field_name in props:
                        props[field_name]["type"] = "string"
                        props[field_name]["format"] = "binary"

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)
