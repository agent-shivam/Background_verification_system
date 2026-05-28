"""
app/api/__init__.py
────────────────────
FastAPI application factory.
Import `create_app` in main.py — keeps startup logic here, not in
the entry-point, making the app easier to test.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.exceptions import BGVBaseError
from app.core.logging import setup_logging
from app.api.routes.health import router as health_router
from app.api.routes.documents import router as document_router


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""

    setup_logging()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "**AI/ML-powered Background Verification System**\n\n"
            "Extracts structured data from identity documents and resumes, "
            "then runs a multi-layer forgery detection suite to produce a "
            "composite risk score.\n\n"
            "### Supported documents\n"
            "Aadhaar · PAN · Passport · Resume · Graduation Certificate · Marksheet\n\n"
            "### Fraud detection modules\n"
            "ELA · EXIF metadata · Blur / sharpness · ORB copy-move · "
            "Layout validation · QR cross-validation · AI-artefact detection"
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        debug=settings.debug,
    )

    # ── CORS (wide-open for development; tighten in production) ───────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Global exception handler for BGV domain errors ────────────────────────
    @app.exception_handler(BGVBaseError)
    async def bgv_exception_handler(request: Request, exc: BGVBaseError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content={"error": type(exc).__name__, "message": exc.message},
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health_router)
    app.include_router(document_router)

    return app