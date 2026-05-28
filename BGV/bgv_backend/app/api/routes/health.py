"""
app/api/routes/health.py
─────────────────────────
GET /health — liveness probe endpoint.
Returns app version and environment. Used by Docker HEALTHCHECK and
load-balancer probes.
"""

from fastapi import APIRouter
from app.core.config import settings
from app.schemas.document import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns `ok` when the service is up and ready to accept requests.",
)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        env=settings.app_env,
    )