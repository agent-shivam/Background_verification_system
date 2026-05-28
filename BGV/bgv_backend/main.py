"""
main.py
────────
Application entry-point.

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Or via the helper at the bottom:
    python main.py
"""
from dotenv import load_dotenv

load_dotenv()
from app.api import create_app
from app.core.config import settings

# Build the ASGI application — uvicorn/gunicorn imports `app` from here
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )