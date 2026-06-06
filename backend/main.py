import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from backend.api.routes import router as api_router
from backend.api.twin_routes import twin_router
from backend.retriever.engine import get_knowledge_base_stats


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: run DB migrations on startup, close pool on shutdown."""
    # ── Startup ──────────────────────────────────────────────────────────────
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        try:
            from backend.db.init_db import run_migrations
            await run_migrations()
        except Exception as e:
            import logging
            logging.getLogger("CDSS.Main").warning(
                f"DB migration skipped (PostgreSQL unavailable): {e}"
            )
    else:
        import logging
        logging.getLogger("CDSS.Main").warning(
            "DATABASE_URL not set — Digital Twin features disabled. "
            "Add DATABASE_URL to .env to enable."
        )

    try:
        from backend.hpo.hpo_updater import HPOUpdater
        updater = HPOUpdater()
        updater.update()
    except Exception as e:
        import logging
        logging.getLogger("CDSS.Main").warning(
            f"HPO update on startup failed: {e}"
        )

    yield  # ── Application runs here ─────────────────────────────────────────

    # ── Shutdown ─────────────────────────────────────────────────────────────
    try:
        from backend.db.database import close_db_pool
        await close_db_pool()
    except Exception:
        pass


app = FastAPI(
    title="ProHealth Rare Disease CDSS",
    description=(
        "AI-powered Rare Disease Clinical Decision Support System. "
        "Combines RAG, HPO mapping, and Patient Digital Twins."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates and Static Files
# Note: Paths are resolved relative to the workspace root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


# Include API Routers
app.include_router(api_router, prefix="/api/v1")
app.include_router(twin_router, prefix="/api/v1")

from fastapi.responses import FileResponse

@app.get("/")
async def read_root(request: Request):
    """Renders the dashboard UI."""
    stats = get_knowledge_base_stats()
    
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "total_articles": stats["articles"],
            "total_genes": stats["genes"],
            "total_diseases": stats["diseases"]
        }
    )

@app.get("/rag.png")
async def get_rag_image():
    """Serves the anatomy visualization image."""
    image_path = os.path.join(BASE_DIR, "static", "images", "rag.png")
    if os.path.exists(image_path):
        return FileResponse(image_path)
    return {"error": "Image not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
