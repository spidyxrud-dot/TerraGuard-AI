from fastapi import FastAPI

from app.api.analysis import router as analysis_router
from app.api.health import router as health_router

app = FastAPI(
    title="TerraGuard AI API",
    description="Environmental intelligence from paired satellite observations.",
    version="0.1.0",
)

app.include_router(health_router, prefix="/api")
app.include_router(analysis_router, prefix="/api")
