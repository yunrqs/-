from fastapi import APIRouter
from ..modules.monitoring.routes import router as monitoring_router

api_router = APIRouter()
api_router.include_router(monitoring_router)
