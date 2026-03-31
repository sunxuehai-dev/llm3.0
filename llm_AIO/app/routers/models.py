from fastapi import APIRouter
from app.services.registry import ModelRegistry

router = APIRouter(prefix="/api/playground", tags=["models"])

@router.get("/models")
async def get_models():
    """
    Get list of available models and their configuration parameters.
    Frontend uses this to render the model selector and parameter sliders.
    """
    return {
        "models": ModelRegistry.get_all_models()
    }
