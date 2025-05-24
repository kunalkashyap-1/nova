"""
Model management API endpoints for retrieving model information.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel

from app.database import get_db
from app.services.model_service import (
    get_all_models,
    get_models_by_provider,
    get_model_by_id,
    get_user_model_preferences
)


# Models Response Schema
class ModelCapability(BaseModel):
    name: str
    value: bool


class ModelParameter(BaseModel):
    name: str
    default_value: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None


class ModelResponse(BaseModel):
    id: int
    name: str
    provider: str
    model_id: str
    description: Optional[str] = None
    capabilities: Optional[dict] = None
    parameters: Optional[dict] = None
    is_active: bool = True
    cost_per_1k_input_tokens: Optional[float] = None
    cost_per_1k_output_tokens: Optional[float] = None
    max_tokens: Optional[int] = None
    
    class Config:
        orm_mode = True


router = APIRouter(prefix="/api/v1/models", tags=["Models"])


@router.get("/", response_model=List[ModelResponse], summary="Get all available models")
async def get_models(
    provider: Optional[str] = Query(None, description="Filter by provider"),
    db: Session = Depends(get_db)
):
    """
    Get all available language models.
    
    This endpoint returns a list of all active models available in the system.
    Optionally filter by provider (e.g., 'ollama', 'huggingface').
    
    Returns:
        List of model information
    """
    print(f"Getting models with provider: {provider}")
    try:
        if provider != "all":
            models = await get_models_by_provider(db, provider)
        else:
            models = await get_all_models(db)
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_id}", response_model=ModelResponse, summary="Get model details")
async def get_model(
    model_id: int,
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific model.
    
    Path parameters:
    - model_id: ID of the model to retrieve
    
    Returns:
        Detailed model information
    
    Raises:
        404: If the model is not found
    """
    model = await get_model_by_id(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return model


@router.get("/user/{user_id}/preferences", summary="Get user model preferences")
async def get_preferences(
    user_id: int,
    db: Session = Depends(get_db)
):
    """
    Get model preferences for a specific user.
    
    Path parameters:
    - user_id: ID of the user
    
    Returns:
        User's model preferences including favorites and default settings
    """
    preferences = await get_user_model_preferences(db, user_id)
    return preferences
