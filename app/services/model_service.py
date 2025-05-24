"""
Model management services for retrieving and caching model information.
"""
import json
from typing import List, Dict, Optional
import os
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.models.model import Model
from app.utils.redis import cache_get, cache_set, cache_hgetall

# Model cache TTL (1 hour)
MODEL_CACHE_TTL = int(os.getenv("MODEL_CACHE_TTL", "3600"))

# Cache key for model lists
MODEL_LIST_KEY = "models:list"
MODEL_BY_PROVIDER_KEY = "models:by_provider"


async def get_all_models(db: Session) -> List[Model]:
    """
    Get all active models with caching.
    
    Args:
        db: Database session
        
    Returns:
        List of active model objects
    """
    # Try to get from cache first
    cached_models = await cache_get(MODEL_LIST_KEY)
    
    if cached_models:
        try:
            # Deserialize models from cache
            models_data = json.loads(cached_models)
            # Convert to ORM objects
            return [Model(**model_data) for model_data in models_data]
        except Exception:
            # In case of cache corruption, continue to DB query
            pass
    
    # Get from database
    models = db.query(Model).filter(
        Model.is_active == True
    ).order_by(
        desc(Model.cost_per_1k_output_tokens),  # Premium models first
        Model.name
    ).all()
    
    # Cache for future use
    if models:
        # Serialize to JSON-compatible format
        models_data = [{
            "id": model.id,
            "name": model.name,
            "provider": model.provider,
            "model_id": model.model_id,
            "description": model.description,
            "capabilities": model.capabilities,
            "parameters": model.parameters,
            "is_active": model.is_active,
            "cost_per_1k_input_tokens": float(model.cost_per_1k_input_tokens) if model.cost_per_1k_input_tokens else 0,
            "cost_per_1k_output_tokens": float(model.cost_per_1k_output_tokens) if model.cost_per_1k_output_tokens else 0,
            "max_tokens": model.max_tokens,
            "created_at": model.created_at.isoformat() if model.created_at else None,
            "updated_at": model.updated_at.isoformat() if model.updated_at else None
        } for model in models]
        
        await cache_set(MODEL_LIST_KEY, json.dumps(models_data), ex=MODEL_CACHE_TTL)
    
    return models


async def get_models_by_provider(db: Session, provider: str) -> List[Model]:
    """
    Get all active models for a specific provider.
    
    Args:
        db: Database session
        provider: Provider name (e.g., 'ollama', 'huggingface')
        
    Returns:
        List of active model objects for the provider
    """
    # Try provider-specific cache
    cache_key = f"{MODEL_BY_PROVIDER_KEY}:{provider}"
    cached_models = await cache_get(cache_key)
    
    if cached_models:
        try:
            models_data = json.loads(cached_models)
            return [Model(**model_data) for model_data in models_data]
        except Exception:
            pass
    
    # Get from database
    models = db.query(Model).filter(
        Model.is_active == True,
        Model.provider == provider
    ).order_by(
        desc(Model.cost_per_1k_output_tokens),  # Premium models first
        Model.name
    ).all()
    
    # Cache for future use
    if models:
        models_data = [{
            "id": model.id,
            "name": model.name,
            "provider": model.provider,
            "model_id": model.model_id,
            "description": model.description,
            "capabilities": model.capabilities,
            "parameters": model.parameters,
            "is_active": model.is_active,
            "cost_per_1k_input_tokens": float(model.cost_per_1k_input_tokens) if model.cost_per_1k_input_tokens else 0,
            "cost_per_1k_output_tokens": float(model.cost_per_1k_output_tokens) if model.cost_per_1k_output_tokens else 0,
            "max_tokens": model.max_tokens,
            "created_at": model.created_at.isoformat() if model.created_at else None,
            "updated_at": model.updated_at.isoformat() if model.updated_at else None
        } for model in models]
        
        await cache_set(cache_key, json.dumps(models_data), ex=MODEL_CACHE_TTL)
    
    return models


async def get_model_by_id(db: Session, model_id: int) -> Optional[Model]:
    """
    Get a specific model by ID with caching.
    
    Args:
        db: Database session
        model_id: ID of the model
        
    Returns:
        Model object if found, None otherwise
    """
    # Try to get from cache
    cache_key = f"model:{model_id}"
    cached_model = await cache_get(cache_key)
    
    if cached_model:
        try:
            model_data = json.loads(cached_model)
            return Model(**model_data)
        except Exception:
            # Continue to DB query on error
            pass
    
    # Get from database
    model = db.query(Model).filter(Model.id == model_id).first()
    
    # Cache for future use
    if model:
        model_data = {
            "id": model.id,
            "name": model.name,
            "provider": model.provider,
            "model_id": model.model_id,
            "description": model.description,
            "capabilities": model.capabilities,
            "parameters": model.parameters,
            "is_active": model.is_active,
            "cost_per_1k_input_tokens": float(model.cost_per_1k_input_tokens) if model.cost_per_1k_input_tokens else 0,
            "cost_per_1k_output_tokens": float(model.cost_per_1k_output_tokens) if model.cost_per_1k_output_tokens else 0,
            "max_tokens": model.max_tokens,
            "created_at": model.created_at.isoformat() if model.created_at else None,
            "updated_at": model.updated_at.isoformat() if model.updated_at else None
        }
        
        await cache_set(cache_key, json.dumps(model_data), ex=MODEL_CACHE_TTL)
    
    return model


async def get_user_model_preferences(db: Session, user_id: int) -> Dict:
    """
    Get user's model preferences with caching.
    
    Args:
        db: Database session
        user_id: ID of the user
        
    Returns:
        Dictionary of model preferences
    """
    # This would be implemented with a proper user_model_preferences table
    # For now, return default values from Redis
    cache_key = f"user:{user_id}:model_preferences"
    prefs = await cache_hgetall(cache_key)
    
    if not prefs:
        # Default preferences
        return {
            "default_model_id": None,
            "favorites": [],
            "custom_parameters": {}
        }
    
    return prefs
