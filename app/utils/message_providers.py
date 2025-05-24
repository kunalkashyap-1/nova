"""
Message provider clients and utility functions for handling different LLM provider APIs.
"""
import os
import asyncio
import logging
import json
from typing import AsyncGenerator, List, Dict, Optional
from datetime import datetime

# Model providers
from ollama import AsyncClient as OllamaClient, ResponseError as OllamaResponseError, ChatResponse as OllamaChatResponse
from huggingface_hub import InferenceClient

# Internal imports
from app.models.message import Message
from app.models.model import Model
from app.schemas.message import MessageStreamRequest
from app.utils.token import estimate_tokens

logger = logging.getLogger(__name__)

# Initialize clients
ollama_client = OllamaClient(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))

# Optional HuggingFace Inference API client
hf_api_key = os.getenv("HF_API_KEY")
hf_client = InferenceClient(token=hf_api_key) if hf_api_key else None


async def process_ollama_response(
    stream,
    request: MessageStreamRequest,
    db,  # SQLAlchemy Session
    user_message_id: int,
    store_message_func,
    store_in_vector_db_func
) -> AsyncGenerator[str, None]:
    """
    Process the streaming response from Ollama.
    
    Args:
        stream: Ollama streaming response
        request: The message request object
        db: Database session
        user_message_id: ID of the user message this is responding to
        store_message_func: Function to store message in database
        store_in_vector_db_func: Function to store message in vector database
        
    Yields:
        Formatted SSE data with message chunks
    """
    collected_reply = ""
    start_time = datetime.now()
    
    try:
        async for chunk in stream:
            if isinstance(chunk, OllamaChatResponse):
                reply = chunk.message.content
                raw = chunk.dict()
            else:
                reply = chunk.get("message", {}).get("content", "")
                raw = chunk
                
            collected_reply += reply

            event = {
                "model": request.model,
                "user_id": request.user_id,
                "chat_id": request.chat_id,  
                "reply": reply,
                "raw": {
                    "text": reply,
                    "done": False,
                    "time": (datetime.now() - start_time).total_seconds()
                }
            }
            yield f"data: {json.dumps(event)}\n\n"

        # Get model_id from the model name
        model = db.query(Model).filter(
            Model.provider == "ollama", 
            Model.model_id == request.model
        ).first()
        model_id = model.id if model else None

        # Store assistant's full reply in the database
        assistant_message = await store_message_func(
            db=db,
            conversation_id=request.chat_id,
            role="assistant",
            content=collected_reply,
            model_id=model_id,
            tokens_used=estimate_tokens(collected_reply),
            message_metadata={"provider": "ollama"},
            parent_message_id=user_message_id
        )
        
        # Store in vector DB for future context
        await store_in_vector_db_func(assistant_message.id, request.chat_id, collected_reply)

        # Send final event
        final_event = {
            "model": request.model,
            "user_id": request.user_id,
            "chat_id": request.chat_id,
            "reply": "",
            "raw": {
                "text": collected_reply,
                "done": True,
                "total_duration": (datetime.now() - start_time).total_seconds()
            }
        }
        yield f"data: {json.dumps(final_event)}\n\n"

    except OllamaResponseError as e:
        error_event = {
            "error": str(e),
            "status_code": getattr(e, "status_code", 500)
        }
        yield f"data: {json.dumps(error_event)}\n\n"


async def process_huggingface_response(
    request: MessageStreamRequest,
    messages: List[Dict],
    db,  # SQLAlchemy Session
    user_message_id: int,
    store_message_func,
    store_in_vector_db_func
) -> AsyncGenerator[str, None]:
    """
    Process the response from HuggingFace models.
    
    Args:
        request: The message request object
        messages: Formatted messages to send to the model
        db: Database session
        user_message_id: ID of the user message this is responding to
        store_message_func: Function to store message in database
        store_in_vector_db_func: Function to store message in vector database
        
    Yields:
        Formatted SSE data with message chunks or complete responses
    """
    if not hf_client:
        error_event = {
            "error": "HuggingFace API key not configured",
            "status_code": 500
        }
        yield f"data: {json.dumps(error_event)}\n\n"
        return

    conversation_text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            conversation_text += f"User: {content}\n"
        elif role == "assistant":
            conversation_text += f"Assistant: {content}\n"
        elif role == "system":
            conversation_text += f"System: {content}\n"
    
    collected_reply = ""
    start_time = datetime.now()
    
    try:
        if request.stream:
            for text_chunk in hf_client.text_generation(
                request.model,
                conversation_text + "Assistant: ",
                max_new_tokens=int(os.getenv("HF_MAX_NEW_TOKENS", "512")),
                temperature=float(os.getenv("HF_TEMPERATURE", "0.7")),
                stream=True
            ):
                collected_reply += text_chunk
                
                event = {
                    "model": request.model,
                    "user_id": request.user_id,
                    "chat_id": request.chat_id,
                    "reply": text_chunk,
                    "raw": {
                        "text": text_chunk,
                        "done": False,
                        "time": (datetime.now() - start_time).total_seconds()
                    }
                }
                yield f"data: {json.dumps(event)}\n\n"
                await asyncio.sleep(float(os.getenv("TYPING_DELAY", "0.01")))
        else:
            response = hf_client.text_generation(
                request.model,
                conversation_text + "Assistant: ",
                max_new_tokens=int(os.getenv("HF_MAX_NEW_TOKENS", "512")),
                temperature=float(os.getenv("HF_TEMPERATURE", "0.7"))
            )
            collected_reply = response
            
            event = {
                "model": request.model,
                "user_id": request.user_id,
                "chat_id": request.chat_id,
                "reply": collected_reply,
                "raw": {
                    "text": collected_reply,
                    "done": True,
                    "total_duration": (datetime.now() - start_time).total_seconds()
                }
            }
            yield f"data: {json.dumps(event)}\n\n"
        
        # Get model_id from the model name
        model = db.query(Model).filter(
            Model.provider == "huggingface", 
            Model.model_id == request.model
        ).first()
        model_id = model.id if model else None

        # Store assistant's reply in the database
        assistant_message = await store_message_func(
            db=db,
            conversation_id=request.chat_id,
            role="assistant",
            content=collected_reply,
            model_id=model_id,
            tokens_used=estimate_tokens(collected_reply),
            message_metadata={"provider": "huggingface"},
            parent_message_id=user_message_id
        )
        
        # Store in vector DB for future context
        await store_in_vector_db_func(assistant_message.id, request.chat_id, collected_reply)
        
        # Send final event for streaming responses
        if request.stream:
            final_event = {
                "model": request.model,
                "user_id": request.user_id,
                "chat_id": request.chat_id,
                "reply": "",
                "raw": {
                    "text": collected_reply,
                    "done": True,
                    "total_duration": (datetime.now() - start_time).total_seconds()
                }
            }
            yield f"data: {json.dumps(final_event)}\n\n"
        
    except Exception as e:
        logger.error(f"HuggingFace API error: {str(e)}")
        error_event = {
            "error": str(e),
            "status_code": 500
        }
        yield f"data: {json.dumps(error_event)}\n\n"
