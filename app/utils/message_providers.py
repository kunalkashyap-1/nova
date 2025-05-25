"""
Message provider clients and utility functions for handling different LLM provider APIs.
"""
import os
import asyncio
import logging
import json
from typing import AsyncGenerator, List, Dict, Optional
from datetime import datetime
import torch
from pathlib import Path
from dotenv import load_dotenv

# Model providers
from ollama import AsyncClient as OllamaClient, ResponseError as OllamaResponseError, ChatResponse as OllamaChatResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import InferenceClient

# Internal imports
from app.models.message import Message
from app.models.model import Model
from app.schemas.message import MessageStreamRequest
from app.utils.token import estimate_tokens

logger = logging.getLogger(__name__)
load_dotenv()


# Initialize clients
# ollama_client = OllamaClient(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))

# Local models directory

LOCAL_MODELS_DIR = Path(os.getenv("HF_MODELS_PATH", "../hf_models"))
LOCAL_MODELS_DIR.mkdir(exist_ok=True)

# Optional HuggingFace Inference API client
hf_api_key = os.getenv("HF_API_KEY")
hf_client = InferenceClient(token=hf_api_key) if hf_api_key else None

# Cache for local models
local_model_cache = {}

def get_local_model(model_name: str):
    """
    Get or load a local HuggingFace model.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Tuple of (model, tokenizer, pipeline)
    """
    if model_name in local_model_cache:
        return local_model_cache[model_name]
    
    model_path = LOCAL_MODELS_DIR / model_name
    if not model_path.exists():
        raise ValueError(f"Model {model_name} not found in local_models directory")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading model {model_name} on {device}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=int(os.getenv("HF_MAX_NEW_TOKENS", "4019")),
            temperature=float(os.getenv("HF_TEMPERATURE", "0.7"))
        )
        
        local_model_cache[model_name] = (model, tokenizer, pipe)
        return local_model_cache[model_name]
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        raise

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
        # Get local model
        model, tokenizer, pipe = get_local_model(request.model)
        
        if request.stream:
            # For streaming, we'll generate token by token
            inputs = tokenizer(conversation_text + "Assistant: ", return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            generated_ids = []
            for _ in range(int(os.getenv("HF_MAX_NEW_TOKENS", "4019"))):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    temperature=float(os.getenv("HF_TEMPERATURE", "0.7")),
                    pad_token_id=tokenizer.eos_token_id
                )
                next_token = outputs[0][-1]
                generated_ids.append(next_token)
                
                # Decode the new token
                new_text = tokenizer.decode([next_token])
                collected_reply += new_text
                
                event = {
                    "model": request.model,
                    "user_id": request.user_id,
                    "chat_id": request.chat_id,
                    "reply": new_text,
                    "raw": {
                        "text": new_text,
                        "done": False,
                        "time": (datetime.now() - start_time).total_seconds()
                    }
                }
                yield f"data: {json.dumps(event)}\n\n"
                
                # Update inputs for next iteration
                inputs = {"input_ids": outputs}
                await asyncio.sleep(float(os.getenv("TYPING_DELAY", "0.01")))
                
                # Check for end of generation
                if next_token == tokenizer.eos_token_id:
                    break
        else:
            # Non-streaming generation
            response = pipe(
                conversation_text + "Assistant: ",
                max_new_tokens=int(os.getenv("HF_MAX_NEW_TOKENS", "4019")),
                temperature=float(os.getenv("HF_TEMPERATURE", "0.7"))
            )
            collected_reply = response[0]["generated_text"][len(conversation_text + "Assistant: "):]
            
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
        logger.error(f"HuggingFace local model error: {str(e)}")
        error_event = {
            "error": str(e),
            "status_code": 500
        }
        yield f"data: {json.dumps(error_event)}\n\n"
