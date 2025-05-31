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
from transformers import TextIteratorStreamer
from threading import Thread

# Internal imports
from app.models.message import Message
from app.models.model import Model
from app.schemas.message import MessageStreamRequest
from app.utils.token import estimate_tokens

logger = logging.getLogger(__name__)
load_dotenv()


# Initialize clients
ollama_client = OllamaClient(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))

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
    print(model_path);
    if not model_path.exists():
        raise ValueError(f"Model {model_name} not found in local_models directory")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading model {model_name} on {device}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            # device_map="auto" if device == "cuda" else None
            device_map=None
        ).to(device)
        
        local_model_cache[model_name] = (model, tokenizer)
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
    Process the response from HuggingFace models using TextIteratorStreamer.
    
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
    # print("messages", messages)
    collected_reply = ""
    start_time = datetime.now()
    
    try:
        # Get local model
        model, tokenizer = get_local_model(request.model)
        
        # Format messages properly using the tokenizer's chat template if available
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            # Use the model's built-in chat template
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Fallback to a more standard chat format
            conversation_text = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    conversation_text += f"<|system|>\n{content}\n\n"
                elif role == "user":
                    conversation_text += f"<|user|>\n{content}\n\n"
                elif role == "assistant":
                    conversation_text += f"<|assistant|>\n{content}\n\n"
            
            # Add the assistant prompt for generation
            prompt = conversation_text + "<|assistant|>\n"
        
        if request.stream:
            # Create TextIteratorStreamer for streaming
            streamer = TextIteratorStreamer(
                tokenizer, 
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=60.0
            )
            
            # Tokenize input
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_attention_mask=True
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generation kwargs - more conservative settings for better quality
            generation_kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": min(int(os.getenv("HF_MAX_NEW_TOKENS", "512")), 512),  # Reduced from 4019
                "temperature": float(os.getenv("HF_TEMPERATURE", "0.3")),  # Lower temperature for more coherent responses
                "top_p": 0.9,  # Add nucleus sampling
                "top_k": 50,   # Add top-k sampling
                "do_sample": True,
                # "repetition_penalty": 1.1,  
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            }
            
            # Start generation in a separate thread
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Stream the tokens as they're generated
            try:
                for new_text in streamer:
                    if new_text:  # Skip empty strings
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
                        
                        # Small delay to prevent overwhelming the client
                        await asyncio.sleep(0.01)
                        
            except Exception as streaming_error:
                logger.error(f"Streaming error: {str(streaming_error)}")
                # Wait for the thread to complete
                thread.join(timeout=5.0)
                raise
            
            # Wait for the generation thread to complete
            thread.join()
            
        else:
            # Non-streaming generation - use model.generate instead of pipeline for consistency
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_attention_mask=True
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=min(int(os.getenv("HF_MAX_NEW_TOKENS", "512")), 512),
                    temperature=float(os.getenv("HF_TEMPERATURE", "0.3")),
                    top_p=0.9,
                    top_k=50,
                    do_sample=True,
                    repetition_penalty=1.1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                )
            
            # Decode only the new tokens (skip the input)
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            collected_reply = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
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
        model_record = db.query(Model).filter(
            Model.provider == "huggingface", 
            Model.model_id == request.model
        ).first()
        model_id = model_record.id if model_record else None

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