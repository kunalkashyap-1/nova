import os
import asyncio
import json
import gc
from typing import AsyncGenerator, List, Dict, Optional
from datetime import datetime
import torch
from pathlib import Path
from dotenv import load_dotenv
import threading
import time

# Model providers
from ollama import AsyncClient as OllamaClient, ResponseError as OllamaResponseError, ChatResponse as OllamaChatResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import InferenceClient
from transformers import TextIteratorStreamer
from threading import Thread, Event
import requests

# Internal imports
from app.models.message import Message
from app.models.model import Model
from app.schemas.message import MessageStreamRequest
from app.utils.token import estimate_tokens
from app.nova_logger import logger

load_dotenv()

# Initialize clients
ollama_client = OllamaClient(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))

# Local models directory
LOCAL_MODELS_DIR = Path(os.getenv("HF_MODELS_PATH", "../hf_models"))
LOCAL_MODELS_DIR.mkdir(exist_ok=True)

# Optional HuggingFace Inference API client
hf_api_key = os.getenv("HF_API_KEY")
hf_client = InferenceClient(token=hf_api_key) if hf_api_key else None

# Global model management
class ModelManager:
    """Manages model loading/unloading to prevent resource conflicts"""
    
    def __init__(self):
        self.current_provider = None
        self.current_model_name = None
        self.hf_model = None
        self.hf_tokenizer = None
        self.ollama_active = False
        self.generation_threads = set()  # Track active generation threads
        
    def cleanup_hf_models(self):
        """Clean up HuggingFace models from memory"""
        if self.hf_model is not None:
            logger.info(f"Cleaning up HF model: {self.current_model_name}")
            del self.hf_model
            del self.hf_tokenizer
            self.hf_model = None
            self.hf_tokenizer = None
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection
            gc.collect()
            
    def cleanup_ollama(self):
        """Clean up Ollama resources"""
        if self.ollama_active:
            logger.info("Cleaning up Ollama resources")
            self.ollama_active = False
            
    def register_thread(self, thread):
        """Register a generation thread"""
        self.generation_threads.add(thread)
        
    def unregister_thread(self, thread):
        """Unregister a generation thread"""
        self.generation_threads.discard(thread)
        
    def stop_all_generations(self):
        """Stop all active generation threads"""
        for thread in list(self.generation_threads):
            if hasattr(thread, 'stop_event'):
                thread.stop_event.set()
            
    def switch_to_provider(self, provider: str, model_name: str = None):
        """Switch to a different provider, cleaning up the previous one"""
        if self.current_provider == provider and self.current_model_name == model_name:
            return  # Already using this provider/model
            
        logger.info(f"Switching from {self.current_provider} to {provider}")
        
        # Stop any running generations
        self.stop_all_generations()
        
        # Clean up current provider
        if self.current_provider == "huggingface":
            self.cleanup_hf_models()
        elif self.current_provider == "ollama":
            self.cleanup_ollama()
            
        # Update current provider
        self.current_provider = provider
        self.current_model_name = model_name
        
        if provider == "ollama":
            self.ollama_active = True
            
    def get_hf_model(self, model_name: str):
        """Get or load a HuggingFace model with proper cleanup"""
        # Switch to HF provider (will cleanup ollama if needed)
        self.switch_to_provider("huggingface", model_name)
        
        # If we already have this model loaded, return it
        if (self.hf_model is not None and 
            self.hf_tokenizer is not None and 
            self.current_model_name == model_name):
            return self.hf_model, self.hf_tokenizer
            
        # Clean up any existing HF model
        self.cleanup_hf_models()
        
        model_path = LOCAL_MODELS_DIR / model_name
        if not model_path.exists():
            raise ValueError(f"Model {model_name} not found in local_models directory")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading HF model {model_name} on {device}")
        
        try:
            self.hf_tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=None
            ).to(device)
            
            self.current_model_name = model_name
            return self.hf_model, self.hf_tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            self.cleanup_hf_models()
            raise
            
    def prepare_ollama(self):
        """Prepare for Ollama usage"""
        self.switch_to_provider("ollama")

# Global model manager instance
model_manager = ModelManager()


class StoppableTextIteratorStreamer(TextIteratorStreamer):
    """TextIteratorStreamer that can be stopped externally"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_event = Event()
        
    def put(self, value):
        if not self.stop_event.is_set():
            super().put(value)
            
    def end(self):
        if not self.stop_event.is_set():
            super().end()
            
    def stop(self):
        """Stop the streamer"""
        self.stop_event.set()
        # Put a sentinel value to unblock any waiting threads
        try:
            super().end()
        except:
            pass


class StoppableGenerationThread(Thread):
    """Thread that can be stopped and monitors for abort conditions"""
    
    def __init__(self, model, generation_kwargs, stop_event):
        super().__init__()
        self.model = model
        self.generation_kwargs = generation_kwargs
        self.stop_event = stop_event
        self.daemon = True  # Dies when main thread dies
        
    def run(self):
        try:
            # Monitor stop event during generation
            original_generate = self.model.generate
            
            def stoppable_generate(*args, **kwargs):
                if self.stop_event.is_set():
                    return None
                return original_generate(*args, **kwargs)
            
            # Replace generate method temporarily
            self.model.generate = stoppable_generate
            
            if not self.stop_event.is_set():
                self.model.generate(**self.generation_kwargs)
                
        except Exception as e:
            logger.error(f"Generation thread error: {str(e)}")
        finally:
            # Restore original generate method
            if hasattr(self, 'model'):
                self.model.generate = original_generate


def get_local_model(model_name: str):
    """
    Get or load a local HuggingFace model with proper resource management.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Tuple of (model, tokenizer)
    """
    return model_manager.get_hf_model(model_name)


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
    Enhanced with proper resource management.
    """
    # Ensure we're using Ollama provider
    model_manager.prepare_ollama()
    
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
    Process the response from HuggingFace models with proper abort handling.
    Enhanced with stop event monitoring and cleanup.
    """
    collected_reply = ""
    start_time = datetime.now()
    stop_event = Event()
    generation_thread = None
    streamer = None
    
    try:
        # Get local model with proper resource management
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
            # Create StoppableTextIteratorStreamer for streaming with abort capability
            streamer = StoppableTextIteratorStreamer(
                tokenizer, 
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=10.0  # Shorter timeout for better responsiveness
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
                "max_new_tokens": min(int(os.getenv("HF_MAX_NEW_TOKENS", "512")), 512),
                "temperature": float(os.getenv("HF_TEMPERATURE", "0.3")),
                "top_p": 0.9,
                "top_k": 50,
                "do_sample": True,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            }
            
            # Start generation in a stoppable thread
            generation_thread = StoppableGenerationThread(model, generation_kwargs, stop_event)
            generation_thread.start()
            
            # Register the thread for cleanup
            model_manager.register_thread(generation_thread)
            
            # Stream the tokens as they're generated with abort monitoring
            try:
                timeout_count = 0
                max_timeouts = 5  # Maximum number of consecutive timeouts before aborting
                
                while generation_thread.is_alive() or not streamer.stop_event.is_set():
                    try:
                        # Check if we should stop
                        if stop_event.is_set():
                            logger.info("Generation aborted by client disconnect")
                            break
                            
                        # Get next token with short timeout
                        new_text = next(iter(streamer), None)
                        
                        if new_text is None:
                            # Check if thread is still alive
                            if not generation_thread.is_alive():
                                break
                            timeout_count += 1
                            if timeout_count >= max_timeouts:
                                logger.warning("Too many timeouts, stopping generation")
                                stop_event.set()
                                break
                            await asyncio.sleep(0.1)
                            continue
                            
                        timeout_count = 0  # Reset timeout counter
                        
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
                            
                            try:
                                yield f"data: {json.dumps(event)}\n\n"
                            except Exception as stream_error:
                                # Client disconnected, stop generation
                                logger.info(f"Client disconnected: {stream_error}")
                                stop_event.set()
                                break
                                
                            # Small delay to prevent overwhelming the client
                            await asyncio.sleep(0.01)
                            
                    except StopIteration:
                        # Streamer finished
                        break
                    except Exception as streaming_error:
                        logger.error(f"Streaming error: {str(streaming_error)}")
                        stop_event.set()
                        break
                        
            except GeneratorExit:
                # Generator was closed (client disconnected)
                logger.info("Generator closed, stopping HF generation")
                stop_event.set()
            except Exception as e:
                logger.error(f"Unexpected error in streaming: {str(e)}")
                stop_event.set()
            finally:
                # Cleanup: Stop the generation thread and streamer
                if streamer:
                    streamer.stop()
                stop_event.set()
                
                if generation_thread and generation_thread.is_alive():
                    logger.info("Waiting for generation thread to stop...")
                    generation_thread.join(timeout=2.0)
                    if generation_thread.is_alive():
                        logger.warning("Generation thread did not stop gracefully")
                
                # Unregister the thread
                if generation_thread:
                    model_manager.unregister_thread(generation_thread)
            
        else:
            # Non-streaming generation with abort capability
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_attention_mask=True
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Quick check if we should abort before generation
            if stop_event.is_set():
                return
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=min(int(os.getenv("HF_MAX_NEW_TOKENS", "512")), 512),
                    temperature=float(os.getenv("HF_TEMPERATURE", "0.7")),
                    top_p=0.9,
                    top_k=50,
                    do_sample=True,
                    repetition_penalty=1.1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                )
            
            # Check again if we should abort after generation
            if stop_event.is_set():
                return
            
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
        
        # Only store in database if we weren't aborted and have content
        if not stop_event.is_set() and collected_reply.strip():
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
        else:
            logger.info("Generation was aborted or produced no content, not storing in database")
        
    except GeneratorExit:
        # Client disconnected
        logger.info("Client disconnected during HuggingFace generation")
        stop_event.set()
    except Exception as e:
        logger.error(f"HuggingFace local model error: {str(e)}")
        stop_event.set()
        error_event = {
            "error": str(e),
            "status_code": 500
        }
        try:
            yield f"data: {json.dumps(error_event)}\n\n"
        except:
            pass  # Client may have disconnected
    finally:
        # Final cleanup
        if streamer:
            streamer.stop()
        stop_event.set()
        if generation_thread:
            model_manager.unregister_thread(generation_thread)


# Additional utility functions for manual cleanup
def cleanup_all_models():
    """Clean up all loaded models - useful for manual cleanup"""
    model_manager.stop_all_generations()
    model_manager.cleanup_hf_models()
    model_manager.cleanup_ollama()
    model_manager.current_provider = None
    model_manager.current_model_name = None


def get_current_provider_status():
    """Get current provider status for debugging"""
    return {
        "current_provider": model_manager.current_provider,
        "current_model": model_manager.current_model_name,
        "hf_model_loaded": model_manager.hf_model is not None,
        "ollama_active": model_manager.ollama_active,
        "active_threads": len(model_manager.generation_threads)
    }