import os
import asyncio
import json
import gc
import threading
import time
import signal
import ctypes
from typing import AsyncGenerator, List, Dict, Optional
from datetime import datetime
import torch
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, Future
import uuid

# Model providers
from ollama import AsyncClient as OllamaClient, ResponseError as OllamaResponseError, ChatResponse as OllamaChatResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
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
hf_api_key = os.getenv("HF_API_key")
hf_client = InferenceClient(token=hf_api_key) if hf_api_key else None


class CustomStoppingCriteria(StoppingCriteria):
    """
    Custom StoppingCriteria to allow immediate termination of generation.
    """
    def __init__(self, stop_event: Event):
        self.stop_event = stop_event

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.stop_event.is_set():
            logger.info("Stopping criteria met: stop event is set.")
            return True
        return False


class UserSessionManager:
    """Manages individual user sessions and their generation processes"""
    
    def __init__(self):
        self.active_sessions = {}  # session_id -> session_info
        self.session_lock = threading.Lock()
    
    def create_session(self, user_id: str, chat_id: str) -> str:
        """Create a new session for a user"""
        session_id = f"{user_id}_{chat_id}_{uuid.uuid4().hex[:8]}"
        
        with self.session_lock:
            self.active_sessions[session_id] = {
                'user_id': user_id,
                'chat_id': chat_id,
                'stop_event': Event(),  # Event to signal generation to stop
                'generation_future': None,
                'generation_thread': None,
                'executor': ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"gen_{session_id}"),
                'created_at': time.time()
            }
        
        logger.info(f"Created session {session_id} for user {user_id}")
        return session_id
    
    def stop_session(self, session_id: str, force: bool = False):
        """Stop a specific session's generation"""
        with self.session_lock:
            if session_id not in self.active_sessions:
                logger.info(f"Session {session_id} already cleaned up or does not exist.")
                return
            
            session = self.active_sessions[session_id]
            logger.info(f"Stopping session {session_id} (force={force})")
            
            # Signal the generation to stop
            session['stop_event'].set()
            
            # If there's a generation future/thread, try to cancel/join it
            if session['generation_future'] and not session['generation_future'].done():
                logger.info(f"Attempting to cancel generation future for session {session_id}")
                session['generation_future'].cancel()
            
            if session['generation_thread'] and session['generation_thread'].is_alive():
                logger.info(f"Waiting for generation thread to finish for session {session_id}")
                session['generation_thread'].join(timeout=5) # Give it some time to finish gracefully
                if session['generation_thread'].is_alive() and force:
                    logger.warning(f"Generation thread for session {session_id} did not terminate gracefully. Attempting force stop.")
                    self._force_stop_thread(session['generation_thread'])
            
            # Shutdown executor
            session['executor'].shutdown(wait=not force)
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            logger.info(f"Cleaned up session {session_id}")
    
    def _force_stop_thread(self, thread: Thread):
        """Forcefully terminate a thread (Linux/Windows compatible)"""
        if not thread.is_alive():
            return
            
        try:
            thread_id = thread.ident
            if thread_id is None:
                logger.warning(f"Could not get thread ID for {thread.name}. Cannot force stop.")
                return
            
            import sys
            if sys.platform.startswith('linux'):
                try:
                    os.kill(thread_id, signal.SIGTERM)
                    logger.warning(f"Sent SIGTERM to thread {thread_id} ({thread.name})")
                except ProcessLookupError:
                    logger.warning(f"Thread {thread_id} ({thread.name}) not found (already terminated?).")
                except Exception as e:
                    logger.error(f"Error sending SIGTERM to thread {thread_id} ({thread.name}): {e}")
            elif sys.platform.startswith('win'):
                try:
                    # Python's ctypes can be used to call Windows API functions
                    # OpenThread access rights: THREAD_TERMINATE (0x0001)
                    handle = ctypes.windll.kernel32.OpenThread(0x0001, False, thread_id)
                    if handle:
                        ctypes.windll.kernel32.TerminateThread(handle, 0)
                        ctypes.windll.kernel32.CloseHandle(handle)
                        logger.warning(f"Forcefully terminated thread {thread_id} ({thread.name}) on Windows.")
                    else:
                        logger.warning(f"Failed to get handle for thread {thread_id} ({thread.name}).")
                except Exception as e:
                    logger.error(f"Error terminating thread {thread_id} ({thread.name}) on Windows: {e}")
            else:
                logger.warning(f"Forceful termination not supported on platform: {sys.platform}")
                
        except Exception as e:
            logger.error(f"Failed to force stop thread: {e}")
    
    def get_session(self, session_id: str):
        """Get session info"""
        with self.session_lock:
            return self.active_sessions.get(session_id)
    
    def cleanup_old_sessions(self, max_age: int = 3600):
        """Cleanup sessions older than max_age seconds"""
        current_time = time.time()
        to_remove = []
        
        with self.session_lock:
            for session_id, session in list(self.active_sessions.items()): # Iterate over a copy
                if current_time - session['created_at'] > max_age:
                    to_remove.append(session_id)
        
        for session_id in to_remove:
            logger.info(f"Cleaning up old session {session_id}")
            self.stop_session(session_id, force=True)


class HuggingFaceGenerationThread(Thread):
    """
    Dedicated thread for HuggingFace model generation,
    responsive to a stop event.
    """
    def __init__(self, model, generation_kwargs, streamer, stop_event: Event, session_id: str):
        super().__init__()
        self.model = model
        self.generation_kwargs = generation_kwargs
        self.streamer = streamer
        self.stop_event = stop_event
        self.session_id = session_id
        self.daemon = True # Allow the program to exit even if this thread is running

    def run(self):
        try:
            logger.info(f"HuggingFace generation thread started for session {self.session_id}")
            # Add stopping criteria to generation_kwargs
            stopping_criteria = StoppingCriteriaList([CustomStoppingCriteria(self.stop_event)])
            self.generation_kwargs['stopping_criteria'] = stopping_criteria

            self.model.generate(**self.generation_kwargs)
            logger.info(f"HuggingFace generation thread completed for session {self.session_id}")
        except Exception as e:
            logger.error(f"HuggingFace generation thread error for session {self.session_id}: {e}")
        finally:
            # Ensure the streamer is ended, even if an error occurs or generation is stopped
            self.streamer.end()


# Global model manager with session support
class ModelManager:
    """Enhanced model manager with session support"""
    
    def __init__(self):
        self.current_provider = None
        self.current_model_name = None
        self.hf_model = None
        self.hf_tokenizer = None
        self.ollama_active = False
        self.session_manager = UserSessionManager()
        
        # Start cleanup thread
        self.cleanup_thread = Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def _cleanup_loop(self):
        """Background thread to cleanup old sessions"""
        while True:
            try:
                self.session_manager.cleanup_old_sessions(max_age=1800)  # 30 minutes
                time.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                time.sleep(60)
    
    def cleanup_ollama(self):
        """Clean up Ollama resources"""
        if self.ollama_active:
            logger.info("Cleaning up Ollama resources")
            self.ollama_active = False
    
    def cleanup_hf_models(self):
        """Clean up HuggingFace models from memory"""
        if self.hf_model is not None:
            logger.info(f"Cleaning up HF model: {self.current_model_name}")
            del self.hf_model
            del self.hf_tokenizer
            self.hf_model = None
            self.hf_tokenizer = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            gc.collect()
    
    def switch_to_provider(self, provider: str, model_name: str = None):
        """Switch to a different provider, cleaning up the previous one"""
        if self.current_provider == provider and self.current_model_name == model_name:
            return
            
        logger.info(f"Switching from {self.current_provider} (model: {self.current_model_name}) to {provider} (model: {model_name})")
        
        if self.current_provider == "huggingface":
            self.cleanup_hf_models()
        elif self.current_provider == "ollama":
            self.ollama_active = False
            
        self.current_provider = provider
        self.current_model_name = model_name
        
        if provider == "ollama":
            self.ollama_active = True
    
    def get_hf_model(self, model_name: str):
        """Get or load a HuggingFace model with proper cleanup"""
        self.switch_to_provider("huggingface", model_name)
        
        if (self.hf_model is not None and 
            self.hf_tokenizer is not None and 
            self.current_model_name == model_name):
            logger.info(f"Returning already loaded HF model: {model_name}")
            return self.hf_model, self.hf_tokenizer
            
        self.cleanup_hf_models() # Ensure a clean slate before loading
        
        model_path = LOCAL_MODELS_DIR / model_name
        if not model_path.exists():
            raise ValueError(f"Model {model_name} not found in local_models directory: {model_path}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading HF model {model_name} on {device}")
        
        try:
            self.hf_tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=None # Let .to(device) handle it
            ).to(device)
            
            self.current_model_name = model_name
            return self.hf_model, self.hf_tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            self.cleanup_hf_models()
            raise
    
    def create_user_session(self, user_id: str, chat_id: str) -> str:
        """Create a new user session"""
        return self.session_manager.create_session(user_id, chat_id)
    
    def stop_user_session(self, session_id: str, force: bool = True):
        """Stop a specific user session"""
        self.session_manager.stop_session(session_id, force=force)
    
    def prepare_ollama(self):
        """Prepare for Ollama usage"""
        self.switch_to_provider("ollama")


# Global model manager instance
model_manager = ModelManager()


async def process_ollama_response(
    stream,
    request: MessageStreamRequest,
    db,  # SQLAlchemy Session
    user_message_id: int,
    store_message_func,
    store_in_vector_db_func,
    session_id: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """
    Process the streaming response from Ollama.
    Enhanced with proper resource management and optional session support.
    """
    # Ensure we're using Ollama provider
    model_manager.prepare_ollama()
    
    collected_reply = ""
    start_time = datetime.now()
    
    # Check session if provided
    session = None
    if session_id:
        session = model_manager.session_manager.get_session(session_id)
        if not session:
            logger.error(f"Session {session_id} not found for Ollama stream.")
            return # Cannot proceed without a valid session if one was expected
    
    try:
        async for chunk in stream:
            # Check if session is stopped (if session management is used)
            if session and session['stop_event'].is_set():
                logger.info(f"Session {session_id} stopped, breaking Ollama stream early.")
                break
                
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
            
            try:
                yield f"data: {json.dumps(event)}\n\n"
            except Exception as stream_error:
                # Client disconnected, typically an asyncio.CancelledError or similar
                logger.info(f"Client disconnected from Ollama stream for session {session_id}: {stream_error}")
                if session_id:
                    model_manager.stop_user_session(session_id, force=True)
                break

        # Get model_id from the model name
        model = db.query(Model).filter(
            Model.provider == "ollama", 
            Model.model_id == request.model
        ).first()
        model_db_id = model.id if model else None

        # Store assistant's full reply in the database (only if not explicitly stopped)
        if not (session and session['stop_event'].is_set()) and collected_reply.strip():
            message_metadata = {"provider": "ollama"}
            if session_id:
                message_metadata["session_id"] = session_id
                
            assistant_message = await store_message_func(
                db=db,
                conversation_id=request.chat_id,
                role="assistant",
                content=collected_reply,
                model_id=model_db_id,
                tokens_used=estimate_tokens(collected_reply),
                message_metadata=message_metadata,
                parent_message_id=user_message_id
            )
            
            # Store in vector DB for future context
            await store_in_vector_db_func(assistant_message.id, request.chat_id, collected_reply)

            # Send final event only if generation completed normally
            final_event = {
                "model": request.model,
                "user_id": request.user_id,
                "chat_id": request.chat_id,
                "reply": "", # Final reply is empty, full content is in raw
                "raw": {
                    "text": collected_reply,
                    "done": True,
                    "total_duration": (datetime.now() - start_time).total_seconds()
                }
            }
            try:
                yield f"data: {json.dumps(final_event)}\n\n"
            except Exception as e:
                logger.warning(f"Failed to send final event for Ollama stream: {e}")

    except GeneratorExit:
        # This occurs when the client disconnects gracefully (e.g., browser tab closed)
        logger.info(f"Client disconnected gracefully from Ollama stream for session {session_id}.")
        if session_id:
            model_manager.stop_user_session(session_id, force=True) # Force stop the session
    except OllamaResponseError as e:
        error_event = {
            "error": f"Ollama API Error: {str(e)}",
            "status_code": getattr(e, "status_code", 500)
        }
        logger.error(f"Ollama Response Error for session {session_id}: {error_event['error']}")
        try:
            yield f"data: {json.dumps(error_event)}\n\n"
        except Exception as e_yield:
            logger.warning(f"Failed to yield error event for Ollama: {e_yield}")
    except Exception as e:
        logger.error(f"Unhandled Ollama processing error for session {session_id}: {str(e)}")
        error_event = {
            "error": f"Internal Server Error: {str(e)}",
            "status_code": 500
        }
        try:
            yield f"data: {json.dumps(error_event)}\n\n"
        except Exception as e_yield:
            logger.warning(f"Failed to yield error event for Ollama: {e_yield}")
    finally:
        # Ensure session is cleaned up after generation, regardless of outcome
        if session_id:
            model_manager.stop_user_session(session_id, force=False) # Normal cleanup

async def process_huggingface_response(
    request: MessageStreamRequest,
    messages: List[Dict],
    db,
    user_message_id: int,
    store_message_func,
    store_in_vector_db_func,
    session_id: str
) -> AsyncGenerator[str, None]:
    """
    Enhanced HuggingFace response processing with proper streaming and force-stop capability.
    """
    collected_reply = ""
    start_time = datetime.now()
    generation_completed_normally = False
    
    # Get session info
    session = model_manager.session_manager.get_session(session_id)
    if not session:
        logger.error(f"Session {session_id} not found for HuggingFace stream.")
        # Attempt to yield an error if the session is somehow missing
        try:
            yield f"data: {json.dumps({'error': 'Session not found', 'status_code': 404})}\n\n"
        except Exception as e:
            logger.warning(f"Failed to yield error for missing session: {e}")
        return
    
    try:
        model, tokenizer = model_manager.get_hf_model(request.model)
        
        # Format messages properly
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
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
            prompt = conversation_text + "<|assistant|>\n"
        
        # Create streamer and inputs
        streamer = TextIteratorStreamer(
            tokenizer, 
            timeout=30.0,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_attention_mask=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
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
        
        # Start the generation in a separate thread
        generation_thread = HuggingFaceGenerationThread(
            model=model,
            generation_kwargs=generation_kwargs,
            streamer=streamer,
            stop_event=session['stop_event'], # Pass the session's stop event
            session_id=session_id
        )
        session['generation_thread'] = generation_thread # Store thread reference in session
        generation_thread.start()
        
        # Consume the stream
        try:
            for new_text in streamer:
                # Check if stop event is set or if client disconnected
                if session['stop_event'].is_set():
                    logger.info(f"Session {session_id} stop event set. Breaking streamer loop.")
                    break
                
                if new_text:
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
                        # This catches client disconnection during streaming
                        logger.info(f"Client disconnected from HuggingFace stream for session {session_id}: {stream_error}")
                        # Signal the generation thread to stop immediately
                        session['stop_event'].set()
                        break # Exit the for loop
                    
                    await asyncio.sleep(0.001) # Small pause to allow other tasks to run
            
            # After streamer loop, check if the thread completed normally
            generation_thread.join(timeout=10) # Wait for generation thread to finish
            if not generation_thread.is_alive() and not session['stop_event'].is_set():
                generation_completed_normally = True
            elif generation_thread.is_alive():
                logger.warning(f"Generation thread for session {session_id} is still alive after streamer loop. Forcing stop.")
                model_manager.stop_user_session(session_id, force=True)

        except GeneratorExit:
            # This is the primary way to detect client disconnection for SSE
            logger.info(f"Client disconnected gracefully from HuggingFace stream for session {session_id}.")
            session['stop_event'].set() # Signal the generation thread to stop
            generation_thread.join(timeout=5) # Give it a moment to react
            model_manager.stop_user_session(session_id, force=True) # Force stop and cleanup
            return # Exit function as client is gone
        except Exception as e_stream:
            logger.error(f"Error during HuggingFace stream iteration for session {session_id}: {e_stream}")
            session['stop_event'].set() # Signal the generation thread to stop
            generation_thread.join(timeout=5) # Give it a moment to react
            
        # Store response if we have content and it wasn't forcibly stopped before completion
        if collected_reply.strip() and not session['stop_event'].is_set():
            model_record = db.query(Model).filter(
                Model.provider == "huggingface", 
                Model.model_id == request.model
            ).first()
            model_db_id = model_record.id if model_record else None

            assistant_message = await store_message_func(
                db=db,
                conversation_id=request.chat_id,
                role="assistant",
                content=collected_reply,
                model_id=model_db_id,
                tokens_used=estimate_tokens(collected_reply),
                message_metadata={
                    "provider": "huggingface", 
                    "session_id": session_id,
                    "completed": generation_completed_normally
                },
                parent_message_id=user_message_id
            )
            
            await store_in_vector_db_func(assistant_message.id, request.chat_id, collected_reply)
            
            # Send final event only if generation completed normally and client is still connected
            if generation_completed_normally:
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
                try:
                    yield f"data: {json.dumps(final_event)}\n\n"
                except Exception as e:
                    logger.warning(f"Failed to send final event for HuggingFace stream: {e}")
        else:
            logger.info(f"Session {session_id} - No content produced or generation was stopped/interrupted.")
        
    except Exception as e:
        logger.error(f"HuggingFace overall processing error for session {session_id}: {str(e)}")
        error_event = {
            "error": f"Internal Server Error: {str(e)}",
            "status_code": 500
        }
        try:
            yield f"data: {json.dumps(error_event)}\n\n"
        except Exception as e_yield:
            logger.warning(f"Failed to yield error event for HuggingFace: {e_yield}")
    finally:
        # Ensure session is cleaned up. If generation completed normally, force=False.
        # Otherwise, force=True to ensure thread shutdown.
        if session_id:
            logger.info(f"HuggingFace final cleanup for session {session_id}. Completed normally: {generation_completed_normally}")
            model_manager.stop_user_session(session_id, force=not generation_completed_normally)


# Utility functions
def cleanup_all_models():
    """Clean up all loaded models"""
    logger.info("Cleaning up all loaded models.")
    model_manager.cleanup_hf_models()
    model_manager.cleanup_ollama()
    model_manager.current_provider = None
    model_manager.current_model_name = None


def get_current_provider_status():
    """Get current provider status for debugging"""
    active_sessions = len(model_manager.session_manager.active_sessions)
    return {
        "current_provider": model_manager.current_provider,
        "current_model": model_manager.current_model_name,
        "hf_model_loaded": model_manager.hf_model is not None,
        "ollama_active": model_manager.ollama_active,
        "active_sessions": active_sessions
    }