import json
from fastapi import APIRouter, status, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import AsyncGenerator, List
import os
import logging

# Internal imports
from app.database import get_db
from app.models.conversation import Conversation
from app.models.message import Message
from app.utils.redis import cache_delete

# Import modularized services
from app.services.message_service import (
    store_message_in_db, 
    store_in_vector_db,
    get_chat_history,
    get_context_for_query
)
from app.utils.token import trim_messages
from app.utils.message_providers import (
    process_ollama_response,
    process_huggingface_response,
    ollama_client
)
from app.schemas.message import MessageOut, MessageStreamRequest

# Environment variables
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "30000"))
MAX_HISTORY = int(os.getenv("MAX_HISTORY_ITEMS", "15"))

# Configure logging
logger = logging.getLogger(__name__)

# Define router
router = APIRouter(prefix="/api/v1/messages", tags=["Messages"])


async def message_streamer(
    request: MessageStreamRequest,
    db: Session = Depends(get_db)
) -> AsyncGenerator[str, None]:
    """
    Stream a model response for a given message request.
    
    This function handles the core message processing workflow:
    1. Verifies the conversation exists
    2. Stores the user message in the database and vector store
    3. Retrieves relevant context using the requested strategy
    4. Gets last 2 messages for immediate context
    5. Streams the model response and stores it
    
    Args:
        request: The message request with model details and content
        db: Database session
        
    Yields:
        Server-sent events with model responses
    """
    # Verify conversation exists
    conversation = db.query(Conversation).filter(Conversation.id == request.chat_id).first()
    if not conversation:
        error_event = {
            "error": "Conversation not found",
            "status_code": 404
        }
        yield f"data: {json.dumps(error_event)}\n\n"
        return
        
    # Store user message in database
    from app.utils.token import estimate_tokens
    user_message = await store_message_in_db(
        db=db,
        conversation_id=request.chat_id,
        role="user",
        content=request.message,
        tokens_used=estimate_tokens(request.message)
    )
    
    # Store in vector DB for future context
    await store_in_vector_db(user_message.id, request.chat_id, request.message)
    
    # Get relevant context using the specified strategy
    context_messages = await get_context_for_query(
        conversation_id=request.chat_id,
        query=request.message,
        strategy=request.context_strategy,
        optimize=request.optimize_context,
        max_docs=request.max_context_docs
    )
    
    # Get last 2 messages for immediate context
    recent_messages = db.query(Message).filter(
        Message.conversation_id == request.chat_id
    ).order_by(Message.created_at.desc()).limit(2).all()
    
    # Format recent messages with clear role indicators
    recent_context = []
    for msg in reversed(recent_messages):  # Reverse to get chronological order
        recent_context.append({
            "role": msg.role,
            "content": f"{msg.role.capitalize()}: {msg.content}"
        })
    
    # Add current user message with consistent formatting
    recent_context.append({
        "role": "user",
        "content": f"User: {request.message}"
    })
    
    # Format context messages consistently
    formatted_context = []
    for msg in context_messages:
        if msg["role"] == "system":
            # Format system messages with clear context markers
            formatted_context.append({
                "role": "system",
                "content": f"Context Information:\n{msg['content']}"
            })
        else:
            # Format other messages consistently
            formatted_context.append({
                "role": msg["role"],
                "content": f"{msg['role'].capitalize()}: {msg['content']}"
            })
    
    # Combine vector store context with recent messages
    # Put recent messages first to maintain conversation flow
    messages = recent_context + formatted_context
    
    # Add a system message at the start to set the context
    # messages.insert(0, {
    #     "role": "system",
    #     "content": "You are a helpful AI assistant. Use the provided context and conversation history to provide accurate and relevant responses."
    # })
    
    # Trim messages to fit token budget
    messages = trim_messages(messages, max_tokens=MAX_TOKENS)
    
    # Log conversation for debugging
    logger.debug(f"Sending {len(messages)} messages to {request.provider} model {request.model}")
    
    # Process based on provider
    if request.provider == "ollama":
        # Get response from Ollama
        stream = await ollama_client.chat(
            model=request.model,
            messages=messages,
            stream=True,
        )
        async for chunk in process_ollama_response(
            stream=stream, 
            request=request, 
            db=db, 
            user_message_id=user_message.id,
            store_message_func=store_message_in_db,
            store_in_vector_db_func=store_in_vector_db
        ):
            yield chunk
            
    elif request.provider == "huggingface":
        # Get response from HuggingFace
        async for chunk in process_huggingface_response(
            request=request, 
            messages=messages, 
            db=db, 
            user_message_id=user_message.id,
            store_message_func=store_message_in_db,
            store_in_vector_db_func=store_in_vector_db
        ):
            yield chunk
            
    else:
        # Provider not supported
        yield f"data: {json.dumps({'error': f'Provider {request.provider} not supported'})}\n\n"


@router.post("/", status_code=status.HTTP_200_OK, summary="Stream a chat message reply")
async def stream_message_reply(
    body: MessageStreamRequest,
    db: Session = Depends(get_db)
):
    """
    Stream a reply to a user message using the specified model and provider.
    
    This endpoint:
    - Creates a streaming response with the model's reply
    - Stores both the user message and model response in the database
    - Indexes messages in the vector database for future context
    
    Request body parameters:
    - chat_id: ID of the conversation
    - user_id: ID of the user sending the message
    - model: ID of the model to use for generation
    - message: Content of the user's message
    - provider: "ollama" or "huggingface"
    - stream: Whether to stream the response (default: True)
    - context_strategy: Strategy to use for retrieving context
    - optimize_context: Whether to optimize retrieved context
    
    Returns:
        A streaming response with server-sent events containing the model's reply
    """
    return StreamingResponse(
        message_streamer(body, db),
        media_type="text/event-stream",
    )


@router.get("/conversation/{conversation_id}", status_code=status.HTTP_200_OK, summary="Get conversation messages")
async def get_conversation_messages(conversation_id: int, db: Session = Depends(get_db)):
    """
    Get all messages for a specific conversation.
    
    This endpoint retrieves messages in chronological order (oldest to newest).
    
    Path parameters:
    - conversation_id: ID of the conversation to retrieve messages for
    
    Returns:
        A JSON object with a 'messages' array containing all messages in the conversation
    """
    messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.created_at.asc()).all()
    
    return {"messages": messages}


@router.delete("/message/{message_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a message")
async def delete_message(message_id: int, db: Session = Depends(get_db)):
    """
    Delete a specific message by ID.
    
    This endpoint:
    - Removes the message from the database
    - Removes the message from the vector database
    - Invalidates related caches
    
    Path parameters:
    - message_id: ID of the message to delete
    
    Returns:
        204 No Content on success
    
    Raises:
        404 Not Found: If the message doesn't exist
    """
    message = db.query(Message).filter(Message.id == message_id).first()
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    
    # Delete from database
    db.delete(message)
    db.commit()
    
    # Delete from vector DB
    from app.routers.memories import get_chroma_client
    collection = get_chroma_client().get_or_create_collection(name="chat_context")
    
    try:
        collection.delete(ids=[f"msg:{message_id}"])
    except Exception as e:
        logger.error(f"Error deleting from vector DB: {str(e)}")
    
    # Invalidate cache
    await cache_delete(f"history:{message.conversation_id}")


# Compatibility endpoint for ?conversation_id=...
@router.get("/", response_model=List[MessageOut])
def get_messages(conversation_id: int = None, db: Session = Depends(get_db)):
    if conversation_id is not None:
        return db.query(Message).filter(Message.conversation_id == conversation_id).all()
    return []


# Alias for /stream
@router.post("/stream", status_code=status.HTTP_200_OK)
async def stream_message_reply_alias(body: MessageStreamRequest, db: Session = Depends(get_db)):
    return await stream_message_reply(body, db)
