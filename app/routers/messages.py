import json
from fastapi import APIRouter, status, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import AsyncGenerator, List
import os
import logging
import pprint

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


def _format_content_for_llm(content: str) -> str:
    """
    Format message content for optimal LLM consumption.
    Removes excessive whitespace and standardizes formatting.
    """
    if not content:
        return ""
    
    # Clean up whitespace while preserving structure
    lines = [line.strip() for line in content.split('\n')]
    cleaned_lines = []
    
    for line in lines:
        if line:  # Skip empty lines
            cleaned_lines.append(line)
        elif cleaned_lines and cleaned_lines[-1]:  # Preserve single line breaks
            cleaned_lines.append("")
    
    # Join and remove excessive spacing
    formatted = '\n'.join(cleaned_lines)
    
    # Remove multiple consecutive spaces but preserve intentional formatting
    import re
    formatted = re.sub(r' {3,}', '  ', formatted)  # Max 2 consecutive spaces
    formatted = re.sub(r'\n{3,}', '\n\n', formatted)  # Max 2 consecutive newlines
    
    return formatted.strip()

async def message_streamer(
    request: MessageStreamRequest,
    db: Session = Depends(get_db)
) -> AsyncGenerator[str, None]:
    """
    Stream a model response with optimized message formatting and context management.
    """
    # Verify conversation exists
    conversation = db.query(Conversation).filter(Conversation.id == request.chat_id).first()
    if not conversation:
        yield f"data: {json.dumps({'error': 'Conversation not found', 'status_code': 404})}\n\n"
        return

    from app.utils.token import estimate_tokens
    
    # Store user message in database
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

    # Get last few messages for immediate context
    recent_messages = db.query(Message).filter(
        Message.conversation_id == request.chat_id
    ).order_by(Message.created_at.desc()).offset(1).limit(6).all()


    # === OPTIMIZED MESSAGE CONSTRUCTION ===
    messages = [{
        "role": "system",
        "content": "You are a helpful AI assistant. Use the conversation history and provided context to give accurate, relevant responses and keep the tone and language human like unless the user says otherwise."
    }]

    # Deduplicate context messages (remove ones already in recent messages)
    recent_content = {msg.content.strip().lower() for msg in recent_messages}
    unique_context = [
        msg for msg in context_messages 
        if msg["content"].strip().lower() not in recent_content
    ]

    # pprint.pprint(context_messages, depth=4)
    # pprint.pprint(unique_context, depth=4)

    # Add unique context messages with optimized formatting
    for msg in unique_context:
        content = _format_content_for_llm(msg["content"])
        messages.append({
            "role": msg["role"],
            "content": content
        })

    # Add recent messages in chronological order with optimized formatting
    for msg in reversed(recent_messages):
        content = _format_content_for_llm(msg.content)
        messages.append({
            "role": msg.role,
            "content": content
        })

    # Add current user message with optimized formatting
    messages.append({
        "role": "user",
        "content": _format_content_for_llm(request.message)
    })

    # Use existing trim_messages function
    messages = trim_messages(messages, max_tokens=MAX_TOKENS)

    # pprint.pprint({"this is complete message": messages},depth=4)

    # Debug logging
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Sending {len(messages)} messages to {request.provider} model {request.model}")

    # === STREAM RESPONSE ===
    if request.provider == "ollama":
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
