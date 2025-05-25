"""
Database and caching services for message operations.
"""
import json
from typing import List, Dict, Optional
from datetime import datetime
import os
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from app.models.message import Message
from app.models.conversation import Conversation
from app.utils.redis import (
    cache_get, cache_set, cache_delete, cache_hset, cache_hget
)
from app.routers.memories import get_chroma_client

# Cache TTLs - these can be moved to environment variables
CHAT_HISTORY_CACHE_TTL = int(os.getenv("CHAT_HISTORY_CACHE_TTL", "3600"))  # 1 hour
CONTEXT_CACHE_TTL = int(os.getenv("CONTEXT_CACHE_TTL", "600"))  # 10 minutes

# Maximum history items
MAX_HISTORY = int(os.getenv("MAX_HISTORY_ITEMS", "15"))

# Initialize ChromaDB collection
collection = get_chroma_client().get_or_create_collection(name="chat_context")


async def store_message_in_db(
    db: Session,
    conversation_id: str,
    role: str,
    content: str,
    model_id: Optional[int] = None,
    tokens_used: Optional[int] = None,
    message_metadata: Optional[Dict] = None,
    parent_message_id: Optional[int] = None
) -> Message:
    """
    Store a message in the database and update related caches.
    
    Args:
        db: Database session
        conversation_id: ID of the conversation (UUID string)
        role: Message role ('user', 'assistant', 'system')
        content: Message content
        model_id: Optional ID of the model that generated the message
        tokens_used: Optional count of tokens used
        message_metadata: Optional additional metadata
        parent_message_id: Optional ID of parent message
        
    Returns:
        Created message object
    """
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        model_id=model_id,
        tokens_used=tokens_used,
        message_metadata=message_metadata,
        parent_message_id=parent_message_id
    )
    
    db.add(message)
    db.commit()
    db.refresh(message)
    
    # Update conversation's last_message_at timestamp
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if conversation:
        conversation.last_message_at = func.now()
        db.commit()
    
    # Update cache to include this new message
    await update_history_cache(conversation_id, message)
    
    return message


async def update_history_cache(conversation_id: str, message: Message):
    """
    Update the cached history with a new message.
    
    Args:
        conversation_id: ID of the conversation (UUID string)
        message: Message object to add to cache
    """
    cache_key = f"history:{conversation_id}"
    
    # Get current cache
    current_history = await cache_get(cache_key)
    
    if current_history:
        # Add to existing history
        history = json.loads(current_history)
        history.append({
            "role": message.role,
            "content": message.content,
            "id": message.id
        })
        
        # Keep only most recent MAX_HISTORY messages
        if len(history) > MAX_HISTORY:
            history = history[-MAX_HISTORY:]
            
        await cache_set(cache_key, json.dumps(history), ex=CHAT_HISTORY_CACHE_TTL)
    else:
        # Create new cache entry
        await cache_set(
            cache_key, 
            json.dumps([{
                "role": message.role,
                "content": message.content,
                "id": message.id
            }]), 
            ex=CHAT_HISTORY_CACHE_TTL
        )


async def get_chat_history(db: Session, conversation_id: str, limit: int = MAX_HISTORY) -> List[Dict]:
    """
    Get recent chat history from database with caching.
    
    Args:
        db: Database session
        conversation_id: ID of the conversation (UUID string)
        limit: Maximum number of messages to retrieve
        
    Returns:
        List of message dictionaries
    """
    # Cache key for this conversation's history
    cache_key = f"history:{conversation_id}"
    
    # Try to get from cache first
    cached_history = await cache_get(cache_key)
    if cached_history:
        return json.loads(cached_history)
    
    # If not in cache, get from database
    messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.created_at.desc()).limit(limit).all()
    
    # Format as list of dicts for the LLM
    history = []
    for msg in reversed(messages):  # Reverse to get chronological order
        history.append({
            "role": msg.role,
            "content": msg.content,
            "id": msg.id
        })
    
    # Cache for future use
    await cache_set(cache_key, json.dumps(history), ex=CHAT_HISTORY_CACHE_TTL)
    
    return history


async def store_in_vector_db(message_id: int, conversation_id: str, content: str):
    """
    Store message content in the vector database for retrieval.
    
    Args:
        message_id: ID of the message
        conversation_id: ID of the conversation (UUID string)
        content: Message content
    """
    collection.add(
        documents=[content],
        metadatas=[{
            "chat_id": conversation_id,  
            "message_id": message_id,
            "timestamp": datetime.now().isoformat()
        }],
        ids=[f"msg:{message_id}"]  
    )


async def get_context_for_query(
    conversation_id: int, 
    query: str, 
    model_name: str = None,
    strategy: str = "hybrid",
    optimize: bool = True,
    max_docs: int = MAX_HISTORY,
    web_search: bool = False,
    web_search_query: Optional[str] = None,
    max_search_results: int = 5
) -> List[Dict]:
    """
    Get relevant context for a query using the specified strategy.
    
    Args:
        conversation_id: ID of the conversation
        query: The search query
        model_name: Name of the model (for tokenization)
        strategy: Strategy to use ('vectordb', 'cache', 'hybrid', 'web_search')
        optimize: Whether to optimize context
        max_docs: Maximum number of documents to retrieve
        web_search: Whether to include web search results
        web_search_query: Optional specific query for web search (defaults to main query)
        max_search_results: Maximum number of web search results to include
        
    Returns:
        List of context messages
    """
    import hashlib
    from app.utils.token import optimize_context
    import time
    
    start_time = time.time()
    
    # Generate a cache key for this specific query in this conversation
    query_hash = hashlib.md5(query.encode()).hexdigest()[:10]
    ctx_cache_key = f"context:{conversation_id}:{query_hash}"
    
    # Strategy 1: Try cached context if strategy allows
    if strategy in ["cache", "hybrid"]:
        cached_ctx = await cache_get(ctx_cache_key)
        if cached_ctx:
            try:
                context_data = json.loads(cached_ctx)
                # Add cache hit telemetry
                await cache_hset("telemetry:cache_hits", f"{int(time.time())}", "context_query")
                return context_data
            except json.JSONDecodeError:
                # Handle corrupt cache data
                await cache_delete(ctx_cache_key)
    
    # Strategy 2: Use vector DB
    if strategy in ["vectordb", "hybrid"]:
        try:
            # For more accurate search, enrich the query with related terms
            enriched_query = query
            
            # Use metadata filtering to scope properly
            where_clause = {"chat_id": str(conversation_id)}
            
            # Execute the vector search
            vector_start = time.time()
            results = collection.query(
                query_texts=[enriched_query],
                n_results=max_docs,
                include=["documents", "metadatas", "distances"],
                where=where_clause
            )
            vector_time = time.time() - vector_start
            
            # Log search performance for monitoring
            if vector_time > 1.0:  # Log slow searches
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Slow vector search: {vector_time:.2f}s for query in conversation {conversation_id}")
            
            docs = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0] if "distances" in results else None
            
            # Enhanced context with metadata
            docs_with_metadata = []
            for i, doc in enumerate(docs):
                if i < len(metadatas):
                    # Add message ID and timestamp to track source
                    metadata = metadatas[i]
                    distance = distances[i] if distances and i < len(distances) else None
                    
                    # Create a context item with source information
                    docs_with_metadata.append({
                        "content": doc,
                        "metadata": metadata,
                        "relevance": 1.0 - (distance / 2.0) if distance is not None else 0.5  # Normalize distance to relevance score
                    })
            
            # Apply optimization if requested, using the enhanced model-aware optimizer
            if optimize and docs:
                # Extract just the content for optimization
                content_only = [item["content"] for item in docs_with_metadata]
                optimized_content = optimize_context(content_only, query, model_name=model_name)
                
                # Match optimized content back to original metadata
                optimized_with_metadata = []
                for opt_doc in optimized_content:
                    # Find matching original document
                    for doc_meta in docs_with_metadata:
                        if opt_doc in doc_meta["content"]:
                            optimized_with_metadata.append(doc_meta)
                            break
                    else:
                        # If no exact match (due to truncation), add without metadata
                        optimized_with_metadata.append({"content": opt_doc, "metadata": {}})
                
                docs_with_metadata = optimized_with_metadata
            
            # Format as context message with enhanced information
            if docs_with_metadata:
                # Create a richer context block with citation information
                citations = []
                context_blocks = []
                
                for i, doc_meta in enumerate(docs_with_metadata):
                    doc = doc_meta["content"]
                    meta = doc_meta.get("metadata", {})
                    message_id = meta.get("message_id")
                    timestamp = meta.get("timestamp")
                    
                    # Add citation marker and index
                    citation_id = f"[{i+1}]"
                    context_blocks.append(f"{citation_id} {doc}")
                    
                    # Add to citations list for reference
                    if message_id:
                        citations.append(f"{citation_id} Message ID: {message_id}, Time: {timestamp if timestamp else 'unknown'}")
                
                # Join all context with citations
                context_with_citations = "\n\n".join(context_blocks)
                if citations:
                    context_with_citations += "\n\n" + "References:\n" + "\n".join(citations)
                
                context_messages = [{
                    "role": "system", 
                    "content": f"Relevant context from previous conversations:\n{context_with_citations}"
                }]
                
                # Cache this context for future similar queries (with TTL based on importance)
                cache_ttl = CONTEXT_CACHE_TTL 
                if len(docs) > 5:  # More docs = more valuable context
                    cache_ttl = cache_ttl * 2  # Longer TTL for valuable context
                
                await cache_set(ctx_cache_key, json.dumps(context_messages), ex=cache_ttl)
                
                # Track performance for monitoring
                end_time = time.time()
                await cache_hset("telemetry:context_timing", f"{int(end_time)}", str(end_time - start_time))
                
                return context_messages
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error retrieving context: {str(e)}")
            
            # Fallback to bare minimum context (just query)
            return [{
                "role": "system",
                "content": f"You're answering the following question: {query}"
            }]
            
    # Strategy 3: Web search (if enabled)
    if strategy == "web_search" or web_search:
        from app.services.web_search_service import get_web_search_context
        search_query = web_search_query if web_search_query else query
        
        # Get web search results
        web_context = await get_web_search_context(search_query, max_results=max_search_results)
        
        # If we also have context from other strategies, combine them
        if context_messages:
            return context_messages + web_context
        
        return web_context
    
    # No context found
    return []
