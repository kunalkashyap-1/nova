from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncGenerator, Literal, List, Dict
from ollama import AsyncClient, ResponseError, ChatResponse
import json
import os

from app.routers.memories import get_chroma_client
# from app.utils.redis import cache_get, cache_set, cache_append_to_list, cache_get_list

MAX_HISTORY = 15
MAX_TOKENS = 30000

router = APIRouter()

# Initialize clients
collection = get_chroma_client().get_or_create_collection(name="chat_context")
ollama_client = AsyncClient(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))


class MessageStreamRequest(BaseModel):
    chat_id: str
    user_id: str
    model: str
    message: str
    provider: Literal["ollama", "huggingface"] = "ollama"


def trim_messages(messages: List[Dict], max_tokens: int) -> List[Dict]:
    trimmed, total = [], 0
    for msg in reversed(messages):
        token_est = len(msg["content"].split())
        if total + token_est > max_tokens:
            break
        trimmed.insert(0, msg)
        total += token_est
    return trimmed


async def message_streamer(request: MessageStreamRequest) -> AsyncGenerator[str, None]:
    # chat_key = f"history:{request.chat_id}"
    # ctx_cache_key = f"context:{request.chat_id}:{hash(request.message)}"

    # # Append user message to session chat history
    # await cache_append_to_list(chat_key, {"role": "user", "content": request.message})

    # Try cached context first (from Redis)
    # cached_ctx = await cache_get(ctx_cache_key)
    # if cached_ctx:
    #     context_messages = json.loads(cached_ctx)
    # else:
    results = collection.query(
        query_texts=[request.message],
        n_results=MAX_HISTORY,
        include=["documents"],
        where={"chat_id": request.chat_id}
    )
    docs = results.get("documents", [[]])[0]
    context_summary = "\n".join(docs)
    context_messages = [{"role": "system", "content": f"Relevant context:\n{context_summary}"}]
    #     await cache_set(ctx_cache_key, json.dumps(context_messages), ex=120)

    # Get recent chat history
    # chat_history = (await cache_get_list(chat_key))[-MAX_HISTORY:]
    chat_history = []  # Empty for now since Redis is disabled

    # Merge and trim for token budget
    messages = trim_messages(context_messages + chat_history, max_tokens=MAX_TOKENS)
    print("messages", messages)
    print("messages length", len(messages))

    if request.provider == "ollama":
        try:
            collected_reply = ""
            stream = await ollama_client.chat(
                model=request.model,
                messages=messages,
                stream=True,
            )
            async for chunk in stream:
                if isinstance(chunk, ChatResponse):
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
                    "raw": raw,
                }
                yield f"data: {json.dumps(event)}\n\n"

            # Store full assistant reply
            # await cache_append_to_list(chat_key, {"role": "assistant", "content": collected_reply})
            collection.add(
                documents=[collected_reply],
                metadatas=[{"chat_id": request.chat_id}],
                ids=[f"{request.chat_id}:{hash(collected_reply)}"]
            )

        except ResponseError as e:
            yield f"data: {json.dumps({'error': str(e), 'status_code': e.status_code})}\n\n"
    else:
        yield f"data: {json.dumps({'error': 'Provider not supported'})}\n\n"


@router.post("/", status_code=status.HTTP_200_OK)
async def stream_message_reply(body: MessageStreamRequest):
    return StreamingResponse(
        message_streamer(body),
        media_type="text/event-stream",
    )
