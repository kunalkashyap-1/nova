from fastapi import APIRouter, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncGenerator, Literal
import httpx
import json

router = APIRouter()

class MessageStreamRequest(BaseModel):
    user_id: str
    model: str
    message: str
    provider: Literal["ollama"] = "ollama"  # default, but extensible

async def generate_stream_ollama(payload: dict) -> AsyncGenerator[dict, None]:
    ollama_url = "http://localhost:11434/api/chat"
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", ollama_url, json=payload) as response:
            async for line in response.aiter_lines():
                if line.strip():
                    try:
                        data = json.loads(line)
                        yield data
                    except Exception:
                        yield {"error": "Malformed chunk from Ollama", "raw": line}

async def message_streamer(request: MessageStreamRequest) -> AsyncGenerator[str, None]:
    if request.provider == "ollama":
        payload = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.message}],
            "stream": True,
            "user_id": request.user_id
        }
        async for chunk in generate_stream_ollama(payload):
            event = {
                # "provider": request.provider,
                "model": request.model,
                "user_id": request.user_id,
                # "message": request.message,
                "reply": chunk.get("message", {}).get("content") if isinstance(chunk, dict) else chunk,
                "raw": chunk
            }
            yield f"data: {json.dumps(event)}\n\n"
    else:
        # Placeholder for other providers (e.g., HuggingFace)
        yield f"data: {json.dumps({'error': 'Provider not supported'})}\n\n"

@router.post("/", status_code=status.HTTP_200_OK)
async def stream_message_reply(body: MessageStreamRequest):
    return StreamingResponse(message_streamer(body), media_type="text/event-stream")
