from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Literal, Union
from datetime import datetime


class MessageBase(BaseModel):
    conversation_id: int
    role: Literal["user", "assistant", "system"]
    content: str
    tokens_used: Optional[int] = None
    model_id: Optional[int] = None
    message_metadata: Optional[Dict[str, Any]] = None
    parent_message_id: Optional[int] = None


class MessageCreate(MessageBase):
    pass


class MessageUpdate(BaseModel):
    content: Optional[str] = None
    tokens_used: Optional[int] = None
    message_metadata: Optional[Dict[str, Any]] = None


class MessageOut(MessageBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class MessageStreamRequest(BaseModel):
    chat_id: str  
    # Accept numeric user IDs for registered users or string identifiers (e.g., "guest-<suffix>") for temporary guests
    user_id: Union[int, str]  
    model: str
    message: str
    provider: Literal["ollama", "huggingface"] = "ollama"
    stream: bool = True
    context_strategy: Literal["hybrid", "vectordb", "cache", "web_search"] = "hybrid"
    optimize_context: bool = True
    max_context_docs: int = 15
    web_search: bool = False
    web_search_query: Optional[str] = None
    max_search_results: int = 5
