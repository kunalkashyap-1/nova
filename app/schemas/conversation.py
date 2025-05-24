from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ConversationCreate(BaseModel):
    title: str
    user_id: int
    model_id: Optional[int] = None
    system_prompt: Optional[str] = None
    folder_id: Optional[int] = None

class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    is_pinned: Optional[bool] = None
    is_archived: Optional[bool] = None
    system_prompt: Optional[str] = None
    folder_id: Optional[int] = None

class ConversationOut(BaseModel):
    id: int
    title: str
    user_id: int
    model_id: Optional[int] = None
    system_prompt: Optional[str] = None
    folder_id: Optional[int] = None
    is_pinned: bool
    is_archived: bool
    last_message_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True 