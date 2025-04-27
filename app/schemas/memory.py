from pydantic import BaseModel
from typing import Optional

class MemoryCreate(BaseModel):
    # Add your fields here
    title: str
    content: str
    created_at: Optional[str] = None

class MemoryOut(BaseModel):
    id: int
    title: str
    content: str
    created_at: Optional[str] = None
