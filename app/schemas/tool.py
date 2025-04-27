from pydantic import BaseModel
from typing import Optional, Any

class ToolInput(BaseModel):
    query: str
    user_id: Optional[str] = None
    # Add more fields as needed

class ToolOutput(BaseModel):
    result: Any
    # Add more fields as needed
