from pydantic import BaseModel
from typing import Optional, Any, List

class ToolInput(BaseModel):
    query: str
    max_results: Optional[int] = 5
    safe_search: Optional[bool] = True

class SearchResult(BaseModel):
    title: str
    snippet: str
    url: str
    source: str

class ToolOutput(BaseModel):
    result: Any
    search_results: Optional[List[SearchResult]] = None
    # Add more fields as needed
