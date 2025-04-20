from fastapu import APIRouter, HTTPException
from app.schemas.tool import ToolInput, ToolOutput

router = APIRouter()

@router.post("memory_lookup", response_model= ToolOutput)
def memory_lookup(tool_input: ToolInput, chroma_client: Chroma = Depends(get_chroma_client), user_id: str = ""):
    # get all memories from chroma
    # return the memories as a list of MemoryOut objects (ToolOutput)
    return


@router.post("/search", response_model= ToolOutput)
def search(tool_input: ToolInput):
    # use beutiful soup to search the web and get data from sites from top 5 results 
    # return the data as a list of ToolOutput objects
    return