from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from app.database import get_db
from schemas.memory import MemoryCreate, MemoryOut
from models.memory import Memory
from chromadb import Chroma
from typing import List

router = APIRouter()

MAX_MEMORIES = 200

@router.post("/", response_model=MemoryOut)
def create_memory(memory: MemoryCreate, chroma_client: Chroma = Depends(get_chroma_client), user_id: str = ""):
    # count existing memories
    # if the count is more than max Remove the oldest memory
    # create generate a consice memory from the prompt given 
    # generate an embedding for the generated memory
    # store the memory in chroma
    # return the memory
    return


@router.get("/", response_model=List[MemoryOut])
def get_memories(chroma_client: Chroma = Depends(get_chroma_client), user_id: str = "") -> List[MemoryOut]:
    # get all memories from chroma
    # return the memories as a list of MemoryOut objects
    return