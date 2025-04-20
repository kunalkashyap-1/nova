from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def get_settings():
    return

@router.put("/")
def update_settings():
    return
