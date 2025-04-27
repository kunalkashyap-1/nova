from fastapi import APIRouter
from pydantic import BaseModel, EmailStr

class UserOut(BaseModel):
    id: int
    username: str
    email: EmailStr

router = APIRouter()

@router.post("/register")
def register():
    return 


@router.post("/login")
def login():
    return 
    
@router.post("/logout")
def logout():
    return 
    
@router.post("/forgot-password")
def forgot_password():
    return 

@router.post("/reset-password")
def reset_password():
    return 
    
@router.get("/me", response_model= UserOut)
def get_me(user_id:str = "" ):
    return 

@router.put("/me")
def update_user():
    return