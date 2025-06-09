from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Response, Request, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from app.database import get_db
from sqlalchemy.orm import Session
from app.models.user import User
import os
import uuid
import shutil
from pathlib import Path
import mimetypes
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])

# Security Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7  
UPLOAD_DIR = Path("uploads/profile_pictures")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token", auto_error=False)

# Pydantic Models
class UserBase(BaseModel):
    email: EmailStr
    username: str
    full_name: str
    bio: Optional[str] = ""
    preferred_language: Optional[str] = ""
    timezone: Optional[str] = ""

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    profile_picture: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

class LoginRequest(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

class TokenData(BaseModel):
    user_id: Optional[int] = None
    username: Optional[str] = None
    full_name: Optional[str] = None
    email: Optional[str] = None

# Helper Functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(user: User, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT token with user data"""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    
    # Include user data in token payload
    to_encode = {
        "sub": str(user.id),  # Subject (user ID)
        "user_id": user.id,
        "username": user.username,
        "full_name": user.full_name,
        "email": user.email,
        "profile_picture": user.profile_picture,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access_token"
    }
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_token_from_cookie(request: Request) -> Optional[str]:
    """Extract token from HTTP-only cookie"""
    return request.cookies.get("auth_token")

async def get_current_user_from_cookie(request: Request, db: Session = Depends(get_db)) -> Optional[User]:
    """Get current user from cookie token"""
    token = get_token_from_cookie(request)
    if not token:
        return None
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("user_id")
        if user_id is None:
            return None
        
        # Verify user still exists in database
        user = db.query(User).filter(User.id == user_id).first()
        return user
    except JWTError:
        return None

async def get_current_user_required(request: Request, db: Session = Depends(get_db)) -> User:
    """Get current user (required - raises exception if not authenticated)"""
    user = await get_current_user_from_cookie(request, db)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

def save_profile_picture(file: UploadFile, user_id: int) -> str:
    """Save uploaded profile picture and return filename"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Generate unique filename
    file_extension = file.filename.split('.')[-1].lower()
    if file_extension not in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    filename = f"user_{user_id}_{uuid.uuid4().hex}.{file_extension}"
    file_path = UPLOAD_DIR / filename
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return filename

def set_auth_cookie(response: Response, token: str) -> None:
    """Set HTTP-only authentication cookie"""
    response.set_cookie(
        key="auth_token",
        value=token,
        max_age=ACCESS_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,  # 7 days in seconds
        httponly=True,  # Prevent XSS attacks
        secure=True,    # Only send over HTTPS in production
        samesite="lax"  # CSRF protection
    )

# API Routes
@router.post("/register", response_model=UserResponse)
async def register(
    response: Response,
    full_name: str = Form(...),
    email: EmailStr = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    bio: Optional[str] = Form(""),
    preferred_language: Optional[str] = Form(""),
    timezone: Optional[str] = Form(""),
    profile_picture: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    """Register a new user"""
    # Check if user already exists
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    
    # Create new user
    hashed_password = get_password_hash(password)
    db_user = User(
        email=email,
        username=username,
        full_name=full_name,
        password_hash=hashed_password,
        bio=bio or "",
        preferred_language=preferred_language or "",
        timezone=timezone or "",
        is_guest=False
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Handle profile picture upload
    if profile_picture and profile_picture.filename:
        try:
            filename = save_profile_picture(profile_picture, db_user.id)
            db_user.profile_picture = filename
            db.commit()
            db.refresh(db_user)
        except Exception as e:
            # Log error but don't fail registration
            print(f"Profile picture upload failed: {e}")
    
    # Create token and set cookie
    access_token = create_access_token(db_user)
    set_auth_cookie(response, access_token)
    
    return db_user

@router.post("/login", response_model=UserResponse)
async def login(
    response: Response,
    login_data: LoginRequest,
    db: Session = Depends(get_db)
):
    """Login user with username and password"""
    user = db.query(User).filter(User.username == login_data.username).first()
    
    if not user or not verify_password(login_data.password, user.password_hash):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password"
        )
    
    # Create token and set cookie
    access_token = create_access_token(user)
    set_auth_cookie(response, access_token)
    
    return user

@router.post("/token", response_model=Token)
async def login_for_access_token(
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """OAuth2 compatible login endpoint"""
    user = db.query(User).filter(User.username == form_data.username).first()
    
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(user)
    set_auth_cookie(response, access_token)
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user
    }

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user_required)
):
    """Get current user information"""
    return current_user

@router.get("/check")
async def check_auth_status(
    request: Request,
    db: Session = Depends(get_db)
):
    """Check if user is authenticated"""
    user = await get_current_user_from_cookie(request, db)
    if user:
        return {
            "authenticated": True,
            "user": UserResponse.from_orm(user)
        }
    return {"authenticated": False}

@router.post("/logout")
async def logout(response: Response):
    """Logout user by clearing auth cookie"""
    response.delete_cookie(
        key="auth_token",
        httponly=True,
        secure=True,
        samesite="lax"
    )
    return {"message": "Successfully logged out"}

@router.post("/refresh")
async def refresh_token(
    response: Response,
    request: Request,
    db: Session = Depends(get_db)
):
    """Refresh authentication token"""
    user = await get_current_user_from_cookie(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Create new token
    new_token = create_access_token(user)
    set_auth_cookie(response, new_token)
    
    return {"message": "Token refreshed successfully"}

@router.put("/me", response_model=UserResponse)
async def update_user_profile(
    full_name: Optional[str] = Form(None),
    bio: Optional[str] = Form(None),
    preferred_language: Optional[str] = Form(None),
    timezone: Optional[str] = Form(None),
    profile_picture: Optional[UploadFile] = File(None),
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """Update user profile"""
    # Update fields if provided
    if full_name is not None:
        current_user.full_name = full_name
    if bio is not None:
        current_user.bio = bio
    if preferred_language is not None:
        current_user.preferred_language = preferred_language
    if timezone is not None:
        current_user.timezone = timezone
    
    # Handle profile picture update
    if profile_picture and profile_picture.filename:
        try:
            # Delete old profile picture if exists
            if current_user.profile_picture:
                old_file_path = UPLOAD_DIR / current_user.profile_picture
                if old_file_path.exists():
                    old_file_path.unlink()
            
            # Save new profile picture
            filename = save_profile_picture(profile_picture, current_user.id)
            current_user.profile_picture = filename
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Profile picture update failed: {str(e)}")
    
    db.commit()
    db.refresh(current_user)
    
    return current_user

@router.post("/forgot-password")
async def forgot_password(email: EmailStr, db: Session = Depends(get_db)):
    """Send password reset email"""
    user = db.query(User).filter(User.email == email).first()
    if not user:
        # Don't reveal if email exists for security
        return {"message": "If the email exists, a reset link has been sent"}
    
    # TODO: Implement email sending logic
    # Generate reset token and send email
    
    return {"message": "If the email exists, a reset link has been sent"}

@router.post("/reset-password")
async def reset_password(
    token: str,
    new_password: str,
    db: Session = Depends(get_db)
):
    """Reset password using reset token"""
    # TODO: Implement password reset logic
    # Verify reset token and update password
    
    return {"message": "Password reset successfully"}

@router.get("/media/{filename}")
async def get_media_file(filename: str):
    """Serve uploaded media files (profile pictures, etc.)"""
    # Construct the file path
    file_path = UPLOAD_DIR / filename
    
    # Check if file exists
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Security check: ensure the file is within the upload directory
    # This prevents directory traversal attacks
    try:
        file_path.resolve().relative_to(UPLOAD_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Get MIME type for proper content-type header
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if mime_type is None:
        mime_type = "application/octet-stream"
    
    # Return the file
    return FileResponse(
        path=str(file_path),
        media_type=mime_type,
        filename=filename
    )