from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import time

# Application routers
from app.routers import auth, commands, tasks, memories, tools, settings, messages, models
from app.routers import conversations
from app.nova_logger import logger

# Load environment variables
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
ENABLE_REQUEST_VALIDATION = os.getenv("ENABLE_REQUEST_VALIDATION", "True").lower() == "true"
APP_VERSION = os.getenv("APP_VERSION", "0.1.0")

# Initialize FastAPI with configuration
app = FastAPI(
    title="Nova", 
    description="An advanced AI chat assistant with RAG capabilities", 
    version=APP_VERSION,
    debug=DEBUG,
    docs_url="/api/docs" if DEBUG else None,  # Only show docs in debug mode
    redoc_url="/api/redoc" if DEBUG else None,
)

logger.info(f"Server starting... Version: {APP_VERSION}, Debug: {DEBUG}")

# CORS configuration - supports development and production modes
allowed_origins = [
    "http://localhost:5173",          # SvelteKit dev server
    "http://localhost:4173",          # SvelteKit preview server
    "http://localhost:8000",          # FastAPI dev server
    "http://localhost:3000",          # Alternative dev server
]

extra_origins = os.getenv("CORS_ORIGINS", "")
if extra_origins:
    allowed_origins.extend(extra_origins.split(","))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Authorization"],
    max_age=3600,
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    log_dict = {
        "request": {
            "url": request.url,
            "method": request.method,
            # "headers": dict(request.headers),
            # "body": request.body,
        }
    }
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    log_dict["process Time"] = process_time
    logger.info(log_dict)
    return response

# Register all API routers
app.include_router(auth.router, tags=["Authentication"])
app.include_router(commands.router, tags=["Commands"])
app.include_router(tasks.router, tags=["Tasks"])
app.include_router(memories.router, tags=["Memories"])
app.include_router(tools.router, tags=["Tools"])
app.include_router(settings.router, tags=["Settings"])
app.include_router(messages.router, tags=["Messages"])
app.include_router(models.router, tags=["Models"])
app.include_router(conversations.router, tags=["Conversations"])

# Health check endpoint
@app.get("/health")
async def root():
    return {
        "status": "healthy", 
        "version": APP_VERSION,
        "message": "Nova API is running"
    }

# Add startup and shutdown events
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup complete")
    
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown")
