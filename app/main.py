from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.routers import auth, commands, tasks, memories, tools, settings, messages
from app.logging import logger
import time

app = FastAPI(title="Nova", description="A chatbot assistant", version="0.0.1")

logger.info("Server starting...")

# Allow CORS only for localhost:5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    log_dict = {
        "request": {
            "url": request.url,
            "method": request.method,
            "headers": dict(request.headers),
            "body": request.body,
        }
    }
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    log_dict["process Time"] = process_time
    logger.info(log_dict)
    return response

app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(commands.router, prefix="/api/v1/commands", tags=["Commands"])
app.include_router(tasks.router, prefix="/api/v1/tasks", tags=["tasks"])
app.include_router(memories.router, prefix="/api/v1/memories", tags=["Memories"])
app.include_router(tools.router, prefix="/api/v1/tools", tags=["tools"])
app.include_router(settings.router, prefix="/api/v1/settings", tags=["Settings"])
app.include_router(messages.router, prefix="/api/v1/messages", tags=["Messages"])

@app.get("/")
async def root():
    return {"message": "Live and UP"}
