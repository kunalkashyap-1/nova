from fastapi import APIRouter
from app.schemas.tool import ToolOutput

router = APIRouter()

@router.get("/", response_model=ToolOutput)
def list_commands():
    """List available commands."""
    return ToolOutput(result=["run", "stop", "status"])

@router.post("/run", response_model=ToolOutput)
def run_command():
    """Mock: Run a command."""
    return ToolOutput(result="Command started.")

@router.post("/stop", response_model=ToolOutput)
def stop_command():
    """Mock: Stop a command."""
    return ToolOutput(result="Command stopped.")

@router.get("/status", response_model=ToolOutput)
def command_status():
    """Mock: Get command status."""
    return ToolOutput(result="Command is running.")

