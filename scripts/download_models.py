"""
Utility script to download and manage HuggingFace models.
"""
import os
import argparse
from pathlib import Path
from typing import List, Optional
import logging
from dotenv import load_dotenv
from huggingface_hub import snapshot_download, hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get models directory from environment variable
MODELS_DIR = Path(os.getenv("HF_MODELS_PATH", "../hf_models"))
MODELS_DIR.mkdir(exist_ok=True)

def download_model(
    model_id: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    use_auth_token: bool = False
) -> Path:
    """
    Download a model from HuggingFace Hub.
    
    Args:
        model_id: The model identifier on HuggingFace Hub
        revision: Specific model revision/branch to download
        token: HuggingFace API token for private models
        use_auth_token: Whether to use authentication token
        
    Returns:
        Path to the downloaded model
    """
    logger.info(f"Downloading model: {model_id}")
    
    try:
        # Download the model files
        model_path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_dir=MODELS_DIR / model_id.split("/")[-1],
            token=token if use_auth_token else None,
            local_dir_use_symlinks=False
        )
        
        # Verify the model can be loaded
        logger.info("Verifying model files...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        
        logger.info(f"Successfully downloaded and verified model: {model_id}")
        return Path(model_path)
        
    except Exception as e:
        logger.error(f"Error downloading model {model_id}: {str(e)}")
        raise

def list_downloaded_models() -> List[str]:
    """
    List all downloaded models in the models directory.
    
    Returns:
        List of model names
    """
    return [d.name for d in MODELS_DIR.iterdir() if d.is_dir()]

def main():
    parser = argparse.ArgumentParser(description="Download and manage HuggingFace models")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download a model")
    download_parser.add_argument("model_id", help="Model ID on HuggingFace Hub")
    download_parser.add_argument("--revision", help="Specific model revision/branch")
    download_parser.add_argument("--token", help="HuggingFace API token for private models")
    download_parser.add_argument("--use-auth", action="store_true", help="Use authentication token")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List downloaded models")
    
    args = parser.parse_args()
    
    if args.command == "download":
        download_model(
            args.model_id,
            revision=args.revision,
            token=args.token,
            use_auth_token=args.use_auth
        )
    elif args.command == "list":
        models = list_downloaded_models()
        if models:
            print("\nDownloaded models:")
            for model in models:
                print(f"- {model}")
        else:
            print("No models downloaded yet.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 