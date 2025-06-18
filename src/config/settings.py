import os
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the directory containing this script
CONFIG_DIR = Path(__file__).parent.resolve()
ENV_PATH = CONFIG_DIR / '.env'

# Load environment variables from config directory
env_loaded = load_dotenv(ENV_PATH)
if not env_loaded:
    logger.warning(f"Environment variables not loaded. Ensure .env file exists at: {ENV_PATH}")
else:
    logger.debug(f"Environment loaded from: {ENV_PATH}")

# Base directory of the project
BASE_DIR = CONFIG_DIR.parent.parent

# --- Model Paths ---
# These should point to where your downloaded models are stored,
# relative to the project or absolute paths.
# Example: os.path.join(BASE_DIR, "models", "yolov8n.pt")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt") # Default, or get from env
BLIP_MODEL_PATH = os.getenv("BLIP_MODEL_PATH", "path/to/your/blip/model")
WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_TYPE", "base") # e.g., "base", "small", "medium"
PANNS_MODEL_PATH = os.getenv("PANNS_MODEL_PATH", os.path.join(BASE_DIR, "models", "cnn14.pth"))

# --- Ollama Settings ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("DEFAULT_OLLAMA_MODEL", "llama2") # Example, choose your preferred

# --- Other Settings ---
DEFAULT_TRANSCRIPTION_LANGUAGE = "en"
VIDEO_FRAME_EXTRACTION_INTERVAL = 5 # seconds

# YOLO settings
YOLO_MODEL_PATH = "yolo11x.pt"  
FRAME_INTERVAL_SECONDS = 5  

# Hugging Face settings
HUGGING_FACE_TOKEN = os.getenv('HUGGING_FACE_TOKEN')