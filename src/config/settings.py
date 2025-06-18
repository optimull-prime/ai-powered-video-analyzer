import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Model Paths ---
# These should point to where your downloaded models are stored,
# relative to the project or absolute paths.
# Example: os.path.join(BASE_DIR, "models", "yolov8n.pt")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt") # Default, or get from env
BLIP_MODEL_PATH = os.getenv("BLIP_MODEL_PATH", "path/to/your/blip/model") # Where is the blip model?
WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_TYPE", "medium.en") # e.g., "base", "small", "medium"
PANNS_MODEL_PATH = os.getenv("PANNS_MODEL_PATH", os.path.join(BASE_DIR, "models", "cnn14.pth"))

# --- Ollama Settings ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("DEFAULT_OLLAMA_MODEL", "llama2") # Example, choose your preferred

# --- Other Settings ---
DEFAULT_TRANSCRIPTION_LANGUAGE = "en"
VIDEO_FRAME_EXTRACTION_INTERVAL = 5 # seconds

# YOLO settings
YOLO_MODEL_PATH = "yolo11x.pt"  # Use small model by default
FRAME_INTERVAL_SECONDS = 5  # Extract frame every 5 seconds
