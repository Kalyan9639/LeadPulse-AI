"""
Configuration settings for the Lead Generation and Extraction Pipeline.
Loads environment variables from .env if present.
"""

import os
from pathlib import Path

# Attempt to load .env file
try:
    from dotenv import load_dotenv
    # Search in current directory and parent directories
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except ImportError:
    pass

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Lead_Gen_ML_Model"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ML Classification & Hugging Face Model Settings
HF_REPO_ID = os.getenv("HF_REPO_ID", "mr-checker/SalesGen")
HF_MODEL_FILENAME = os.getenv("HF_MODEL_FILENAME", "sales_model_balanced.joblib")
DEFAULT_MODEL_PATH = None  # None indicates downloading from Hugging Face at runtime
FALLBACK_LOCAL_MODEL = str(MODEL_DIR / "sales_model_balanced.joblib")

# Ollama Embedding Settings
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "granite-embedding:30m")
FALLBACK_EMBED_MODELS = ["granite-embeddings:30m", "granite-embedding:30m"]

# Groq LLM Settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# Excel Storage Settings
DEFAULT_EXCEL_PATH = str(OUTPUT_DIR / "leads.xlsx")

# Target Extraction Schema (Strict 7 Features)
TARGET_SCHEMA = [
    "Name",
    "Phone no.",
    "Source",
    "Last Discussed topic",
    "Status",
    "Assigned to",
    "Follow up?",
]

# Chunking & Retrieval Parameters
CHUNK_SIZE = 450
CHUNK_OVERLAP = 80
TOP_K_CHUNKS = 5
MAX_CONCURRENT_WORKERS = 4
