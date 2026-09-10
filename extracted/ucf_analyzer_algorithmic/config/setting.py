import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_ARCHIVE = BASE_DIR / "dataset.rar"
IMG_DIR = BASE_DIR / "img"

# Optional API keys. Keep real values only in .env, never in source control.
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
