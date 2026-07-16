from pathlib import Path

# Root directory of demo
DEMO_ROOT = Path(__file__).resolve().parent.parent

# Generated Files
GENERATED_DIR = DEMO_ROOT / "generated"

TRANSCRIPT_DIR = GENERATED_DIR / "transcripts"

STATE_DIR = GENERATED_DIR / "states"

# Context Memory
MEMORY_DIR = DEMO_ROOT / "memory"

CONTEXT_DB = MEMORY_DIR / "context_db.json"

# Logs
LOG_DIR = DEMO_ROOT / "logs"

# Ollama

OLLAMA_MODEL = "qwen2.5:7b-instruct"

OLLAMA_URL = "http://localhost:11434/api/generate"

# Random Seed

RANDOM_SEED = 42