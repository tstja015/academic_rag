# config.py
# Central configuration for the Academic Paper RAG system.
# Edit PAPER_DIRS and Bedrock settings before first use.
import os

# ---------------------------------------------------------------------------
# Paper sources
# ---------------------------------------------------------------------------
PAPER_DIRS = [
    "./papers",
    "/home/ans000/doc/CEED",
    "/home/ans000/doc/New_Projects",
    "/home/ans000/doc/MeOH_ADH",
    "/home/ans000/doc/MeOH_FalDH",
    "/home/ans000/doc/MeOH_FDH",
    "/home/ans000/doc/papers",
    "/home/ans000/doc/copper_nanowire",
    "/home/ans000/doc/FAS",
    "/home/ans000/doc/GOx",
    "/home/ans000/doc/H2O2 project",
    "/home/ans000/doc/MeOH_unclear",
    "/home/ans000/doc/Methane_monooxygenase",
    "/home/ans000/doc/Nanowire",
    "/home/ans000/doc/Nitrogenase",
    "/home/ans000/doc/Procedures",
    "/home/ans000/doc/ACPS",
]

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
DB_DIR          = "./db"
COLLECTION_NAME = "academic_papers"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
EMBED_MODEL  = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# Chunking  (ingest.py)
# ---------------------------------------------------------------------------
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100

# ---------------------------------------------------------------------------
# Retrieval  (query.py)
# ---------------------------------------------------------------------------
N_RETRIEVE       = 40
N_FINAL          = 8
USE_HYDE         = True
RERANK_THRESHOLD = -5.0

# ---------------------------------------------------------------------------
# LLM Backend
# ---------------------------------------------------------------------------
LLM_BACKEND = "bedrock"   # "bedrock" or "ollama"

# ---------------------------------------------------------------------------
# AWS Bedrock
# ---------------------------------------------------------------------------
BEDROCK_REGION        = "us-east-2"
BEDROCK_MODEL         = "global.anthropic.claude-sonnet-4-6"
AWS_ACCESS_KEY_ID     = os.environ.get("AWS_ACCESS_KEY_ID",     "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
AWS_SESSION_TOKEN     = os.environ.get("AWS_SESSION_TOKEN",     "")

# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "gemma2:latest"

# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------
MAX_TURNS        = 15    # increased from 8; auto-reduced to 3 in full-doc mode
MAX_ANSWER_CHARS = 4000  # per-turn truncation to keep history manageable

# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------
WEB_SEARCH_ENABLED = False
TAVILY_API_KEY     = os.environ.get("TAVILY_API_KEY", "")

# ---------------------------------------------------------------------------
# Full-document mode
# ---------------------------------------------------------------------------
MAX_FULL_DOC_CHARS = 400_000

# ---------------------------------------------------------------------------
# OneDrive space management
# ---------------------------------------------------------------------------
ONEDRIVE_FREE_AFTER_INGEST = True
ONEDRIVE_PATH_HINT         = "/mnt/c/Users/tonys"
ONEDRIVE_FREE_ALL          = True
ONEDRIVE_CLEANUP_LOGS      = True

# ---------------------------------------------------------------------------
# Ingestion progress tracking
# ---------------------------------------------------------------------------
PROGRESS_FILE = "./db/ingest_progress.json"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a rigorous scientific research assistant with deep expertise \
in computational chemistry, biochemistry, molecular simulation, and scientific programming.

Rules:
- When given paper text, extract information ONLY from that text. Do not substitute \
values from memory when the paper provides them.
- Reproduce equations exactly as written in the paper, noting equation numbers.
- When asked to implement code, include ALL edge cases and conditional logic \
described in the source material.
- If the paper describes multiple functional forms for the same term (e.g., \
harmonic vs Morse for bonds), implement ALL of them.
- Never silently drop part of the user's request. If you cannot complete \
something, say so explicitly.
- When reproducing tables, include every row. Do not summarize with "..." or \
"similar entries follow".
- Cite specific sections, equations, tables, and page numbers for every claim.
- When asked to write code that computes forces AND energies, you MUST include both. \
Forces are the negative gradient of the energy with respect to atomic coordinates.
"""
