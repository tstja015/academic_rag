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
    # "/path/to/more/papers",
]
# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
DB_DIR          = "./db"
COLLECTION_NAME = "academic_papers"
# ---------------------------------------------------------------------------
# Models
# allenai-specter  : trained on scientific paper citations -- best for science
# BAAI/bge-large   : strong general alternative if SPECTER is too slow
# Reranker         : cross-encoder reads query+chunk together, much more accurate
#                    than cosine similarity alone
# ---------------------------------------------------------------------------
EMBED_MODEL  = "BAAI/bge-base-en-v1.5" # "intfloat/e5-large-v2" #"allenai-specter"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# ---------------------------------------------------------------------------
# Chunking  (ingest.py)
# CHUNK_SIZE    : target words per chunk (sentence boundaries are respected,
#                 so actual size will vary slightly)
# CHUNK_OVERLAP : words carried over from previous chunk to preserve context
# ---------------------------------------------------------------------------
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100
# ---------------------------------------------------------------------------
# Retrieval  (query.py)
# N_RETRIEVE       : candidate chunks pulled from ChromaDB before reranking
#                    pull more than you need -- reranker selects the best ones
# N_FINAL          : chunks actually sent to the LLM after reranking
# USE_HYDE         : generate a hypothetical answer, embed it, and search with
#                    that instead of the raw query -- improves recall
# RERANK_THRESHOLD : minimum cross-encoder score to include a chunk.
#                    Chunks below this are irrelevant and excluded from the
#                    prompt.  Set to -100 to disable filtering.
# ---------------------------------------------------------------------------
N_RETRIEVE       = 40
N_FINAL          = 8
USE_HYDE         = True
RERANK_THRESHOLD = -5.0 # If this doesn't work, just do -100
# ---------------------------------------------------------------------------
# AWS Bedrock
# Leave the key fields blank -- boto3 reads ~/.aws/credentials automatically.
# Only fill them in if you need to override the credential file.
# ---------------------------------------------------------------------------
BEDROCK_REGION        = "us-east-2"
BEDROCK_MODEL         = "global.anthropic.claude-sonnet-4-6"
AWS_ACCESS_KEY_ID     = os.environ.get("AWS_ACCESS_KEY_ID",     "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
AWS_SESSION_TOKEN     = os.environ.get("AWS_SESSION_TOKEN",     "")
# ---------------------------------------------------------------------------
# Web search  (query.py)
# Tavily is used if TAVILY_API_KEY is set; falls back to DuckDuckGo.
# ---------------------------------------------------------------------------
WEB_SEARCH_ENABLED = False
TAVILY_API_KEY     = os.environ.get("TAVILY_API_KEY", "")
# ---------------------------------------------------------------------------
# Full-document mode
# Full text is sent directly in the context window when the user specifies
# one or more papers explicitly (paper: prefix).
# MAX_FULL_DOC_CHARS caps total characters sent to avoid exceeding context.
# At ~4 chars/token, 400k chars ≈ 100k tokens (safe for Claude Sonnet).
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
# Allows resuming interrupted ingest runs without re-processing files.
# ---------------------------------------------------------------------------
PROGRESS_FILE = "./db/ingest_progress.json"
# ---------------------------------------------------------------------------
# System prompt for LLM
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
