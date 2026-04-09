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
EMBED_MODEL  = "allenai-specter"
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
# N_RETRIEVE : candidate chunks pulled from ChromaDB before reranking
#              pull more than you need -- reranker selects the best ones
# N_FINAL    : chunks actually sent to the LLM after reranking
# USE_HYDE   : generate a hypothetical answer, embed it, and search with that
#              instead of the raw query -- improves recall on specific questions
# ---------------------------------------------------------------------------
N_RETRIEVE = 20
N_FINAL    = 6
USE_HYDE   = True
# ---------------------------------------------------------------------------
# AWS Bedrock
# Leave the key fields blank -- boto3 reads ~/.aws/credentials automatically.
# Only fill them in if you need to override the credential file.
# ---------------------------------------------------------------------------
BEDROCK_REGION        = "us-east-2"
BEDROCK_MODEL         = "us.anthropic.claude-sonnet-4-5-20251101-v1:0"
AWS_ACCESS_KEY_ID     = os.environ.get("AWS_ACCESS_KEY_ID",     "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
AWS_SESSION_TOKEN     = os.environ.get("AWS_SESSION_TOKEN",     "")
# ---------------------------------------------------------------------------
# Web search  (query.py)
# Tavily is used if TAVILY_API_KEY is set; falls back to DuckDuckGo.
# ---------------------------------------------------------------------------
WEB_SEARCH_ENABLED = False
TAVILY_API_KEY     = os.environ.get("TAVILY_API_KEY", "")
## config.py additions

# ---------------------------------------------------------------------------
# Full-document mode
# Full text is sent directly in the context window when the user specifies
# one or more papers explicitly (paper: prefix).
# MAX_FULL_DOC_CHARS caps total characters sent to avoid exceeding context.
# At ~4 chars/token, 400k chars ≈ 100k tokens (safe for Claude Sonnet).
# ---------------------------------------------------------------------------
MAX_FULL_DOC_CHARS = 400_000
