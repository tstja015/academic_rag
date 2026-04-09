# Academic Paper RAG
A local retrieval-augmented generation (RAG) system for scientific papers,
backed by AWS Bedrock (Claude) and ChromaDB.
## What makes this optimised for scientific papers
| Component | Choice | Why |
|-----------|--------|-----|
| Embeddings | `allenai-specter` | Trained on academic paper citations -- understands scientific terminology better than general models |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reads query + chunk together; far more accurate than cosine similarity for discriminating which chunk actually answers a specific question |
| HyDE | Enabled by default | Generates a hypothetical answer to the query, embeds that instead of the raw question -- significantly improves recall for factual questions about methods and results |
| Chunking | Sentence-boundary-aware (NLTK) | No mid-sentence splits; overlapping windows preserve cross-boundary context |
| Context prefix | Paper title + section prepended to every chunk | Embedding encodes document identity alongside content |
| Section parsing | Markdown + bold + ALL-CAPS + numbered headers | Handles the full variety of PDF-to-markdown conversion artefacts |
## Requirements
- Anaconda or Miniconda
- AWS account with Bedrock access (Claude enabled in your region)
- AWS credentials in `~/.aws/credentials` or environment variables
## Setup
```bash
git clone 
cd academic_rag
bash setup.sh
setup.sh will:
1. Create a fresh rag conda environment (Python 3.11)
2. Install all Python dependencies
3. Download NLTK punkt tokeniser
4. Pre-download and cache SPECTER + reranker models (so queries work offline)
5. Verify your AWS credentials


Quickstart

conda activate rag
# 1. Edit config.py -- add your PDF folder paths to PAPER_DIRS
#    and confirm BEDROCK_MODEL / BEDROCK_REGION
# 2. Index your papers (re-run any time you add new PDFs)
python ingest.py
# 3. Start querying
python query.py
Query syntax
All prefixes are composable -- combine them in a single query:
methods: web: what optimiser did they use?
results: folder:MeOH_ADH: what was the best yield?
Prefix / Command	Effect
methods:	Restrict retrieval to methods / methodology sections
results:	Restrict retrieval to results / findings sections
folder::	Restrict to papers in a specific topic folder
web:	Force a live web search in addition to paper retrieval
summarize:	Structured summary of one paper
summarize:all	Summary drawing on all indexed papers
list	Show all indexed papers and their folders
model	Switch Claude model interactively
history	Print the current conversation history
clear	Clear conversation history
webon / weboff	Toggle web search globally
quit	Exit
Configuration
All settings are in config.py:

PAPER_DIRS   = ["./papers", "/path/to/more"]  # folders to scan
EMBED_MODEL  = "allenai-specter"               # embedding model
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BEDROCK_MODEL  = "us.anthropic.claude-..."     # your Claude profile ID
BEDROCK_REGION = "us-east-2"
N_RETRIEVE   = 20     # candidates before reranking
N_FINAL      = 6      # chunks sent to LLM
USE_HYDE     = True   # hypothetical document embedding
Adding more papers

# Add new paths to PAPER_DIRS in config.py, then:
python ingest.py
# Already-indexed files are skipped automatically via MD5 hash check.
Folder structure
academic_rag/
├── config.py        # all settings
├── ingest.py        # PDF indexing pipeline
├── query.py         # interactive query interface
├── requirements.txt
├── setup.sh
├── papers/          # default paper directory
└── db/              # ChromaDB persistent storage
    └── indexed.json # hashes of already-indexed files

Retrieval pipeline (per query)
User query
    │
    ▼
HyDE: generate hypothetical answer paragraph
    │
    ▼
SPECTER embedding of hypothetical answer
    │
    ▼
ChromaDB ANN search → top 20 candidate chunks
    │
    ▼
Cross-encoder reranking of all 20 pairs (query, chunk)
    │
    ▼
Top 6 chunks by rerank score
    │
    ▼
Build prompt: history + instructions + chunks + question
    │
    ▼
Claude (AWS Bedrock) → answer with citations

