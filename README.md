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

Create a fresh rag conda environment (Python 3.11)
Install all Python dependencies
Download NLTK punkt tokeniser
Pre-download and cache SPECTER + reranker models (so queries work offline)
Verify your AWS credentials
Quickstart
conda activate rag
# 1. Edit config.py -- add your PDF folder paths to PAPER_DIRS
#    and confirm BEDROCK_MODEL / BEDROCK_REGION
# 2. Index your papers (re-run any time you add new PDFs)
python ingest.py
# 3. Start querying
python query.py
Query modes
Standard RAG (default)
Retrieves the most relevant chunks via SPECTER + HyDE + cross-encoder reranking. Best for broad questions across many papers.

Full-document mode (paper: prefix)
Sends the complete PDF text directly to Claude -- no chunking, no retrieval. Identical to uploading the file. Use this when you need:

Deep analysis of a specific paper
Questions about tables, equations, or figures
Cross-section reasoning ("how do the methods relate to the limitations?")
Precise numerical values that might fall between chunk boundaries
paper:Smith2023.pdf: what is the kinetic mechanism proposed?
paper:Smith2023.pdf: summarize
paper:Smith2023.pdf,Jones2024.pdf: compare their experimental designs
Partial filenames work too:

paper:Smith2023: what activation energy did they report?
paper:ADH: summarize          -- matches first PDF with "ADH" in the filename
All prefixes (composable)
All prefixes can be combined in a single query:

methods: web: what optimiser did they use?
results: folder:MeOH_ADH: what was the best yield?
Prefix	Mode	Effect
paper:<name>:	Full-doc	Send complete PDF to Claude
paper:,:	Full-doc	Send multiple complete PDFs
methods:	RAG	Restrict to methods sections
results:	RAG	Restrict to results sections
folder::	RAG	Restrict to one topic folder
web:	RAG	Add live web search
summarize:	RAG	Chunk-based summary of one paper
summarize:all	RAG	Summary across all papers
Commands
Command	Effect
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
    |
    v
HyDE: generate hypothetical answer paragraph
    |
    v
SPECTER embedding of hypothetical answer
    |
    v
ChromaDB ANN search -> top 20 candidate chunks
    |
    v
Cross-encoder reranking of all 20 pairs (query, chunk)
    |
    v
Top 6 chunks by rerank score
    |
    v
Build prompt: history + instructions + chunks + question
    |
    v
Claude (AWS Bedrock) -> answer with citations
Full-document pipeline (paper: prefix)
paper:: 
    |
    v
Resolve filename -> full path in ChromaDB metadata
    |
    v
pymupdf4llm extracts complete PDF -> clean markdown
    |
    v
Full text sent directly to Claude (up to 400,000 chars)
    |
    v
Claude reads entire paper -> answer with section citations
Save it:

```bash
cat > README.md << 'ENDOFFILE'
# Academic Paper RAG
... (paste the content above)
ENDOFFILE
