#!/bin/bash
# setup.sh -- Create conda environment and install all dependencies.
# Usage: bash setup.sh
set -e

ENV_NAME="rag"

echo "========================================"
echo "  Academic Paper RAG -- Environment Setup"
echo "========================================"
echo ""

# -------------------------------------------------------------------------
# Create environment
# -------------------------------------------------------------------------
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda environment '${ENV_NAME}' already exists."
    read -p "Recreate it from scratch? [y/N]: " yn
    case "$yn" in
        [Yy]*)
            echo "Removing existing environment..."
            conda env remove -n "${ENV_NAME}" -y
            echo "Creating conda environment: ${ENV_NAME} (Python 3.11)..."
            conda create -n "${ENV_NAME}" python=3.11 -y
            ;;
        *)
            echo "Keeping existing environment.  Running pip install only."
            ;;
    esac
else
    echo "Creating conda environment: ${ENV_NAME} (Python 3.11)..."
    conda create -n "${ENV_NAME}" python=3.11 -y
fi

# -------------------------------------------------------------------------
# Activate
# -------------------------------------------------------------------------
echo ""
echo "Activating environment..."
# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# -------------------------------------------------------------------------
# Install dependencies
# -------------------------------------------------------------------------
echo "Upgrading pip..."
pip install --upgrade pip

echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

# -------------------------------------------------------------------------
# Download NLTK data (ingest.py does this too, but nice to pre-fetch)
# -------------------------------------------------------------------------
echo ""
echo "Downloading NLTK punkt tokeniser..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# -------------------------------------------------------------------------
# Download models into local cache (so ingest/query can run offline)
# -------------------------------------------------------------------------
echo ""
echo "Pre-downloading embedding and reranker models..."
python - <<'PYEOF'
import config
from sentence_transformers import SentenceTransformer, CrossEncoder

print("  Downloading embedding model: {}".format(config.EMBED_MODEL))
SentenceTransformer(config.EMBED_MODEL)

print("  Downloading reranker model : {}".format(config.RERANK_MODEL))
CrossEncoder(config.RERANK_MODEL)

print("  Models cached successfully.")
PYEOF

# -------------------------------------------------------------------------
# Verify AWS credentials
# -------------------------------------------------------------------------
echo ""
echo "Checking AWS credentials..."
python - <<'PYEOF'
import boto3, config
try:
    sts = boto3.client("sts", region_name=config.BEDROCK_REGION)
    identity = sts.get_caller_identity()
    print("  AWS identity : {}".format(identity.get("Arn", "unknown")))
    print("  Credentials  : OK")
except Exception as e:
    print("  WARNING: Could not verify AWS credentials: {}".format(e))
    print("  Ensure ~/.aws/credentials is configured before running query.py")
PYEOF

# -------------------------------------------------------------------------
# Create default directories
# -------------------------------------------------------------------------
echo ""
echo "Creating default directories..."
mkdir -p ./db ./papers
echo "  ./db      -- ChromaDB storage"
echo "  ./papers  -- default paper directory (also reads from config.PAPER_DIRS)"

# -------------------------------------------------------------------------
# Quick sanity check -- import all modules
# -------------------------------------------------------------------------
echo ""
echo "Verifying imports..."
python - <<'PYEOF'
errors = []
for mod in ["fitz", "pymupdf4llm", "chromadb", "sentence_transformers",
            "boto3", "nltk", "tqdm", "httpx"]:
    try:
        __import__(mod)
    except ImportError as e:
        errors.append(f"  FAIL: {mod} -- {e}")

# Optional modules (warn but don't fail)
for mod in ["duckduckgo_search", "tavily", "ocrmypdf"]:
    try:
        __import__(mod)
    except ImportError:
        print(f"  Optional: {mod} not installed (OK -- graceful fallback)")

if errors:
    print("Import errors:")
    for e in errors:
        print(e)
    exit(1)
else:
    print("  All core imports OK.")
PYEOF

# -------------------------------------------------------------------------
# Done
# -------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  Setup complete."
echo "========================================"
echo ""
echo "Next steps:"
echo "  1.  conda activate ${ENV_NAME}"
echo "  2.  Edit config.py -- verify PAPER_DIRS and BEDROCK_MODEL"
echo "  3.  python ingest.py    # index your PDFs"
echo "  4.  python query.py     # start querying"
echo ""
echo "To re-index after adding new papers:"
echo "  python ingest.py        # already-indexed files are skipped"
echo ""
echo "To wipe and rebuild the database (e.g., after schema changes):"
echo "  rm -rf ./db && python ingest.py"
echo ""
