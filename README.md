# Academic Paper RAG

## Requirements
- Anaconda or Miniconda
- AWS account with Bedrock access
- AWS credentials in `~/.aws/credentials`
- Claude Sonnet 4.6 enabled in AWS Bedrock Model Access

## Setup
```bash
git clone 
cd academic_rag
bash setup.sh
```

## Usage
```bash
conda activate rag
# edit config.py -- add PDF folder paths to PAPER_DIRS
python ingest.py    # once, or when adding new papers
python query.py
```

## Query Prefixes
| Prefix | Effect |
|--------|--------|
| `methods: ` | search only methods sections |
| `results: ` | search only results sections |
| `folder:/path: ` | search only papers in that folder |
| `summarize:` | summarize a specific paper |
| `web: ` | force web search for this query |
| `model` | switch Claude model interactively |
| `webon` / `weboff` | toggle web search globally |

## Adding More Papers
1. Add new paths to `PAPER_DIRS` in `config.py`
2. Run `python ingest.py` again
3. Already-indexed files are skipped automatically

## Config
Edit `config.py` to change:
- `PAPER_DIRS` -- folders to scan for PDFs
- `EMBED_MODEL` -- embedding model
- `BEDROCK_MODEL` -- Claude inference profile ID
- `BEDROCK_REGION` -- AWS region