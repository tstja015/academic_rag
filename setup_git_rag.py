import os
import stat
import subprocess

# ---------------------------------------------------------------------------
# file contents
# ---------------------------------------------------------------------------

files = {}

files["config.py"] = "\n".join([
    "PAPER_DIRS = [",
    '    "./papers",',
    '    # "/path/to/more/papers",',
    '    # "/mnt/nas/research",',
    "]",
    "",
    'DB_DIR          = "./db"',
    'COLLECTION_NAME = "academic_papers"',
    'EMBED_MODEL     = "BAAI/bge-base-en-v1.5"',
    'CHUNK_SIZE      = 500',
    "",
    "# bedrock settings",
    'BACKEND        = "bedrock"',
    'BEDROCK_REGION = "us-east-2"',
    'BEDROCK_MODEL  = "global.anthropic.claude-sonnet-4-6"',
    "",
    "# leave blank -- boto3 reads ~/.aws/credentials automatically",
    'AWS_ACCESS_KEY_ID     = ""',
    'AWS_SECRET_ACCESS_KEY = ""',
    'AWS_SESSION_TOKEN     = ""',
])

files["requirements.txt"] = "\n".join([
    "pymupdf4llm",
    "chromadb",
    "sentence-transformers",
    "boto3",
    "tqdm",
    "httpx",
])

files["ingest.py"] = r"""
import os
import sys
import json
import hashlib
import pymupdf4llm
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

def get_file_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def find_all_pdfs(paper_dirs):
    pdfs = []
    for base_dir in paper_dirs:
        if not os.path.exists(base_dir):
            print("  Warning: directory not found, skipping: {}".format(base_dir))
            continue
        for root, dirs, filenames in os.walk(base_dir, followlinks=True):
            for f in filenames:
                if f.endswith(".pdf"):
                    pdfs.append(os.path.join(root, f))
    print("Found {} PDFs across {} directories".format(len(pdfs), len(paper_dirs)))
    return pdfs

def clean_section_name(name):
    name = re.sub(r"[*_]", "", name)
    return name.strip().lower()

def parse_sections(markdown_text):
    sections = {}
    current_section = "abstract"
    current_text = []
    for line in markdown_text.split("\n"):
        if re.match(r"^#{1,3}\s+", line):
            if current_text:
                sections[current_section] = "\n".join(current_text).strip()
            current_section = clean_section_name(line.lstrip("#").strip())
            current_text = []
        else:
            current_text.append(line)
    if current_text:
        sections[current_section] = "\n".join(current_text).strip()
    return sections

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def ingest_papers():
    print("Loading embedding model: " + config.EMBED_MODEL)
    embedder = SentenceTransformer(config.EMBED_MODEL)

    client = chromadb.PersistentClient(path=config.DB_DIR)
    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    indexed_hashes = set()
    hash_file = os.path.join(config.DB_DIR, "indexed.json")
    os.makedirs(config.DB_DIR, exist_ok=True)
    if os.path.exists(hash_file):
        with open(hash_file) as f:
            indexed_hashes = set(json.load(f))

    os.makedirs("./papers", exist_ok=True)
    pdfs = find_all_pdfs(config.PAPER_DIRS)

    if not pdfs:
        print("No PDFs found. Add paths to PAPER_DIRS in config.py")
        return

    new_hashes = set()
    skipped   = 0
    processed = 0
    errors    = 0

    for filepath in tqdm(pdfs, desc="Ingesting papers"):
        pdf_file  = os.path.basename(filepath)
        file_hash = get_file_hash(filepath)

        if file_hash in indexed_hashes:
            new_hashes.add(file_hash)
            skipped += 1
            continue

        try:
            md_text  = pymupdf4llm.to_markdown(filepath)
            sections = parse_sections(md_text)
            skip_sections = {"references", "bibliography", "acknowledgements"}

            chunks_to_add   = []
            ids_to_add      = []
            metadata_to_add = []

            base_meta = {
                "filename": pdf_file,
                "name":     pdf_file.replace(".pdf", ""),
                "full_path": filepath,
                "folder":   os.path.dirname(filepath),
            }

            for section_name, section_text in sections.items():
                if any(skip in section_name for skip in skip_sections):
                    continue
                if len(section_text.strip()) < 50:
                    continue

                chunks = chunk_text(section_text, chunk_size=config.CHUNK_SIZE)

                for i, chunk in enumerate(chunks):
                    chunk_id = "{}_{}_{}" .format(file_hash, section_name, i)
                    metadata = {
                        "filename":    base_meta["filename"],
                        "name":        base_meta["name"],
                        "full_path":   base_meta["full_path"],
                        "folder":      base_meta["folder"],
                        "section":     section_name,
                        "chunk_index": i,
                        "file_hash":   file_hash,
                    }
                    chunks_to_add.append(chunk)
                    ids_to_add.append(chunk_id)
                    metadata_to_add.append(metadata)

            if chunks_to_add:
                embeddings = embedder.encode(
                    chunks_to_add,
                    show_progress_bar=False,
                    batch_size=32
                ).tolist()

                collection.add(
                    documents=chunks_to_add,
                    embeddings=embeddings,
                    ids=ids_to_add,
                    metadatas=metadata_to_add
                )

            new_hashes.add(file_hash)
            processed += 1

        except Exception as e:
            print("\n  Error processing {}: {}".format(filepath, e))
            errors += 1
            continue

    all_hashes = indexed_hashes | new_hashes
    with open(hash_file, "w") as f:
        json.dump(list(all_hashes), f)

    print("\n--- Ingest complete ---")
    print("  Processed : {}".format(processed))
    print("  Skipped   : {} (already indexed)".format(skipped))
    print("  Errors    : {}".format(errors))
    print("  Total chunks in DB: {}".format(collection.count()))

if __name__ == "__main__":
    ingest_papers()
""".strip()

files["query.py"] = r"""
import os
import sys

# must be before sentence_transformers import
os.environ["HF_HUB_OFFLINE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import logging
import boto3
import chromadb
from sentence_transformers import SentenceTransformer
import config

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# startup
# ---------------------------------------------------------------------------

def load_resources():
    print("Loading embedding model: {}".format(config.EMBED_MODEL))
    embedder   = SentenceTransformer(config.EMBED_MODEL)
    client     = chromadb.PersistentClient(path=config.DB_DIR)
    collection = client.get_collection(config.COLLECTION_NAME)
    print("Ready.\n")
    return embedder, collection

# ---------------------------------------------------------------------------
# web search
# ---------------------------------------------------------------------------

def web_search(query, n_results=3):
    tavily_key = os.environ.get("TAVILY_API_KEY") or getattr(config, "TAVILY_API_KEY", "")

    if tavily_key:
        try:
            from tavily import TavilyClient
            client  = TavilyClient(api_key=tavily_key)
            results = client.search(
                query=query,
                search_depth="basic",
                max_results=n_results
            )
            hits = results.get("results", [])
            if hits:
                return hits, "tavily"
        except Exception as e:
            print("  Tavily search failed: {} -- trying DuckDuckGo".format(e))

    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=n_results))
        hits = [
            {
                "url":     r.get("href", ""),
                "title":   r.get("title", ""),
                "content": r.get("body", "")
            }
            for r in raw
        ]
        return hits, "duckduckgo"
    except Exception as e:
        print("  DuckDuckGo search failed: {}".format(e))

    return [], "none"

# ---------------------------------------------------------------------------
# bedrock / llm
# ---------------------------------------------------------------------------

def _bedrock_client():
    session_kwargs = {"region_name": config.BEDROCK_REGION}
    if config.AWS_ACCESS_KEY_ID and config.AWS_SECRET_ACCESS_KEY:
        session_kwargs["aws_access_key_id"]     = config.AWS_ACCESS_KEY_ID
        session_kwargs["aws_secret_access_key"] = config.AWS_SECRET_ACCESS_KEY
        if config.AWS_SESSION_TOKEN:
            session_kwargs["aws_session_token"] = config.AWS_SESSION_TOKEN
    return boto3.Session(**session_kwargs).client("bedrock-runtime")

def _invoke(prompt):
    client = _bedrock_client()
    body   = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": prompt}]
    })
    response = client.invoke_model(
        modelId=config.BEDROCK_MODEL,
        body=body,
        contentType="application/json",
        accept="application/json"
    )
    result = json.loads(response["body"].read())

    if result.get("stop_reason") == "refusal" or not result.get("content"):
        return None
    if result["content"][0].get("type") != "text":
        return None

    return result["content"][0]["text"]

def ask_llm(prompt):
    return _invoke(prompt)

def ask_llm_general(question):
    prompt = (
        "You are a helpful research assistant.\n"
        "Answer the following question as clearly and accurately as you can.\n\n"
        "QUESTION:\n{}\n\nANSWER:".format(question)
    )
    result = _invoke(prompt)
    return result if result else "Could not get a response."

# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------

def retrieve(query, collection, embedder, n_results=5,
             section_filter=None, folder_filter=None):
    query_embedding = embedder.encode(query).tolist()

    where      = None
    conditions = []

    if section_filter:
        conditions.append({"section": {"$in": section_filter}})
    if folder_filter:
        conditions.append({"folder": {"$eq": folder_filter}})

    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

# ---------------------------------------------------------------------------
# prompt building
# ---------------------------------------------------------------------------

def build_prompt(query, rag_results, web_results=None):
    paper_parts = []
    for i, (doc, meta) in enumerate(zip(
        rag_results["documents"][0],
        rag_results["metadatas"][0]
    )):
        source = "{} [{}]".format(
            meta.get("filename", "unknown"),
            meta.get("section", "")
        )
        paper_parts.append("[P{}] Source: {}\n{}".format(i + 1, source, doc))
    paper_context = "\n\n---\n\n".join(paper_parts)

    web_context = ""
    if web_results:
        web_parts = []
        for i, r in enumerate(web_results):
            web_parts.append("[W{}] URL: {}\nTitle: {}\n{}".format(
                i + 1,
                r.get("url", ""),
                r.get("title", ""),
                r.get("content", "")
            ))
        web_context = "\n\n---\n\n".join(web_parts)

    if web_context:
        context_block = (
            "PAPER CONTEXT (from the user's own documents):\n{}\n\n"
            "WEB CONTEXT (live search results):\n{}"
        ).format(paper_context, web_context)

        instructions = (
            "You are a research assistant with access to the user's papers "
            "and live web search results.\n"
            "\n"
            "Rules:\n"
            "  1. Answer primarily from PAPER CONTEXT where relevant.\n"
            "  2. Use WEB CONTEXT to supplement or add up-to-date information.\n"
            "  3. If neither context is relevant, answer from general knowledge.\n"
            "  4. Clearly label each part of your answer:\n"
            "       [From papers]          -- from the user's documents\n"
            "       [From web]             -- from web search results\n"
            "       [General knowledge]    -- from your training knowledge\n"
            "  5. Always cite sources: filename for papers, URL for web results.\n"
        )
    else:
        context_block = "PAPER CONTEXT (from the user's own documents):\n{}".format(
            paper_context
        )
        instructions = (
            "You are a research assistant helping a scientist understand their papers.\n"
            "\n"
            "Rules:\n"
            "  1. Answer primarily from PAPER CONTEXT where relevant.\n"
            "  2. If the context does not fully cover the question, supplement\n"
            "     with general knowledge and label it clearly:\n"
            "       [General knowledge, not from the provided papers]\n"
            "  3. If the context is entirely irrelevant, answer from general\n"
            "     knowledge and say so at the top:\n"
            "       [Not found in papers -- answering from general knowledge]\n"
            "  4. Always cite sources: filename and section.\n"
        )

    return "{}\n\n{}\n\nQUESTION:\n{}\n\nANSWER:".format(
        instructions, context_block, query
    )

# ---------------------------------------------------------------------------
# main ask
# ---------------------------------------------------------------------------

def ask(query, embedder, collection,
        section_filter=None, folder_filter=None,
        force_web=False):

    rag_results = retrieve(
        query, collection, embedder,
        n_results=5,
        section_filter=section_filter,
        folder_filter=folder_filter
    )

    web_results  = []
    search_engine = "none"
    use_web = force_web or getattr(config, "WEB_SEARCH_ENABLED", False)

    if use_web:
        print("  Searching web...")
        web_results, search_engine = web_search(query, n_results=3)
        if web_results:
            print("  Got {} web results via {}".format(len(web_results), search_engine))
        else:
            print("  No web results returned")

    prompt = build_prompt(query, rag_results, web_results or None)
    answer = ask_llm(prompt)

    if answer is None:
        print("\n  [model refused context -- retrying as general question]")
        answer = ask_llm_general(query)
        print("\n=== ANSWER ===")
        print("[Not found in papers or web -- answering from general knowledge]")
        print(answer)
        print("\n=== SOURCES ===")
        print("  (none -- answered from general knowledge)")
        return answer

    print("\n=== ANSWER ===")
    print(answer)

    print("\n=== SOURCES ===")
    print("  -- Papers --")
    for meta, dist in zip(rag_results["metadatas"][0], rag_results["distances"][0]):
        print("  {} | section: {} | score: {:.3f}".format(
            meta["filename"], meta["section"], 1 - dist
        ))

    if web_results:
        print("  -- Web ({}) --".format(search_engine))
        for r in web_results:
            print("  {} | {}".format(r.get("url", ""), r.get("title", "")))

    return answer

# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

def summarize_paper(filename, embedder, collection):
    print("Summarizing: {}".format(filename))
    results = retrieve(
        query="main contribution research objective summary conclusions",
        collection=collection,
        embedder=embedder,
        n_results=8,
        section_filter=["abstract", "introduction", "conclusion",
                        "conclusions", "summary", "discussion"]
    )

    if filename and filename != "all":
        docs  = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        filtered = [
            (d, m, s) for d, m, s in zip(docs, metas, dists)
            if filename.lower() in m.get("filename", "").lower()
        ]
        if not filtered:
            print("  Warning: no chunks found for '{}'".format(filename))
            print("  Check the filename spelling or try: summarize:all")
            return
        results["documents"][0] = [x[0] for x in filtered]
        results["metadatas"][0]  = [x[1] for x in filtered]
        results["distances"][0]  = [x[2] for x in filtered]

    ask(
        "Provide a clear summary of this paper covering: "
        "(1) the research problem, "
        "(2) the methods used, "
        "(3) the main findings, "
        "(4) the conclusions and implications.",
        embedder,
        collection
    )

# ---------------------------------------------------------------------------
# model selector
# ---------------------------------------------------------------------------

def get_available_models():
    try:
        client   = boto3.client("bedrock", region_name=config.BEDROCK_REGION)
        response = client.list_inference_profiles()
        return [
            p["inferenceProfileId"]
            for p in response["inferenceProfileSummaries"]
            if "claude" in p["inferenceProfileId"].lower()
        ]
    except Exception as e:
        print("  Could not fetch models: {}".format(e))
        return []

def select_model():
    models = get_available_models()

    print("\n--- Model Selector ---")
    print("Current: {}".format(config.BEDROCK_MODEL))
    print("")

    if models:
        print("Available Claude inference profiles:")
        for i, m in enumerate(models):
            marker = " <-- current" if m == config.BEDROCK_MODEL else ""
            print("  [{}] {}{}".format(i + 1, m, marker))
    else:
        print("Could not fetch model list.")

    print("")
    raw = input("Enter number or full profile ID, or Enter to keep current: ").strip()

    if not raw:
        return config.BEDROCK_MODEL

    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(models):
            chosen = models[idx]
            print("Switched to: {}".format(chosen))
            return chosen
        print("Invalid number. Valid range: 1-{}".format(len(models)))
        return config.BEDROCK_MODEL

    print("Switched to: {}".format(raw))
    return raw

# ---------------------------------------------------------------------------
# query parser
# ---------------------------------------------------------------------------

def parse_query(raw):
    section_filter = None
    folder_filter  = None
    force_web      = False
    query          = raw.strip()

    if query.startswith("methods:"):
        section_filter = ["methods", "methodology"]
        query = query[8:].strip()

    elif query.startswith("results:"):
        section_filter = ["results", "findings"]
        query = query[8:].strip()

    elif query.startswith("folder:"):
        rest = query[7:]
        if ":" in rest:
            folder_filter, query = rest.split(":", 1)
            folder_filter = folder_filter.strip()
            query         = query.strip()

    elif query.startswith("summarize:"):
        return None, None, None, False, query[10:].strip()

    elif query.startswith("web:"):
        force_web = True
        query     = query[4:].strip()

    return query, section_filter, folder_filter, force_web, None

# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    embedder, collection = load_resources()

    web_status = "on" if getattr(config, "WEB_SEARCH_ENABLED", False) else "off"

    print("Academic Paper RAG -- type 'quit' to exit")
    print("Backend    : AWS Bedrock")
    print("Model      : {}".format(config.BEDROCK_MODEL))
    print("Web search : {} (toggle with: webon / weboff)".format(web_status))
    print("")
    print("Prefixes:")
    print("  methods:              search only methods sections")
    print("  results:              search only results sections")
    print("  folder:/path/to:      search only papers in that folder")
    print("  summarize:<filename>  summarize a specific paper")
    print("  web:                  force web search for this query")
    print("  model                 switch Claude model")
    print("  webon / weboff        toggle web search globally")
    print()

    while True:
        try:
            raw = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not raw:
            continue

        if raw.lower() == "quit":
            break

        if raw.lower() == "model":
            new_model = select_model()
            if new_model != config.BEDROCK_MODEL:
                config.BEDROCK_MODEL = new_model
            print("Using model: {}".format(config.BEDROCK_MODEL))
            continue

        if raw.lower() == "webon":
            config.WEB_SEARCH_ENABLED = True
            print("Web search enabled.")
            continue

        if raw.lower() == "weboff":
            config.WEB_SEARCH_ENABLED = False
            print("Web search disabled.")
            continue

        query, section_filter, folder_filter, force_web, summarize_target = parse_query(raw)

        if summarize_target is not None:
            summarize_paper(summarize_target, embedder, collection)
        else:
            ask(query, embedder, collection,
                section_filter=section_filter,
                folder_filter=folder_filter,
                force_web=force_web)
        print()
""".strip()

files["setup.sh"] = "\n".join([
    "#!/bin/bash",
    "set -e",
    "echo 'Setting up conda environment: rag'",
    "conda create -n rag python=3.11 -y",
    "echo 'Activating environment...'",
    "source $(conda info --base)/etc/profile.d/conda.sh",
    "conda activate rag",
    "echo 'Installing dependencies...'",
    "pip install --upgrade pip",
    "pip install -r requirements.txt",
    "echo ''",
    "echo 'Done. Next steps:'",
    "echo '  conda activate rag'",
    "echo '  Edit config.py -- add your PDF paths to PAPER_DIRS'",
    "echo '  python ingest.py'",
    "echo '  python query.py'",
])

files[".gitignore"] = "\n".join([
    "# python",
    "__pycache__/",
    "*.py[cod]",
    "*.egg-info/",
    ".eggs/",
    "",
    "# project runtime data -- not committed",
    "db/",
    "papers/",
    "",
    "# conda / venv",
    ".env",
    "env/",
    "venv/",
    "",
    "# aws credentials -- never commit these",
    ".aws/",
    "",
    "# os",
    ".DS_Store",
    "Thumbs.db",
])

files["README.md"] = "\n".join([
    "# Academic Paper RAG",
    "",
    "## Requirements",
    "- Anaconda or Miniconda",
    "- AWS account with Bedrock access",
    "- AWS credentials in `~/.aws/credentials`",
    "- Claude Sonnet 4.6 enabled in AWS Bedrock Model Access",
    "",
    "## Setup",
    "```bash",
    "git clone ",
    "cd academic_rag",
    "bash setup.sh",
    "```",
    "",
    "## Usage",
    "```bash",
    "conda activate rag",
    "# edit config.py -- add PDF folder paths to PAPER_DIRS",
    "python ingest.py    # once, or when adding new papers",
    "python query.py",
    "```",
    "",
    "## Query Prefixes",
    "| Prefix | Effect |",
    "|--------|--------|",
    "| `methods: ` | search only methods sections |",
    "| `results: ` | search only results sections |",
    "| `folder:/path: ` | search only papers in that folder |",
    "| `summarize:` | summarize a specific paper |",
    "| `web: ` | force web search for this query |",
    "| `model` | switch Claude model interactively |",
    "| `webon` / `weboff` | toggle web search globally |",
    "",
    "## Adding More Papers",
    "1. Add new paths to `PAPER_DIRS` in `config.py`",
    "2. Run `python ingest.py` again",
    "3. Already-indexed files are skipped automatically",
    "",
    "## Config",
    "Edit `config.py` to change:",
    "- `PAPER_DIRS` -- folders to scan for PDFs",
    "- `EMBED_MODEL` -- embedding model",
    "- `BEDROCK_MODEL` -- Claude inference profile ID",
    "- `BEDROCK_REGION` -- AWS region",
])

# ---------------------------------------------------------------------------
# build the repo
# ---------------------------------------------------------------------------

REPO_DIR = "academic_rag"

def write_file(rel_path, content):
    full_path = os.path.join(REPO_DIR, rel_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return full_path

def make_executable(rel_path):
    full_path = os.path.join(REPO_DIR, rel_path)
    current   = os.stat(full_path).st_mode
    os.chmod(full_path, current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

def git(*args, cwd=REPO_DIR):
    result = subprocess.run(
        ["git"] + list(args),
        cwd=cwd,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError("git {} failed:\n{}".format(" ".join(args), result.stderr))
    return result.stdout.strip()

# write all files
os.makedirs(REPO_DIR, exist_ok=True)

for rel_path, content in files.items():
    write_file(rel_path, content)
    print("  wrote {}".format(rel_path))

# papers/ and db/ are gitignored but the dirs should exist locally
for d in ("papers", "db"):
    os.makedirs(os.path.join(REPO_DIR, d), exist_ok=True)

make_executable("setup.sh")

# initialise git repo and make first commit
git("init")
git("config", "user.email", "setup@academic-rag")   # needed in fresh envs
git("config", "user.name",  "Academic RAG Setup")
git("add", ".")
git("commit", "-m", "Initial commit: academic paper RAG")

# summary
print()
print("Git repository created at: {}".format(os.path.abspath(REPO_DIR)))
print()
print("Log:")
print(git("log", "--oneline"))
print()
print("Tracked files:")
for line in git("ls-files").splitlines():
    print("  " + line)
print()
print("Next steps:")
print("  cd {}".format(REPO_DIR))
print("  git remote add origin ")
print("  git push -u origin main")
