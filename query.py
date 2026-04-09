## query.py (full rebuild)

"""
query.py -- Interactive RAG query interface for academic papers.

Retrieval modes
---------------
Standard (default):
    Two-stage retrieval -- SPECTER embeddings via HyDE → top-N candidates →
    cross-encoder reranking → best chunks sent to Claude.

Full-document mode  (paper: prefix):
    The complete text of the specified paper(s) is extracted from the PDF
    and sent directly in the context window, exactly as if you had uploaded
    the file to Claude.  This enables deep analysis, precise equation reading,
    table interpretation, and cross-section reasoning that chunk-based RAG
    cannot match.  Use it when you need thorough analysis of a specific paper.

    Syntax:
        paper:<filename>: <question>
        paper:<file1>,<file2>: <question>
        paper:<filename>: summarize
"""

import os
import sys
import re
import json
import logging

# Must be set before any sentence_transformers import
os.environ["HF_HUB_OFFLINE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import boto3
import chromadb
import pymupdf4llm
from sentence_transformers import SentenceTransformer, CrossEncoder

import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Terminal colours
# ---------------------------------------------------------------------------
class C:
    ANSWER = "\033[96m"   # cyan
    LABEL  = "\033[93m"   # yellow
    DIM    = "\033[2m"    # grey
    BOLD   = "\033[1m"    # bold
    RESET  = "\033[0m"


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------
class ConversationHistory:
    """
    Rolling window of past Q/A turns prepended to each prompt.
    Lets the model resolve follow-up references like "they", "that paper",
    "the method mentioned above", etc.
    """
    def __init__(self, max_turns: int = 5):
        self.turns:     list = []
        self.max_turns: int  = max_turns

    def add(self, question: str, answer: str):
        self.turns.append({"q": question, "a": answer})
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def clear(self):
        self.turns = []

    def format_for_prompt(self) -> str:
        if not self.turns:
            return ""
        lines = ["CONVERSATION HISTORY (oldest first):"]
        for t in self.turns:
            a_short = t["a"][:400] + "..." if len(t["a"]) > 400 else t["a"]
            lines.append("  Q: {}".format(t["q"]))
            lines.append("  A: {}".format(a_short))
        return "\n".join(lines) + "\n\n"


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def load_resources():
    """
    Load embedding model, cross-encoder reranker, and ChromaDB collection.
    All three stay in memory for the lifetime of the process.
    """
    print("Loading embedding model  : {}".format(config.EMBED_MODEL))
    embedder = SentenceTransformer(config.EMBED_MODEL)

    print("Loading reranker model   : {}".format(config.RERANK_MODEL))
    reranker = CrossEncoder(config.RERANK_MODEL)

    client     = chromadb.PersistentClient(path=config.DB_DIR)
    collection = client.get_collection(config.COLLECTION_NAME)

    print("ChromaDB chunks loaded   : {}".format(collection.count()))
    print("Ready.\n")
    return embedder, reranker, collection


# ---------------------------------------------------------------------------
# Bedrock client (cached)
# ---------------------------------------------------------------------------

_BEDROCK_CLIENT = None


def _get_bedrock_client():
    global _BEDROCK_CLIENT
    if _BEDROCK_CLIENT is None:
        session_kwargs = {"region_name": config.BEDROCK_REGION}
        if config.AWS_ACCESS_KEY_ID and config.AWS_SECRET_ACCESS_KEY:
            session_kwargs["aws_access_key_id"]     = config.AWS_ACCESS_KEY_ID
            session_kwargs["aws_secret_access_key"] = config.AWS_SECRET_ACCESS_KEY
            if config.AWS_SESSION_TOKEN:
                session_kwargs["aws_session_token"] = config.AWS_SESSION_TOKEN
        _BEDROCK_CLIENT = boto3.Session(**session_kwargs).client("bedrock-runtime")
    return _BEDROCK_CLIENT


def reset_bedrock_client():
    global _BEDROCK_CLIENT
    _BEDROCK_CLIENT = None


# ---------------------------------------------------------------------------
# LLM invocation
# ---------------------------------------------------------------------------

def _invoke(prompt: str, max_tokens: int = 4096) -> str | None:
    """
    Send *prompt* to Claude via Bedrock.
    max_tokens is raised to 4096 by default to handle long full-doc answers.
    Returns None on refusal or unexpected response structure.
    """
    client = _get_bedrock_client()
    body   = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages":   [{"role": "user", "content": prompt}],
    })

    try:
        response = client.invoke_model(
            modelId     = config.BEDROCK_MODEL,
            body        = body,
            contentType = "application/json",
            accept      = "application/json",
        )
    except Exception as exc:
        print("  Bedrock invocation error: {}".format(exc))
        return None

    result = json.loads(response["body"].read())

    if result.get("stop_reason") == "refusal":
        return None
    content = result.get("content", [])
    if not content or content[0].get("type") != "text":
        return None

    return content[0]["text"]


def ask_llm(prompt: str, max_tokens: int = 4096) -> str | None:
    return _invoke(prompt, max_tokens=max_tokens)


def ask_llm_general(question: str) -> str:
    prompt = (
        "You are a helpful research assistant specialising in scientific literature.\n"
        "Answer the following question as clearly and accurately as you can.\n\n"
        "QUESTION:\n{}\n\nANSWER:".format(question)
    )
    result = _invoke(prompt)
    return result if result else "Could not get a response from the model."


# ---------------------------------------------------------------------------
# Full-document mode
# ---------------------------------------------------------------------------

def resolve_paper_paths(name_tokens: list, collection) -> list:
    """
    For each token in *name_tokens*, find the best-matching full_path stored
    in ChromaDB metadata.

    Matching strategy (in order):
      1. Exact filename match          e.g. "Smith2023.pdf"
      2. Case-insensitive filename     e.g. "smith2023.pdf"
      3. Filename contains the token   e.g. "smith" matches "Smith2023_ADH.pdf"
      4. full_path contains the token  e.g. partial directory match

    Returns a list of (filename, full_path) tuples for every resolved paper.
    Prints a warning for any token that could not be resolved.
    """
    # Build a deduplicated map:  filename -> full_path
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    path_map: dict = {}
    for m in all_meta:
        fname = m.get("filename", "")
        fpath = m.get("full_path", "")
        if fname and fpath and fname not in path_map:
            path_map[fname] = fpath

    resolved = []
    for token in name_tokens:
        token_lower = token.lower()
        match = None

        # 1. Exact
        if token in path_map:
            match = (token, path_map[token])

        # 2. Case-insensitive exact
        if match is None:
            for fname, fpath in path_map.items():
                if fname.lower() == token_lower:
                    match = (fname, fpath)
                    break

        # 3. Filename contains token
        if match is None:
            candidates = [
                (fname, fpath)
                for fname, fpath in path_map.items()
                if token_lower in fname.lower()
            ]
            if len(candidates) == 1:
                match = candidates[0]
            elif len(candidates) > 1:
                # Prefer shortest filename (most specific match)
                candidates.sort(key=lambda x: len(x[0]))
                print("  Ambiguous token '{}' -- matched multiple files:".format(token))
                for i, (fn, _) in enumerate(candidates):
                    print("    [{}] {}".format(i + 1, fn))
                print("  Using: {}  (use full filename to disambiguate)".format(
                    candidates[0][0]))
                match = candidates[0]

        # 4. full_path contains token
        if match is None:
            for fname, fpath in path_map.items():
                if token_lower in fpath.lower():
                    match = (fname, fpath)
                    break

        if match:
            resolved.append(match)
        else:
            print("  Warning: could not resolve paper '{}'. "
                  "Run 'list' to see indexed filenames.".format(token))

    # Deduplicate while preserving order
    seen   = set()
    unique = []
    for item in resolved:
        if item[0] not in seen:
            seen.add(item[0])
            unique.append(item)

    return unique


def extract_full_text(filepath: str) -> str:
    """
    Convert a PDF to clean markdown text via pymupdf4llm.
    Returns the raw markdown string, or an empty string on failure.
    """
    try:
        return pymupdf4llm.to_markdown(filepath)
    except Exception as exc:
        print("  Error reading {}: {}".format(filepath, exc))
        return ""


def build_fulldoc_prompt(query:    str,
                          papers:   list,
                          history:  ConversationHistory = None) -> tuple:
    """
    Build a prompt that contains the *complete text* of every paper in
    *papers* (list of (filename, full_text) tuples).

    The total character count is capped at config.MAX_FULL_DOC_CHARS to
    avoid exceeding the model's context window.  If the combined text is
    too large, papers are truncated in order (last paper truncated first)
    with a visible warning.

    Returns:
        (prompt_string, list_of_included_filenames, was_truncated_bool)
    """
    max_chars    = getattr(config, "MAX_FULL_DOC_CHARS", 400_000)
    total_chars  = 0
    doc_sections = []
    truncated    = False
    included     = []

    for filename, full_text in papers:
        if not full_text.strip():
            print("  Warning: no text extracted from {}.".format(filename))
            continue

        available = max_chars - total_chars
        if available <= 0:
            print("  Warning: context limit reached -- omitting {}.".format(filename))
            truncated = True
            continue

        if len(full_text) > available:
            print("  Warning: {} truncated to fit context window "
                  "({:,} of {:,} chars).".format(
                      filename, available, len(full_text)))
            full_text = full_text[:available]
            truncated = True

        doc_sections.append(
            "=" * 60 + "\n"
            "FULL TEXT OF PAPER: {}\n".format(filename) +
            "=" * 60 + "\n\n" +
            full_text
        )
        total_chars += len(full_text)
        included.append(filename)

    if not doc_sections:
        return None, [], False

    paper_block = "\n\n".join(doc_sections)

    if len(papers) == 1:
        paper_descriptor = "the paper '{}'".format(included[0])
    else:
        paper_descriptor = "{} papers: {}".format(
            len(included), ", ".join(included))

    # Tailor instructions to query type
    query_lower = query.lower().strip()
    is_summary  = query_lower in ("summarize", "summarise", "summary", "")

    if is_summary:
        task = (
            "Provide a comprehensive structured summary of {} covering:\n"
            "  1. The research problem and motivation\n"
            "  2. Materials, methods, and experimental design\n"
            "  3. Key results and quantitative findings\n"
            "  4. Conclusions and broader implications\n"
            "  5. Limitations acknowledged by the authors\n"
            "  6. Any equations, models, or algorithms central to the work"
        ).format(paper_descriptor)
    else:
        task = (
            "Answer the following question about {}:\n\n"
            "QUESTION:\n{}"
        ).format(paper_descriptor, query)

    instructions = (
        "You are a research assistant performing deep analysis of academic papers.\n"
        "You have been given the COMPLETE TEXT of {} below.\n\n"
        "Important:\n"
        "  - Read the full text carefully before answering.\n"
        "  - You have access to every section: abstract, methods, results,\n"
        "    figures, tables, equations, discussion, and conclusions.\n"
        "  - Cite specific sections, equations, tables, or page regions\n"
        "    when making claims.\n"
        "  - If comparing multiple papers, address each explicitly.\n"
        "  - Do not rely on general knowledge when the answer is in the text.\n"
    ).format(paper_descriptor)

    history_block = history.format_for_prompt() if history else ""

    prompt = (
        "{}"            # conversation history
        "{}\n\n"        # instructions
        "{}\n\n"        # full paper text(s)
        "{}\n\n"        # task (summary or question)
        "ANSWER:"
    ).format(history_block, instructions, paper_block, task)

    return prompt, included, truncated


def ask_fulldoc(query:      str,
                name_tokens: list,
                collection,
                history:    ConversationHistory = None) -> str:
    """
    Full-document mode entry point.

    1. Resolve paper name tokens to (filename, full_path) pairs.
    2. Extract complete text from each PDF.
    3. Build a prompt containing all full texts.
    4. Send to Claude with an elevated token budget.
    5. Display the answer and which files were included.
    """
    print(C.BOLD + "  [Full-document mode]" + C.RESET)

    resolved = resolve_paper_paths(name_tokens, collection)
    if not resolved:
        print("  No papers resolved.  Run 'list' to see available filenames.")
        return ""

    print("  Loading full text of {} paper(s)...".format(len(resolved)))
    papers_text = []
    for filename, filepath in resolved:
        print("    {}".format(filename))
        text = extract_full_text(filepath)
        papers_text.append((filename, text))

    prompt, included, truncated = build_fulldoc_prompt(query, papers_text, history)

    if prompt is None:
        print("  Could not extract usable text from the specified paper(s).")
        return ""

    if truncated:
        print(C.LABEL + "  Note: one or more papers were truncated to fit the "
              "context window." + C.RESET)

    char_count = len(prompt)
    print("  Sending {:,} chars (~{:,} tokens) to Claude...".format(
        char_count, char_count // 4))

    # Use a higher token budget for full-document analysis
    answer = ask_llm(prompt, max_tokens=8192)

    if answer is None:
        print("  Model refused or returned empty response.")
        answer = ask_llm_general(query)
        print(C.ANSWER + "[Fallback to general knowledge]\n" + answer + C.RESET)
    else:
        print("\n" + C.ANSWER + answer + C.RESET)

    print("\n=== SOURCE ===")
    print(C.LABEL + "  [Full-document mode -- complete PDF text sent to model]" + C.RESET)
    for fname in included:
        print(C.DIM + "  {}".format(fname) + C.RESET)

    if history and answer:
        history.add(
            "paper:{}: {}".format(",".join(included), query),
            answer,
        )

    return answer


# ---------------------------------------------------------------------------
# HyDE -- Hypothetical Document Embedding
# ---------------------------------------------------------------------------

def hyde_query_embedding(query: str, embedder) -> list:
    """
    Generate a hypothetical answer paragraph and embed it.
    The embedding of a plausible answer is closer in vector space to real
    answer chunks than the embedding of the question alone.
    Falls back to raw query embedding if the LLM call fails.
    """
    hyde_prompt = (
        "You are a scientific writing assistant.\n"
        "Write a short paragraph (4-6 sentences) that looks like it came from "
        "an academic paper and directly answers the following question.\n"
        "Write only the paragraph -- no preamble, no labels.\n\n"
        "Question: {}\n\nParagraph:".format(query)
    )
    hypothetical = _invoke(hyde_prompt, max_tokens=200)

    text_to_embed = (
        hypothetical.strip()
        if hypothetical and len(hypothetical.strip()) > 20
        else query
    )
    return embedder.encode(text_to_embed).tolist()


# ---------------------------------------------------------------------------
# Standard retrieval + reranking
# ---------------------------------------------------------------------------

def _build_where_filter(section_filter  = None,
                        folder_filter:  str = None,
                        filename_filter: str = None) -> dict | None:
    conditions = []
    if section_filter:
        conditions.append({"section": {"$in": section_filter}})
    if folder_filter:
        conditions.append({"folder": {"$eq": folder_filter}})
    if filename_filter:
        conditions.append({"filename": {"$eq": filename_filter}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def retrieve_and_rerank(query:           str,
                        collection,
                        embedder,
                        reranker,
                        section_filter   = None,
                        folder_filter:   str  = None,
                        filename_filter: str  = None,
                        n_retrieve:      int  = None,
                        n_final:         int  = None,
                        use_hyde:        bool = None) -> dict:
    """
    Stage 1 -- broad semantic search with SPECTER + optional HyDE.
    Stage 2 -- cross-encoder reranking of all candidates.

    Returns a dict with keys:
      documents, metadatas, distances, rerank_scores
    """
    n_retrieve = n_retrieve if n_retrieve is not None else config.N_RETRIEVE
    n_final    = n_final    if n_final    is not None else config.N_FINAL
    use_hyde   = use_hyde   if use_hyde   is not None else config.USE_HYDE

    query_vec = (
        hyde_query_embedding(query, embedder)
        if use_hyde
        else embedder.encode(query).tolist()
    )

    where = _build_where_filter(section_filter, folder_filter, filename_filter)

    raw = collection.query(
        query_embeddings = [query_vec],
        n_results        = n_retrieve,
        where            = where,
        include          = ["documents", "metadatas", "distances"],
    )

    docs  = raw["documents"][0]
    metas = raw["metadatas"][0]
    dists = raw["distances"][0]

    if not docs:
        return {
            "documents":    [[]],
            "metadatas":    [[]],
            "distances":    [[]],
            "rerank_scores": [],
        }

    scores = reranker.predict([(query, doc) for doc in docs]).tolist()

    ranked = sorted(
        zip(scores, docs, metas, dists),
        key     = lambda x: x[0],
        reverse = True,
    )[:n_final]

    return {
        "documents":    [[r[1] for r in ranked]],
        "metadatas":    [[r[2] for r in ranked]],
        "distances":    [[r[3] for r in ranked]],
        "rerank_scores": [r[0] for r in ranked],
    }


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------

def web_search(query: str, n_results: int = 3) -> tuple:
    """Try Tavily first; fall back to DuckDuckGo."""
    tavily_key = os.environ.get("TAVILY_API_KEY") or config.TAVILY_API_KEY

    if tavily_key:
        try:
            from tavily import TavilyClient
            tc      = TavilyClient(api_key=tavily_key)
            results = tc.search(query=query, search_depth="basic",
                                max_results=n_results)
            hits = results.get("results", [])
            if hits:
                return hits, "tavily"
        except Exception as exc:
            print("  Tavily failed: {} -- trying DuckDuckGo".format(exc))

    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=n_results))
        hits = [{"url":     r.get("href", ""),
                 "title":   r.get("title", ""),
                 "content": r.get("body", "")} for r in raw]
        return hits, "duckduckgo"
    except Exception as exc:
        print("  DuckDuckGo failed: {}".format(exc))

    return [], "none"


# ---------------------------------------------------------------------------
# Prompt building (standard RAG mode)
# ---------------------------------------------------------------------------

def build_rag_prompt(query:       str,
                     rag_results: dict,
                     web_results: list = None,
                     history:     ConversationHistory = None) -> str:
    paper_parts = []
    for i, (doc, meta) in enumerate(zip(
        rag_results["documents"][0],
        rag_results["metadatas"][0],
    )):
        source = "{} [{}]".format(
            meta.get("filename", "unknown"),
            meta.get("section",  ""),
        )
        paper_parts.append("[P{}] Source: {}\n{}".format(i + 1, source, doc))
    paper_context = "\n\n---\n\n".join(paper_parts)

    web_context = ""
    if web_results:
        web_parts = []
        for i, r in enumerate(web_results):
            web_parts.append("[W{}] URL: {}\nTitle: {}\n{}".format(
                i + 1,
                r.get("url",     ""),
                r.get("title",   ""),
                r.get("content", ""),
            ))
        web_context = "\n\n---\n\n".join(web_parts)

    if web_context:
        instructions = (
            "You are a research assistant with access to the user's papers "
            "and live web search results.\n\n"
            "Rules:\n"
            "  1. Answer primarily from PAPER CONTEXT where relevant.\n"
            "  2. Use WEB CONTEXT to supplement or add up-to-date information.\n"
            "  3. If neither is relevant, answer from general knowledge.\n"
            "  4. Label each part of your answer:\n"
            "       [From papers]       -- from the user's documents\n"
            "       [From web]          -- from web search results\n"
            "       [General knowledge] -- from training knowledge\n"
            "  5. Cite sources: filename + section for papers, URL for web.\n"
        )
        context_block = (
            "PAPER CONTEXT (retrieved chunks from the user's documents):\n{}\n\n"
            "WEB CONTEXT (live search results):\n{}"
        ).format(paper_context, web_context)
    else:
        instructions = (
            "You are a research assistant helping a scientist understand "
            "their own papers.\n\n"
            "Rules:\n"
            "  1. Answer primarily from PAPER CONTEXT where relevant.\n"
            "  2. Supplement with general knowledge where needed and label it:\n"
            "       [General knowledge, not from the provided papers]\n"
            "  3. If context is entirely irrelevant, say so:\n"
            "       [Not found in papers -- answering from general knowledge]\n"
            "  4. Cite filename and section for every claim.\n"
        )
        context_block = (
            "PAPER CONTEXT (retrieved chunks from the user's documents):\n{}"
        ).format(paper_context)

    history_block = history.format_for_prompt() if history else ""

    return (
        "{}"
        "{}\n\n"
        "{}\n\n"
        "QUESTION:\n{}\n\n"
        "ANSWER:"
    ).format(history_block, instructions, context_block, query)


# ---------------------------------------------------------------------------
# Source display
# ---------------------------------------------------------------------------

def _display_rag_sources(rag_results: dict,
                          web_results: list,
                          engine:      str):
    print("\n=== SOURCES ===")
    print(C.LABEL + "  -- Papers (chunk retrieval) --" + C.RESET)

    rerank_scores = rag_results.get("rerank_scores", [])

    for idx, (meta, dist) in enumerate(zip(
        rag_results["metadatas"][0],
        rag_results["distances"][0],
    )):
        rscore     = rerank_scores[idx] if idx < len(rerank_scores) else None
        score_str  = "cosine={:.3f}".format(1.0 - dist)
        if rscore is not None:
            score_str += "  rerank={:.3f}".format(rscore)

        print(C.DIM + "  {:<45} section={:<25} {}".format(
            meta.get("filename", "?"),
            meta.get("section",  "?"),
            score_str,
        ) + C.RESET)

    if web_results:
        print(C.LABEL + "  -- Web ({}) --".format(engine) + C.RESET)
        for r in web_results:
            print(C.DIM + "  {} | {}".format(
                r.get("url",   ""),
                r.get("title", ""),
            ) + C.RESET)


# ---------------------------------------------------------------------------
# Standard ask (RAG mode)
# ---------------------------------------------------------------------------

def ask(query:           str,
        embedder,
        reranker,
        collection,
        history:         ConversationHistory = None,
        section_filter                       = None,
        folder_filter:   str                 = None,
        filename_filter: str                 = None,
        force_web:       bool                = False) -> str:

    rag_results = retrieve_and_rerank(
        query           = query,
        collection      = collection,
        embedder        = embedder,
        reranker        = reranker,
        section_filter  = section_filter,
        folder_filter   = folder_filter,
        filename_filter = filename_filter,
    )

    web_results   = []
    search_engine = "none"

    if force_web or getattr(config, "WEB_SEARCH_ENABLED", False):
        print("  Searching web...")
        web_results, search_engine = web_search(query)
        if web_results:
            print("  Got {} result(s) via {}.".format(
                len(web_results), search_engine))
        else:
            print("  No web results returned.")

    prompt = build_rag_prompt(query, rag_results, web_results or None, history)
    answer = ask_llm(prompt)

    if answer is None:
        print("\n  [Model refused context -- retrying as general question]")
        answer = ask_llm_general(query)
        print(C.ANSWER +
              "[Not found in papers -- answering from general knowledge]\n" +
              answer + C.RESET)
        print("\n=== SOURCES ===")
        print("  (none -- general knowledge fallback)")
    else:
        print("\n" + C.ANSWER + answer + C.RESET)
        _display_rag_sources(rag_results, web_results, search_engine)

    if history and answer:
        history.add(query, answer)

    return answer


# ---------------------------------------------------------------------------
# Summarize via RAG (no specific paper specified)
# ---------------------------------------------------------------------------

def summarize_paper(target:    str,
                    embedder,
                    reranker,
                    collection,
                    history:   ConversationHistory = None):
    """
    Summarise using chunk retrieval.
    For deep single-paper summaries, prefer: paper:<filename>: summarize
    """
    if not target:
        print("  Usage: summarize:<filename>  or  summarize:all")
        return

    print("Summarizing via RAG: {}".format(
        target if target != "all" else "all papers"))

    filename_filter = None if target == "all" else target

    rag_results = retrieve_and_rerank(
        query           = (
            "main contribution research objective methods results "
            "conclusions findings implications"
        ),
        collection      = collection,
        embedder        = embedder,
        reranker        = reranker,
        section_filter  = ["abstract", "introduction", "conclusion",
                           "conclusions", "summary", "discussion",
                           "results", "findings"],
        filename_filter = filename_filter,
        n_retrieve      = 30,
        n_final         = 10,
        use_hyde        = False,
    )

    if not rag_results["documents"][0]:
        print("  No chunks found for '{}'.".format(target))
        print("  Tip: for a deep single-paper summary use:")
        print("       paper:{}: summarize".format(target))
        return

    prompt = build_rag_prompt(
        query = (
            "Provide a structured summary covering:\n"
            "  1. The research problem and motivation\n"
            "  2. The methods and experimental design\n"
            "  3. The main results and findings\n"
            "  4. The conclusions and broader implications"
        ),
        rag_results = rag_results,
        history     = history,
    )

    answer = ask_llm(prompt)
    if answer:
        print("\n" + C.ANSWER + answer + C.RESET)
        _display_rag_sources(rag_results, [], "none")
        if history:
            history.add("summarize: " + target, answer)
    else:
        print("  Could not generate summary.")


# ---------------------------------------------------------------------------
# List indexed papers
# ---------------------------------------------------------------------------

def list_papers(collection):
    """Print a deduplicated table of all indexed filenames and folders."""
    results = collection.get(include=["metadatas"])
    seen: dict = {}
    for meta in results["metadatas"]:
        fname = meta.get("filename", "?")
        if fname not in seen:
            seen[fname] = meta.get("folder", "?")

    if not seen:
        print("  No papers indexed yet.  Run: python ingest.py")
        return

    print("\n{} paper{} indexed:\n".format(
        len(seen), "" if len(seen) == 1 else "s"))
    for fname in sorted(seen):
        print("  {:<55} {}".format(fname, seen[fname]))
    print()


# ---------------------------------------------------------------------------
# Model selector
# ---------------------------------------------------------------------------

def get_available_models() -> list:
    try:
        client   = boto3.client("bedrock", region_name=config.BEDROCK_REGION)
        response = client.list_inference_profiles()
        return [
            p["inferenceProfileId"]
            for p in response["inferenceProfileSummaries"]
            if "claude" in p["inferenceProfileId"].lower()
        ]
    except Exception as exc:
        print("  Could not fetch model list: {}".format(exc))
        return []


def select_model() -> str:
    models = get_available_models()

    print("\n--- Model Selector ---")
    print("Current : {}".format(config.BEDROCK_MODEL))

    if models:
        print("\nAvailable Claude inference profiles:")
        for i, m in enumerate(models):
            marker = "  <-- current" if m == config.BEDROCK_MODEL else ""
            print("  [{}] {}{}".format(i + 1, m, marker))
    else:
        print("  (Could not retrieve model list from Bedrock.)")

    raw = input("\nEnter number, full profile ID, or Enter to keep current: ").strip()

    if not raw:
        return config.BEDROCK_MODEL

    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(models):
            print("Switched to: {}".format(models[idx]))
            return models[idx]
        print("  Invalid number.  Valid range: 1-{}.".format(len(models)))
        return config.BEDROCK_MODEL

    print("Switched to: {}".format(raw))
    return raw


# ---------------------------------------------------------------------------
# Query parser  (composable prefixes)
# ---------------------------------------------------------------------------

def parse_query(raw: str) -> tuple:
    """
    Parse all prefix flags from the raw input string.

    Returns:
        (query, section_filter, folder_filter, force_web,
         summarize_target, paper_tokens)

    paper_tokens is a list of filename tokens when paper: is used,
    otherwise None.

    Prefix reference
    ----------------
    paper:<name>:              full-document mode for one paper
    paper:<n1>,<n2>:           full-document mode for multiple papers
    summarize:<target>         RAG-based summary (exclusive)
    methods:                   restrict to methods sections
    results:                   restrict to results sections
    folder:<name>:             restrict to a topic folder
    web:                       force web search

    All prefixes except summarize: and paper: are composable.
    """
    query = raw.strip()

    # -- paper:  full-document mode (exclusive) -------------------------
    # Syntax: paper:<name>: <question>
    #     or: paper:<n1>,<n2>: <question>
    #     or: paper:<name>: summarize
    m = re.match(r"^paper:(.+?):\s*(.*)", query, re.IGNORECASE)
    if m:
        names_raw   = m.group(1).strip()
        rest_query  = m.group(2).strip()
        paper_tokens = [t.strip() for t in names_raw.split(",") if t.strip()]
        return rest_query, None, None, False, None, paper_tokens

    # -- summarize:  RAG summary (exclusive) ----------------------------
    if query.lower().startswith("summarize:"):
        return None, None, None, False, query[10:].strip(), None

    # -- composable flags -----------------------------------------------
    section_filter = None
    folder_filter  = None
    force_web      = False

    if re.search(r"\bweb:\s*", query):
        force_web = True
        query     = re.sub(r"\bweb:\s*", "", query)

    if re.search(r"\bmethods:\s*", query):
        section_filter = ["methods", "methodology",
                          "experimental setup", "materials and methods"]
        query = re.sub(r"\bmethods:\s*", "", query)

    if re.search(r"\bresults:\s*", query) and section_filter is None:
        section_filter = ["results", "findings", "experiments", "evaluation"]
        query = re.sub(r"\bresults:\s*", "", query)

    fm = re.search(r"\bfolder:([^:\s]+):", query)
    if fm:
        folder_filter = fm.group(1).strip()
        query         = re.sub(r"\bfolder:[^:\s]+:\s*", "", query)

    return query.strip(), section_filter, folder_filter, force_web, None, None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    embedder, reranker, collection = load_resources()
    history = ConversationHistory(max_turns=5)

    web_status = "on" if getattr(config, "WEB_SEARCH_ENABLED", False) else "off"

    print("=" * 62)
    print("  Academic Paper RAG")
    print("=" * 62)
    print("  Backend      : AWS Bedrock")
    print("  Model        : {}".format(config.BEDROCK_MODEL))
    print("  Embeddings   : {}".format(config.EMBED_MODEL))
    print("  Reranker     : {}".format(config.RERANK_MODEL))
    print("  HyDE         : {}".format("on" if config.USE_HYDE else "off"))
    print("  Web search   : {} (toggle: webon / weboff)".format(web_status))
    print("  Chunks in DB : {}".format(collection.count()))
    print("=" * 62)
    print()
    print("Query prefixes:")
    print("  paper:<filename>: <question>   deep analysis -- full PDF sent to model")
    print("  paper:<f1>,<f2>: <question>    deep analysis of multiple papers")
    print("  paper:<filename>: summarize    deep summary of one paper")
    print("  methods:                       search only methods sections")
    print("  results:                       search only results sections")
    print("  folder:<name>:                 search only that topic folder")
    print("  web:                           force web search for this query")
    print("  summarize:<filename>           RAG-based summary (chunk retrieval)")
    print("  summarize:all                  RAG-based summary across all papers")
    print()
    print("Commands:")
    print("  list       show all indexed papers")
    print("  model      switch Claude model interactively")
    print("  history    show conversation history")
    print("  clear      clear conversation history")
    print("  webon/off  toggle web search globally")
    print("  quit       exit")
    print()

    while True:
        try:
            raw = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not raw:
            continue

        cmd = raw.lower()

        if cmd == "quit":
            break

        if cmd == "list":
            list_papers(collection)
            continue

        if cmd == "history":
            block = history.format_for_prompt()
            print(block if block else "  (no history yet)")
            continue

        if cmd == "clear":
            history.clear()
            print("  Conversation history cleared.")
            continue

        if cmd == "model":
            new_model = select_model()
            if new_model != config.BEDROCK_MODEL:
                config.BEDROCK_MODEL = new_model
                reset_bedrock_client()
            print("Using model: {}".format(config.BEDROCK_MODEL))
            continue

        if cmd == "webon":
            config.WEB_SEARCH_ENABLED = True
            print("  Web search enabled.")
            continue

        if cmd == "weboff":
            config.WEB_SEARCH_ENABLED = False
            print("  Web search disabled.")
            continue

        (query,
         section_filter,
         folder_filter,
         force_web,
         summarize_target,
         paper_tokens) = parse_query(raw)

        # -- Full-document mode -----------------------------------------
        if paper_tokens is not None:
            ask_fulldoc(
                query       = query,
                name_tokens = paper_tokens,
                collection  = collection,
                history     = history,
            )

        # -- RAG summary ------------------------------------------------
        elif summarize_target is not None:
            summarize_paper(
                target     = summarize_target,
                embedder   = embedder,
                reranker   = reranker,
                collection = collection,
                history    = history,
            )

        # -- Standard RAG -----------------------------------------------
        else:
            ask(
                query          = query,
                embedder       = embedder,
                reranker       = reranker,
                collection     = collection,
                history        = history,
                section_filter = section_filter,
                folder_filter  = folder_filter,
                force_web      = force_web,
            )

        print()
