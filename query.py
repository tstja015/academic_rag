# query.py

"""
query.py -- Interactive RAG query interface for academic papers.

Retrieval modes
---------------
Standard (default):
    Two-stage retrieval -- embeddings via HyDE -> top-N candidates ->
    cross-encoder reranking -> best chunks sent to Claude.

Full-document mode  (paper: prefix):
    The complete text of the specified paper(s) is extracted from the PDF
    and sent directly in the context window.

    Syntax:
        paper:: 
        paper:,: 
        paper:: summarize
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
# Metadata normalisation  (handles old-schema chunks gracefully)
# ---------------------------------------------------------------------------

def _normalize_meta(meta: dict) -> dict:
    """
    Map old-schema metadata keys (source, path, chunk) to the current schema
    (filename, full_path, section, folder, chunk_index).  Modifies in place.
    """
    if "filename" not in meta and "source" in meta:
        meta["filename"] = meta["source"]
    if "full_path" not in meta and "path" in meta:
        meta["full_path"] = meta["path"]
    if "chunk_index" not in meta and "chunk" in meta:
        meta["chunk_index"] = meta["chunk"]
    if "section" not in meta:
        meta["section"] = "unknown"
    if "folder" not in meta:
        path = meta.get("full_path", "")
        parts = path.replace("\\", "/").split("/")
        # Try to extract folder from path -- look for 'doc/' marker
        if "doc" in parts and parts.index("doc") + 1 < len(parts):
            meta["folder"] = parts[parts.index("doc") + 1]
        else:
            meta["folder"] = "unknown"
    return meta


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough estimate: ~4 chars per token for English."""
    return len(text) // 4


# ---------------------------------------------------------------------------
# Conversation history (multi-turn native messages)
# ---------------------------------------------------------------------------

class ConversationHistory:
    def __init__(self, max_turns: int = 8, max_answer_chars: int = 4000):
        self.turns:            list = []
        self.max_turns:        int  = max_turns
        self.max_answer_chars: int  = max_answer_chars

    def add(self, question: str, answer: str):
        self.turns.append({"q": question, "a": answer})
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def clear(self):
        self.turns = []

    def as_messages(self) -> list:
        """Return history as alternating user/assistant message dicts."""
        msgs = []
        for t in self.turns:
            msgs.append({"role": "user", "content": t["q"]})
            a = t["a"]
            if len(a) > self.max_answer_chars:
                a = a[:self.max_answer_chars] + "\n[... truncated ...]"
            msgs.append({"role": "assistant", "content": a})
        return msgs

    def format_for_prompt(self) -> str:
        """Legacy text-based format (used for display only)."""
        if not self.turns:
            return ""
        lines = ["CONVERSATION HISTORY (oldest first):"]
        for t in self.turns:
            a = t["a"]
            if len(a) > self.max_answer_chars:
                a = a[:self.max_answer_chars] + "\n[... truncated ...]"
            lines.append("  Q: {}".format(t["q"]))
            lines.append("  A: {}".format(a))
        return "\n".join(lines) + "\n\n"

    def total_chars(self) -> int:
        return sum(len(t["q"]) + len(t["a"]) for t in self.turns)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def load_resources():
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
# LLM invocation (multi-turn native)
# ---------------------------------------------------------------------------

def _invoke_multiturn(messages: list, system: str = None,
                      max_tokens: int = 4096) -> str | None:
    """
    Send a multi-turn conversation to Bedrock.

    Parameters
    ----------
    messages : list of {"role": "user"|"assistant", "content": str}
    system   : optional system prompt string
    max_tokens : max output tokens
    """
    client = _get_bedrock_client()

    body_dict = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages":   messages,
    }
    if system:
        body_dict["system"] = system

    body = json.dumps(body_dict)

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


def _invoke(prompt: str, max_tokens: int = 4096,
            system: str = None,
            history: "ConversationHistory" = None) -> str | None:
    """
    Single-turn convenience wrapper.  If history is provided, builds a
    proper multi-turn message list.
    """
    messages = []
    if history:
        messages = history.as_messages()
    messages.append({"role": "user", "content": prompt})

    sys_prompt = system or getattr(config, "SYSTEM_PROMPT", None)
    return _invoke_multiturn(messages, system=sys_prompt, max_tokens=max_tokens)


def ask_llm(prompt: str, max_tokens: int = 4096,
            history: "ConversationHistory" = None) -> str | None:
    return _invoke(prompt, max_tokens=max_tokens, history=history)


def ask_llm_general(question: str,
                    history: "ConversationHistory" = None) -> str:
    prompt = (
        "Answer the following question as clearly and accurately as you can.\n\n"
        "QUESTION:\n{}\n\nANSWER:".format(question)
    )
    result = _invoke(prompt, history=history)
    return result if result else "Could not get a response from the model."


# ---------------------------------------------------------------------------
# Output verification
# ---------------------------------------------------------------------------

def verify_output(original_query: str, answer: str) -> str | None:
    """Ask the model to self-check its own output for completeness."""
    verify_prompt = (
        "You previously answered this question:\n\n"
        "QUESTION: {}\n\n"
        "YOUR ANSWER (abbreviated):\n{}\n\n"
        "Review your answer for:\n"
        "1. Did you address EVERY part of the user's request?\n"
        "2. Are all equations faithful to the source, or did you substitute "
        "   familiar forms from memory?\n"
        "3. Are any parameters, tables, or edge cases missing?\n"
        "4. Are there any numerical values you stated without verifying "
        "   against the paper?\n\n"
        "List specific problems found. If none, say 'No issues found.'"
    ).format(original_query, answer[:6000])

    return _invoke(verify_prompt, max_tokens=2000)


# ---------------------------------------------------------------------------
# Full-document mode
# ---------------------------------------------------------------------------

def resolve_paper_paths(name_tokens: list, collection) -> list:
    path_map: dict = {}
    batch_size = 5000
    offset = 0
    total = collection.count()

    while offset < total:
        batch = collection.get(include=["metadatas"], limit=batch_size, offset=offset)
        for m in batch["metadatas"]:
            m = _normalize_meta(m)
            fname = m.get("filename", "")
            fpath = m.get("full_path", "")
            if fname and fpath and fname not in path_map:
                path_map[fname] = fpath
        offset += batch_size

    resolved = []
    for token in name_tokens:
        token_lower = token.lower()
        match = None

        if token in path_map:
            match = (token, path_map[token])

        if match is None:
            for fname, fpath in path_map.items():
                if fname.lower() == token_lower:
                    match = (fname, fpath)
                    break

        if match is None:
            candidates = [
                (fname, fpath)
                for fname, fpath in path_map.items()
                if token_lower in fname.lower()
            ]
            if len(candidates) == 1:
                match = candidates[0]
            elif len(candidates) > 1:
                candidates.sort(key=lambda x: len(x[0]))
                print("  Ambiguous token '{}' -- matched multiple files:".format(token))
                for i, (fn, _) in enumerate(candidates):
                    print("    [{}] {}".format(i + 1, fn))
                print("  Using: {}  (use full filename to disambiguate)".format(
                    candidates[0][0]))
                match = candidates[0]

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

    seen   = set()
    unique = []
    for item in resolved:
        if item[0] not in seen:
            seen.add(item[0])
            unique.append(item)

    return unique


def extract_full_text(filepath: str) -> str:
    try:
        return pymupdf4llm.to_markdown(filepath)
    except Exception as exc:
        print("  Error reading {}: {}".format(filepath, exc))
        return ""


def build_fulldoc_prompt(query:    str,
                          papers:   list,
                          history:  ConversationHistory = None) -> tuple:
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

    if len(included) == 1:
        paper_descriptor = "the paper '{}'".format(included[0])
    else:
        paper_descriptor = "{} papers: {}".format(
            len(included), ", ".join(included))

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

    prompt = (
        "{}\n\n"
        "{}\n\n"
        "{}\n\n"
        "ANSWER:"
    ).format(instructions, paper_block, task)

    # Token budget warning
    token_est   = estimate_tokens(prompt)
    model_limit = 200_000
    if token_est > model_limit * 0.9:
        print(C.LABEL +
              "  WARNING: prompt is ~{:,} tokens "
              "({:.0%} of context window)".format(
                  token_est, token_est / model_limit) +
              C.RESET)

    return prompt, included, truncated


def ask_fulldoc(query:      str,
                name_tokens: list,
                collection,
                history:    ConversationHistory = None) -> str:
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

    answer = ask_llm(prompt, max_tokens=8192, history=history)

    if answer is None:
        print("  Model refused or returned empty response.")
        answer = ask_llm_general(query, history=history)
        print(C.ANSWER + "[Fallback to general knowledge]\n" + answer + C.RESET)
    else:
        print("\n" + C.ANSWER + answer + C.RESET)

        # Self-verification for substantial outputs
        if len(answer) > 2000:
            print(C.LABEL + "\n  [Running self-verification...]" + C.RESET)
            issues = verify_output(query, answer)
            if issues and "no issues found" not in issues.lower():
                print(C.LABEL + "  Self-check found potential issues:" + C.RESET)
                print(C.DIM + issues + C.RESET)
            else:
                print(C.DIM + "  Self-check: no issues found." + C.RESET)

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
    hyde_prompt = (
        "Given the user's question, generate a short academic paper TITLE and "
        "ABSTRACT (3-4 sentences) that would answer this question.\n"
        "Write ONLY the title and abstract — no labels, no preamble.\n\n"
        "Question: {}\n\nTitle and Abstract:".format(query)
    )
    hypothetical = _invoke(hyde_prompt, max_tokens=200)

    if hypothetical and len(hypothetical.strip()) > 20:
        text_to_embed = hypothetical.strip()
        print("  [HyDE] Generated search document ({} chars)".format(
            len(text_to_embed)))
    else:
        print("  [HyDE fallback -- using raw query embedding]")
        text_to_embed = query

    return embedder.encode(text_to_embed).tolist()


# ---------------------------------------------------------------------------
# Expand acronyms before embedding
# ---------------------------------------------------------------------------

def expand_query(query: str) -> str:
    """Ask the LLM to expand acronyms and clarify ambiguous terms."""
    prompt = (
        "Expand all acronyms and rewrite the following query as a precise "
        "scientific search query. Keep it under 30 words.\n"
        "Write ONLY the rewritten query — nothing else.\n\n"
        "Original: {}\n\nRewritten:".format(query)
    )
    result = _invoke(prompt, max_tokens=100)
    if result and len(result.strip()) > 10:
        expanded = result.strip().split("\n")[0]  # take first line only
        print("  [Query expansion] {} -> {}".format(query, expanded))
        return expanded
    return query

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
    n_retrieve = n_retrieve if n_retrieve is not None else config.N_RETRIEVE
    n_final    = n_final    if n_final    is not None else config.N_FINAL
    use_hyde   = use_hyde   if use_hyde   is not None else config.USE_HYDE

    # -- Expand acronyms / clarify ambiguous terms --------------------------
    search_query = expand_query(query)

    query_vec = (
        hyde_query_embedding(search_query, embedder)
        if use_hyde
        else embedder.encode(search_query).tolist()
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

    # -- Rerank against the ORIGINAL query (not expanded) -------------------
    scores = reranker.predict([(query, doc) for doc in docs]).tolist()

    threshold = getattr(config, "RERANK_THRESHOLD", 0.0)

    ranked = sorted(
        zip(scores, docs, metas, dists),
        key     = lambda x: x[0],
        reverse = True,
    )

    ranked = [r for r in ranked if r[0] >= threshold][:n_final]

    if not ranked:
        return {
            "documents":    [[]],
            "metadatas":    [[]],
            "distances":    [[]],
            "rerank_scores": [],
        }

    return {
        "documents":    [[r[1] for r in ranked]],
        "metadatas":    [[_normalize_meta(r[2]) for r in ranked]],
        "distances":    [[r[3] for r in ranked]],
        "rerank_scores": [r[0] for r in ranked],
    }


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------

def web_search(query: str, n_results: int = 3) -> tuple:
    tavily_key = os.environ.get("TAVILY_API_KEY") or getattr(config, "TAVILY_API_KEY", "")

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
                     web_results: list = None) -> str:
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

    return (
        "{}\n\n"
        "{}\n\n"
        "QUESTION:\n{}\n\n"
        "ANSWER:"
    ).format(instructions, context_block, query)


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

    # Early exit if no relevant chunks survived reranking
    if not rag_results["documents"][0]:
        print("  No relevant chunks found (all below rerank threshold).")
        answer = ask_llm_general(query, history=history)
        print(C.ANSWER +
              "[Not found in papers -- answering from general knowledge]\n" +
              answer + C.RESET)
        if history and answer:
            history.add(query, answer)
        return answer

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

    prompt = build_rag_prompt(query, rag_results, web_results or None)
    answer = ask_llm(prompt, history=history)

    if answer is None:
        print("\n  [Model refused context -- retrying as general question]")
        answer = ask_llm_general(query, history=history)
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
# Summarize via RAG
# ---------------------------------------------------------------------------

def summarize_paper(target:    str,
                    embedder,
                    reranker,
                    collection,
                    history:   ConversationHistory = None):
    if not target:
        print("  Usage: summarize:  or  summarize:all")
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
    )

    answer = ask_llm(prompt, history=history)
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

def list_papers(collection, pattern: str = None):
    seen: dict = {}
    batch_size = 5000
    offset = 0
    total = collection.count()

    while offset < total:
        results = collection.get(
            include=["metadatas"],
            limit=batch_size,
            offset=offset,
        )
        for meta in results["metadatas"]:
            meta = _normalize_meta(meta)
            fname = meta.get("filename", "?")
            if fname not in seen:
                seen[fname] = meta.get("folder", "?")
        offset += batch_size

    if not seen:
        print("  No papers indexed yet.  Run: python ingest.py")
        return

    # Apply filter if given
    if pattern:
        import fnmatch
        pattern_lower = pattern.lower()
        # Support both wildcard and plain substring search
        if "*" in pattern or "?" in pattern:
            matched = {f: v for f, v in seen.items()
                       if fnmatch.fnmatch(f.lower(), pattern_lower)}
        else:
            matched = {f: v for f, v in seen.items()
                       if pattern_lower in f.lower()}
    else:
        matched = seen

    if not matched:
        print("  No papers matched '{}'.".format(pattern))
        print("  {} total papers indexed.".format(len(seen)))
        return

    print("\n{} paper{} matched{}:\n".format(
        len(matched),
        "" if len(matched) == 1 else "s",
        " '{}'".format(pattern) if pattern else "",
    ))
    for fname in sorted(matched):
        print("  {:<55} {}".format(fname, matched[fname]))
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
    query = raw.strip()

    m = re.match(r"^paper:(.+?):\s*(.*)", query, re.IGNORECASE)
    if m:
        names_raw    = m.group(1).strip()
        rest_query   = m.group(2).strip()
        paper_tokens = [t.strip() for t in names_raw.split(",") if t.strip()]
        return rest_query, None, None, False, None, paper_tokens

    if query.lower().startswith("summarize:"):
        return None, None, None, False, query[10:].strip(), None

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
    history = ConversationHistory(max_turns=8, max_answer_chars=4000)

    web_status = "on" if getattr(config, "WEB_SEARCH_ENABLED", False) else "off"

    print("=" * 62)
    print("  Academic Paper RAG")
    print("=" * 62)
    print("  Backend      : AWS Bedrock")
    print("  Model        : {}".format(config.BEDROCK_MODEL))
    print("  Embeddings   : {}".format(config.EMBED_MODEL))
    print("  Reranker     : {}".format(config.RERANK_MODEL))
    print("  HyDE         : {}".format("on" if config.USE_HYDE else "off"))
    print("  Rerank min   : {}".format(
        getattr(config, "RERANK_THRESHOLD", 0.0)))
    print("  Web search   : {} (toggle: webon / weboff)".format(web_status))
    print("  Chunks in DB : {}".format(collection.count()))
    print("  Verify mode  : on (auto self-check for long outputs)")
    print("=" * 62)
    print()
    print("Query prefixes:")
    print("  paper::     deep analysis -- full PDF sent to model")
    print("  paper:,:  deep analysis of multiple papers")
    print("  paper:: summarize     deep summary of one paper")
    print("  methods:          search only methods sections")
    print("  results:          search only results sections")
    print("  folder::    search only that topic folder")
    print("  web:              force web search for this query")
    print("  summarize:        RAG-based summary (chunk retrieval)")
    print("  summarize:all               RAG-based summary across all papers")
    print()
    print("Commands:")
    print("  list       show all indexed papers")
    print("  list                   filter by name (substring)")
    print("  list *                 filter by name (wildcard)")
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

     #   if cmd == "list":
        if cmd.startswith("list"):
            pattern = cmd[4:].strip() or None
            list_papers(collection, pattern)
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

        if paper_tokens is not None:
            ask_fulldoc(
                query       = query,
                name_tokens = paper_tokens,
                collection  = collection,
                history     = history,
            )

        elif summarize_target is not None:
            summarize_paper(
                target     = summarize_target,
                embedder   = embedder,
                reranker   = reranker,
                collection = collection,
                history    = history,
            )

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
