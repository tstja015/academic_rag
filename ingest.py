
# ingest.py
"""
Ingest academic PDFs into ChromaDB with:
  - Resume capability (skips already-ingested files via SHA-256 hash)
  - Path-indexed fast-skip (no disk read or ChromaDB call for known files)
  - Rename/move detection (updates metadata in place -- no re-embedding)
  - Idempotent re-ingestion (removes stale chunks before inserting)
  - Orphan cleanup (removes chunks whose source PDFs no longer exist)
  - Per-file error isolation (one bad PDF won't kill the run)
  - Section detection for chunk metadata
  - OneDrive eviction after each successful ingest (WSL-aware)
  - OneDrive log cleanup at start, periodically, and at end
  - Periodic disk space reporting
"""

import os
import re
import sys
import json
import hashlib
import logging
from pathlib import Path
from tqdm import tqdm

import argparse

import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

import config
from onedrive_utils import (
    free_onedrive_file,
    free_onedrive_files_bulk,
    get_file_size_mb,
    log_drive_free_space,
    cleanup_onedrive_logs,
)

# -- logging ----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def load_progress() -> dict:
    """
    Load ingest progress from disk.

    Schema
    ------
    {
        "completed":  ["<sha256>", ...],          # hashes of ingested files
        "path_index": {"/abs/path": ""},  # fast-skip lookup
        "failed":     {"/abs/path": ""}    # paths that errored
    }

    Automatically migrates the old schema (no path_index key) on first load.
    """
    pf = getattr(config, "PROGRESS_FILE", "./db/ingest_progress.json")
    if os.path.exists(pf):
        with open(pf) as f:
            data = json.load(f)

        # -- migrate old schema (list-only) to path-indexed schema ----------
        if "path_index" not in data:
            log.info(
                "Migrating progress file to path-indexed schema "
                "(first run only -- subsequent runs will be much faster)"
            )
            data["path_index"] = {}
            save_progress(data)

        return data

    return {"completed": [], "path_index": {}, "failed": {}}


def save_progress(progress: dict):
    pf = getattr(config, "PROGRESS_FILE", "./db/ingest_progress.json")
    os.makedirs(os.path.dirname(pf), exist_ok=True)
    with open(pf, "w") as f:
        json.dump(progress, f, indent=2)


def file_hash(path: str) -> str:
    """SHA-256 of file bytes -- stable document ID regardless of filename."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# PDF collection
# ---------------------------------------------------------------------------

def collect_pdfs() -> list:
    pdfs = []
    for d in config.PAPER_DIRS:
        if not os.path.isdir(d):
            log.warning("Directory not found, skipping: %s", d)
            continue
        for root, _, files in os.walk(d):
            for fn in files:
                if fn.lower().endswith(".pdf"):
                    pdfs.append(os.path.join(root, fn))
    unique = sorted(set(pdfs))
    log.info(
        "Found %d unique PDFs across %d directories",
        len(unique), len(config.PAPER_DIRS),
    )
    return unique


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text(path: str) -> str:
    """Extract text from PDF using PyMuPDF. Raises on corrupt files."""
    doc = fitz.open(path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    text = "\n".join(pages)
    if not text.strip():
        raise ValueError("No extractable text found (scanned PDF?)")
    return text


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

_SECTION_PATTERNS = [
    (r'\b(abstract)\b',                             'abstract'),
    (r'\b(introduction)\b',                         'introduction'),
    (r'\b(background)\b',                           'background'),
    (r'\b(related\s+work)\b',                       'related work'),
    (r'\b(literature\s+review)\b',                  'literature review'),
    (r'\b(materials?\s+and\s+methods?)\b',          'materials and methods'),
    (r'\b(methods?|methodology)\b',                 'methods'),
    (r'\b(experimental\s+setup)\b',                 'experimental setup'),
    (r'\b(experiments?)\b',                         'experiments'),
    (r'\b(results?\s+and\s+discussion)\b',          'results and discussion'),
    (r'\b(results?)\b',                             'results'),
    (r'\b(findings)\b',                             'findings'),
    (r'\b(discussion)\b',                           'discussion'),
    (r'\b(conclusion|conclusions|concluding)\b',    'conclusions'),
    (r'\b(summary)\b',                              'summary'),
    (r'\b(acknowledg[e]?ments?)\b',                 'acknowledgements'),
    (r'\b(references|bibliography)\b',              'references'),
    (r'\b(appendix|supplementary)\b',               'appendix'),
]


def detect_section(text: str) -> str:
    head = text[:200].lower()
    for pattern, label in _SECTION_PATTERNS:
        if re.search(pattern, head, re.IGNORECASE):
            return label
    return "body"


# ---------------------------------------------------------------------------
# Folder extraction
# ---------------------------------------------------------------------------

def extract_folder(path: str) -> str:
    abs_path = os.path.abspath(path)
    for base_dir in config.PAPER_DIRS:
        abs_base = os.path.abspath(base_dir)
        if abs_path.startswith(abs_base + os.sep):
            relative = os.path.relpath(abs_path, abs_base)
            parts = Path(relative).parts
            if len(parts) > 1:
                return parts[0]
            return Path(abs_base).name
    return Path(path).parent.name


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str) -> list:
    sentences   = re.split(r'(?<=[.!?])\s+', text)
    chunks      = []
    current     = []
    current_len = 0

    for sent in sentences:
        words = sent.split()
        if current_len + len(words) > config.CHUNK_SIZE and current:
            chunks.append(" ".join(current))
            overlap_words = " ".join(current).split()[-config.CHUNK_OVERLAP:]
            current     = overlap_words + words
            current_len = len(current)
        else:
            current.extend(words)
            current_len += len(words)

    if current:
        chunks.append(" ".join(current))

    return chunks


# ---------------------------------------------------------------------------
# Rename / move detection and metadata update
# ---------------------------------------------------------------------------

def check_and_update_metadata(fhash: str, current_path: str,
                               collection) -> str:
    """
    Check whether stored metadata matches the current file path.

    Returns
    -------
    "match"   -- everything is current, skip
    "updated" -- metadata was stale and has been updated in place
    "missing" -- no chunks found for this hash, needs full ingest
    "error"   -- could not check (old schema), treat as skip
    """
    current_filename = Path(current_path).name
    current_abspath  = os.path.abspath(current_path)
    current_folder   = extract_folder(current_path)

    try:
        existing = collection.get(
            where   = {"file_hash": {"$eq": fhash}},
            include = ["metadatas"],
        )
    except Exception:
        return "error"

    if not existing["ids"]:
        log.info(
            "No chunks found for hash %s... -- will ingest %s",
            fhash[:12], current_filename,
        )
        return "missing"

    stored_meta     = existing["metadatas"][0]
    stored_filename = stored_meta.get("filename", "")
    stored_path     = stored_meta.get("full_path", "")

    if stored_filename == current_filename and stored_path == current_abspath:
        return "match"

    if stored_filename != current_filename:
        log.info(
            "Rename detected: '%s' -> '%s'  (updating metadata)",
            stored_filename, current_filename,
        )
    elif stored_path != current_abspath:
        log.info(
            "Move detected: '%s' -> '%s'  (updating metadata)",
            stored_path, current_abspath,
        )

    updated_metas = []
    for meta in existing["metadatas"]:
        meta["filename"]  = current_filename
        meta["full_path"] = current_abspath
        meta["folder"]    = current_folder
        updated_metas.append(meta)

    collection.update(
        ids       = existing["ids"],
        metadatas = updated_metas,
    )

    log.info("  Updated metadata on %d chunks.", len(existing["ids"]))
    return "updated"


# ---------------------------------------------------------------------------
# Single-file ingest
# ---------------------------------------------------------------------------

def ingest_one(path: str, collection, embed_model, fhash: str):
    """Extract, chunk, embed, and store one PDF. Idempotent."""

    # -- Remove any existing chunks for this file hash ----------------------
    try:
        existing = collection.get(
            where   = {"file_hash": {"$eq": fhash}},
            include = [],
        )
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            log.info(
                "  Removed %d stale chunks (hash %s...)",
                len(existing["ids"]), fhash[:12],
            )
    except Exception:
        pass

    # -- Extract and chunk --------------------------------------------------
    text   = extract_text(path)
    chunks = chunk_text(text)

    if not chunks:
        raise ValueError("Chunking produced no output")

    filename = Path(path).name
    folder   = extract_folder(path)

    # -- Compute stats for logging ------------------------------------------
    word_count    = len(text.split())
    char_count    = len(text)
    page_count    = fitz.open(path).page_count
    file_size_mb  = os.path.getsize(path) / (1024 * 1024)
    avg_chunk_len = sum(len(c.split()) for c in chunks) / len(chunks)

    # -- Embed and store ----------------------------------------------------
    batch_size = 32
    for i in range(0, len(chunks), batch_size):
        batch      = chunks[i:i + batch_size]
        embeddings = embed_model.encode(
            batch, show_progress_bar=False
        ).tolist()

        ids       = [f"{fhash}_{i + j}" for j in range(len(batch))]
        metadatas = [
            {
                "filename":     filename,
                "full_path":    os.path.abspath(path),
                "folder":       folder,
                "section":      detect_section(batch[j]),
                "chunk_index":  i + j,
                "total_chunks": len(chunks),
                "file_hash":    fhash,
            }
            for j in range(len(batch))
        ]

        collection.add(
            ids        = ids,
            documents  = batch,
            embeddings = embeddings,
            metadatas  = metadatas,
        )

    # -- Section breakdown --------------------------------------------------
    section_counts = {}
    for c in chunks:
        sec = detect_section(c)
        section_counts[sec] = section_counts.get(sec, 0) + 1

    section_summary = ", ".join(
        f"{s}:{n}" for s, n in sorted(section_counts.items())
    )

    log.info(
        "Ingested %-50s  |  chunks: %3d  |  pages: %3d  |  words: %6d  |  "
        "chars: %7d  |  size: %.2f MB  |  avg chunk: %3d words  |  "
        "folder: %s  |  sections: {%s}",
        filename,
        len(chunks),
        page_count,
        word_count,
        char_count,
        file_size_mb,
        int(avg_chunk_len),
        folder,
        section_summary,
    )


# ---------------------------------------------------------------------------
# Orphan cleanup
# ---------------------------------------------------------------------------

def cleanup_orphans(collection, live_hashes: set, progress: dict) -> int:
    """
    Remove chunks from ChromaDB whose file_hash no longer corresponds to
    any PDF on disk.  Also cleans the progress file.
    """
    log.info("Scanning for orphaned chunks...")

    batch_size    = 10_000
    offset        = 0
    stored_hashes = {}   # hash -> {"filename": ..., "count": ...}

    total = collection.count()
    if total == 0:
        log.info("  Collection is empty -- nothing to clean.")
        return 0

    while offset < total:
        batch = collection.get(
            limit   = batch_size,
            offset  = offset,
            include = ["metadatas"],
        )
        for meta in batch["metadatas"]:
            fh = meta.get("file_hash", "")
            if fh:
                if fh not in stored_hashes:
                    stored_hashes[fh] = {
                        "filename": meta.get("filename", "?"),
                        "count":    0,
                    }
                stored_hashes[fh]["count"] += 1
        offset += batch_size

    if not stored_hashes:
        log.info("  No file_hash metadata found -- skipping cleanup.")
        log.info(
            "  (Old-schema chunks cannot be cleaned automatically.  "
            "Run: rm -rf ./db && python ingest.py)"
        )
        return 0

    orphan_hashes = set(stored_hashes.keys()) - live_hashes

    if not orphan_hashes:
        log.info("  No orphaned chunks found.  Database is clean.")
        return 0

    log.info("  Found %d orphaned file hash(es):", len(orphan_hashes))
    for oh in sorted(orphan_hashes):
        info = stored_hashes[oh]
        log.info(
            "    %s...  %s  (%d chunks)",
            oh[:12], info["filename"], info["count"],
        )

    total_removed = 0
    for oh in orphan_hashes:
        try:
            existing = collection.get(
                where   = {"file_hash": {"$eq": oh}},
                include = [],
            )
            if existing["ids"]:
                collection.delete(ids=existing["ids"])
                total_removed += len(existing["ids"])
                log.info(
                    "    Deleted %d chunks for hash %s...",
                    len(existing["ids"]), oh[:12],
                )
        except Exception as e:
            log.warning(
                "    Failed to delete chunks for hash %s...: %s",
                oh[:12], e,
            )

    # -- Clean progress file ------------------------------------------------
    before_count = len(progress["completed"])
    progress["completed"] = [
        h for h in progress["completed"] if h not in orphan_hashes
    ]
    removed_from_progress = before_count - len(progress["completed"])

    # Remove orphaned entries from path_index too
    path_index = progress.get("path_index", {})
    stale_paths = [p for p, h in path_index.items() if h in orphan_hashes]
    for p in stale_paths:
        del path_index[p]
    if stale_paths:
        log.info(
            "  Removed %d stale path_index entries.", len(stale_paths)
        )

    if "failed" in progress:
        old_failed = dict(progress["failed"])
        progress["failed"] = {
            p: err for p, err in old_failed.items()
            if os.path.exists(p)
        }
        removed_failed = len(old_failed) - len(progress["failed"])
        if removed_failed:
            log.info(
                "  Removed %d stale entries from failed list.",
                removed_failed,
            )

    if removed_from_progress:
        log.info(
            "  Removed %d stale entries from progress file.",
            removed_from_progress,
        )

    save_progress(progress)
    log.info(
        "  Cleanup complete: %d orphaned chunks removed.", total_removed
    )
    return total_removed


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_path_index(all_pdfs: list, completed_set: set,
                     path_index: dict, progress: dict):
    """
    First-run index builder.

    Hashes every unindexed PDF whose hash is already in completed_set and
    writes the abs_path -> hash mapping into path_index in-place.  Saves
    in batches of 100 so a mid-run restart doesn't lose all progress.

    Files whose hash is NOT in completed_set are left out -- they will be
    picked up by the main ingest loop as genuinely new files.

    This function is a no-op on all subsequent runs once every path has
    been indexed.
    """
    unindexed = [
        p for p in all_pdfs
        if os.path.abspath(p) not in path_index
    ]

    if not unindexed:
        log.info("Path index is current -- no hashing needed.")
        return

    log.info(
        "Building path index for %d unindexed files "
        "(hashing only, no embedding)...",
        len(unindexed),
    )

    dirty      = 0   # unsaved changes since last save
    save_every = 100

    for pdf_path in tqdm(unindexed, desc="Building path index"):
        abs_path = os.path.abspath(pdf_path)
        try:
            fhash = file_hash(pdf_path)
        except OSError as e:
            log.warning("Cannot hash %s: %s", pdf_path, e)
            continue

        # Only index files we have already successfully ingested.
        # New files are intentionally left out so the main loop handles them.
        if fhash in completed_set:
            path_index[abs_path] = fhash
            dirty += 1

        if dirty >= save_every:
            progress["path_index"] = path_index
            save_progress(progress)
            dirty = 0

    # -- final flush --------------------------------------------------------
    if dirty:
        progress["path_index"] = path_index
        save_progress(progress)

    indexed = sum(
        1 for p in all_pdfs
        if os.path.abspath(p) in path_index
    )
    log.info(
        "Path index complete: %d / %d files indexed.",
        indexed, len(all_pdfs),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # -- Argument parsing ---------------------------------------------------
    parser = argparse.ArgumentParser(description="Ingest academic PDFs into ChromaDB")
    parser.add_argument(
        "--dir",
        metavar="DIR",
        default=None,
        help="If specified, only ingest PDFs under this directory.",
    )
    args = parser.parse_args()

    filter_dir = os.path.abspath(args.dir) if args.dir else None

    if filter_dir and not os.path.isdir(filter_dir):
        log.error("--dir does not exist or is not a directory: %s", filter_dir)
        sys.exit(1)

    # -- ChromaDB -----------------------------------------------------------
    os.makedirs(config.DB_DIR, exist_ok=True)
    client = chromadb.PersistentClient(
        path     = config.DB_DIR,
        settings = Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name     = config.COLLECTION_NAME,
        metadata = {"hnsw:space": "cosine"},
    )
    log.info(
        "Collection '%s' has %d existing chunks",
        config.COLLECTION_NAME, collection.count(),
    )

    # -- Embedding model ----------------------------------------------------
    log.info("Loading embedding model: %s", config.EMBED_MODEL)
    embed_model = SentenceTransformer(config.EMBED_MODEL)

    # -- Collect files ------------------------------------------------------
    if filter_dir:
        all_pdfs = []
        for root, _, files in os.walk(filter_dir):
            for fn in files:
                if fn.lower().endswith(".pdf"):
                    all_pdfs.append(os.path.join(root, fn))
        all_pdfs = sorted(set(all_pdfs))
        log.info(
            "Scoped run: found %d PDF(s) under %s",
            len(all_pdfs), filter_dir,
        )
        if not all_pdfs:
            log.error("No PDFs found under --dir: %s", filter_dir)
            sys.exit(1)
    else:
        all_pdfs = collect_pdfs()
        if not all_pdfs:
            log.error("No PDFs found. Check PAPER_DIRS in config.py")
            sys.exit(1)

    # -- Load resume state --------------------------------------------------
    progress      = load_progress()
    completed_set = set(progress["completed"])      # set of hashes
    path_index    = progress.get("path_index", {})  # {abs_path -> hash}
    log.info(
        "Already completed: %d files  |  path index: %d entries",
        len(completed_set), len(path_index),
    )

    # -- Clean OneDrive logs before starting --------------------------------
    log.info("Cleaning OneDrive logs before ingest...")
    cleanup_onedrive_logs()

    # -- Report starting disk space -----------------------------------------
    log_drive_free_space("C")

    # -- Build path index ---------------------------------------------------
    build_path_index(all_pdfs, completed_set, path_index, progress)

    # -- Partition: fast-skip vs needs-check --------------------------------
    fast_skip   = []
    needs_check = []

    for pdf_path in all_pdfs:
        abs_path    = os.path.abspath(pdf_path)
        stored_hash = path_index.get(abs_path)
        if stored_hash and stored_hash in completed_set:
            fast_skip.append(pdf_path)
        else:
            needs_check.append(pdf_path)

    log.info(
        "Fast-skip: %d files  |  Need checking: %d files",
        len(fast_skip), len(needs_check),
    )

    # -- Bulk-free fast-skip OneDrive files ---------------------------------
    freed_mb = 0.0
    if fast_skip:
        log.info(
            "Releasing %d already-ingested OneDrive files (bulk)...",
            len(fast_skip),
        )
        bulk = free_onedrive_files_bulk(fast_skip)
        freed_mb += bulk["mb_reclaimed"]
        log.info(
            "  Bulk free: freed=%d  skipped=%d  reclaimed=%.1f MB",
            bulk["freed"], bulk["skipped"], bulk["mb_reclaimed"],
        )

    # -- Live hashes for orphan cleanup ------------------------------------
    live_hashes: set = {
        path_index[os.path.abspath(p)]
        for p in fast_skip
        if os.path.abspath(p) in path_index
    }

    # -- Stats -------------------------------------------------------------
    stats = {
        "ingested":       0,
        "skipped":        0,
        "metadata_fixed": 0,
        "failed":         0,
        "freed_mb":       freed_mb,
    }

    # -- Main ingest loop --------------------------------------------------
    for i, pdf_path in enumerate(tqdm(needs_check, desc="Ingesting papers")):

        abs_path = os.path.abspath(pdf_path)

        try:
            fhash = file_hash(pdf_path)
        except OSError as e:
            log.warning("Cannot read %s: %s", pdf_path, e)
            stats["failed"] += 1
            continue

        live_hashes.add(fhash)

        # -- Already ingested: check for rename/move -----------------------
        if fhash in completed_set:
            status = check_and_update_metadata(fhash, pdf_path, collection)

            if status == "match":
                path_index[abs_path] = fhash
                progress["path_index"] = path_index
                save_progress(progress)
                stats["skipped"] += 1
                free_onedrive_file(pdf_path)
                continue

            elif status == "error":
                stats["skipped"] += 1
                free_onedrive_file(pdf_path)
                continue

            elif status == "updated":
                path_index[abs_path] = fhash
                progress["path_index"] = path_index
                save_progress(progress)
                stats["metadata_fixed"] += 1
                free_onedrive_file(pdf_path)
                continue

            # status == "missing" -- fall through to full ingest

        # -- Skip known bad files ------------------------------------------
        if pdf_path in progress.get("failed", {}):
            log.debug("Previously failed, skipping: %s", Path(pdf_path).name)
            stats["skipped"] += 1
            free_onedrive_file(pdf_path)
            continue

        # -- Full ingest ---------------------------------------------------
        try:
            ingest_one(pdf_path, collection, embed_model, fhash)

            if fhash not in completed_set:
                progress["completed"].append(fhash)
                completed_set.add(fhash)

            path_index[abs_path] = fhash
            progress["path_index"] = path_index
            save_progress(progress)
            stats["ingested"] += 1

            mb = get_file_size_mb(pdf_path)
            if free_onedrive_file(pdf_path):
                stats["freed_mb"] += mb

        except Exception as e:
            log.error("Error processing %s: %s", Path(pdf_path).name, e)
            progress.setdefault("failed", {})[pdf_path] = str(e)
            save_progress(progress)
            stats["failed"] += 1
            free_onedrive_file(pdf_path)

        # -- Periodic disk space report + log cleanup ----------------------
        if i > 0 and i % 50 == 0:
            log_drive_free_space("C")
            cleanup_onedrive_logs()

    # -- Orphan cleanup (skipped for scoped runs) --------------------------
    if filter_dir:
        log.info(
            "Skipping orphan cleanup -- scoped run via --dir does not "
            "have visibility into all files."
        )
        orphans_removed = 0
    else:
        orphans_removed = cleanup_orphans(collection, live_hashes, progress)

    # -- Final OneDrive log cleanup ----------------------------------------
    log.info("Final OneDrive log cleanup...")
    cleanup_onedrive_logs()

    # -- Final report ------------------------------------------------------
    log_drive_free_space("C")
    log.info(
        "Done.  ingested=%d  metadata_fixed=%d  skipped=%d  "
        "failed=%d  orphans_removed=%d  freed=%.1f MB",
        stats["ingested"], stats["metadata_fixed"],
        stats["skipped"], stats["failed"],
        orphans_removed, stats["freed_mb"],
    )
    log.info("Total chunks in collection: %d", collection.count())


if __name__ == "__main__":
    main()
