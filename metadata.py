# metadata.py
"""
Paper metadata manager.

Fetches and caches metadata (authors, DOI, title, journal, year, etc.)
using CrossRef API via DOI extraction from PDFs.
"""

import os
import re
import json
import logging
import hashlib
from pathlib import Path

import fitz  # PyMuPDF

import config
from onedrive_utils import free_onedrive_file

log = logging.getLogger(__name__)

METADATA_FILE = os.path.join(config.DB_DIR, "paper_metadata.json")


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------

def load_metadata() -> dict:
    """
    Load metadata cache from disk.

    Schema:
    {
        "<file_hash>": {
            "filename": "paper.pdf",
            "full_path": "/abs/path/paper.pdf",
            "folder": "project_name",
            "title": "Paper Title",
            "authors": ["First Last", "First Last", ...],
            "year": 2023,
            "journal": "Journal Name",
            "doi": "10.1234/example",
            "volume": "12",
            "issue": "3",
            "pages": "100-110",
            "publisher": "Publisher Name",
            "abstract": "...",
            "fetch_status": "success" | "no_doi" | "crossref_failed" | "pending",
        },
        ...
    }
    """
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE) as f:
            return json.load(f)
    return {}


def save_metadata(metadata: dict):
    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# DOI extraction from PDF text
# ---------------------------------------------------------------------------

_DOI_PATTERNS = [
    # Standard DOI patterns
    re.compile(r'(?:doi[:\s]*|https?://(?:dx\.)?doi\.org/)(10\.\d{4,9}/[^\s,;}$$]+)', re.IGNORECASE),
    # Bare DOI on its own
    re.compile(r'\b(10\.\d{4,9}/[^\s,;}$$]+)'),
]


def extract_doi(pdf_path: str) -> str | None:
    """Extract DOI from the first few pages of a PDF."""
    try:
        doc = fitz.open(pdf_path)
        # Check first 3 pages (DOI usually on first page or in header/footer)
        text = ""
        for i in range(min(3, len(doc))):
            text += doc[i].get_text("text") + "\n"
        doc.close()

        for pattern in _DOI_PATTERNS:
            match = pattern.search(text)
            if match:
                doi = match.group(1).rstrip(".")
                # Basic validation
                if len(doi) > 10 and "/" in doi:
                    return doi

    except Exception as e:
        log.debug("DOI extraction failed for %s: %s", pdf_path, e)

    return None


# ---------------------------------------------------------------------------
# CrossRef lookup
# ---------------------------------------------------------------------------

def fetch_crossref_metadata(doi: str) -> dict | None:
    """
    Query CrossRef API for paper metadata.

    Returns dict with normalized fields, or None on failure.
    """
    try:
        import requests
    except ImportError:
        log.warning("'requests' not installed -- cannot fetch CrossRef metadata")
        return None

    url = "https://api.crossref.org/works/{}".format(doi)
    headers = {
        "User-Agent": "AcademicRAG/1.0 (mailto:research@example.com)",
    }

    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            log.debug("CrossRef returned %d for DOI %s", r.status_code, doi)
            return None

        data = r.json().get("message", {})

        # Extract authors
        authors = []
        for author in data.get("author", []):
            given = author.get("given", "")
            family = author.get("family", "")
            if given and family:
                authors.append("{} {}".format(given, family))
            elif family:
                authors.append(family)

        # Extract title
        title_list = data.get("title", [])
        title = title_list[0] if title_list else ""

        # Extract journal
        container = data.get("container-title", [])
        journal = container[0] if container else ""

        # Extract year
        year = None
        date_parts = data.get("published-print", {}).get("date-parts", [[]])
        if not date_parts or not date_parts[0]:
            date_parts = data.get("published-online", {}).get("date-parts", [[]])
        if date_parts and date_parts[0]:
            year = date_parts[0][0]

        # Extract volume, issue, pages
        volume = data.get("volume", "")
        issue = data.get("issue", "")
        pages = data.get("page", "")
        publisher = data.get("publisher", "")

        # Extract abstract (if available)
        abstract = data.get("abstract", "")
        # Clean HTML tags from abstract
        if abstract:
            abstract = re.sub(r'<[^>]+>', '', abstract).strip()

        return {
            "title": title,
            "authors": authors,
            "year": year,
            "journal": journal,
            "volume": volume,
            "issue": issue,
            "pages": pages,
            "publisher": publisher,
            "doi": doi,
            "abstract": abstract[:500] if abstract else "",
        }

    except Exception as e:
        log.debug("CrossRef lookup failed for %s: %s", doi, e)
        return None


# ---------------------------------------------------------------------------
# Single-file metadata fetch
# ---------------------------------------------------------------------------

def fetch_metadata_for_file(pdf_path: str, file_hash: str,
                            force: bool = False) -> dict:
    """
    Get or fetch metadata for a single PDF.

    Returns the metadata dict for this file.
    """
    metadata = load_metadata()

    # Already cached and not forcing refresh
    if file_hash in metadata and not force:
        existing = metadata[file_hash]
        if existing.get("fetch_status") in ("success", "no_doi"):
            return existing

    filename = Path(pdf_path).name
    folder = Path(pdf_path).parent.name

    entry = {
        "filename": filename,
        "full_path": os.path.abspath(pdf_path),
        "folder": folder,
        "title": "",
        "authors": [],
        "year": None,
        "journal": "",
        "doi": "",
        "volume": "",
        "issue": "",
        "pages": "",
        "publisher": "",
        "abstract": "",
        "fetch_status": "pending",
    }

    # Try to extract DOI
    doi = extract_doi(pdf_path)

    if not doi:
        entry["fetch_status"] = "no_doi"
        # Use filename as fallback title
        entry["title"] = filename.replace(".pdf", "").replace("_", " ")
        metadata[file_hash] = entry
        save_metadata(metadata)
        return entry

    entry["doi"] = doi

    # Fetch from CrossRef
    crossref_data = fetch_crossref_metadata(doi)

    if crossref_data:
        entry.update(crossref_data)
        entry["fetch_status"] = "success"
    else:
        entry["fetch_status"] = "crossref_failed"
        entry["title"] = filename.replace(".pdf", "").replace("_", " ")

    metadata[file_hash] = entry
    save_metadata(metadata)
    return entry


# ---------------------------------------------------------------------------
# Bulk metadata fetch (called standalone or from query.py fetchmeta command)
# ---------------------------------------------------------------------------

def fetch_all_metadata(progress: dict = None, force: bool = False):
    """
    Fetch metadata for all ingested papers.

    Uses the ingest progress file to find file hashes and paths.
    Frees OneDrive files after reading each PDF (standalone-safe).
    """
    from ingest import load_progress

    if progress is None:
        progress = load_progress()

    metadata = load_metadata()
    path_index = progress.get("path_index", {})
    completed = set(progress.get("completed", []))

    # Build hash -> path mapping
    hash_to_path = {}
    for abs_path, fhash in path_index.items():
        if fhash in completed:
            hash_to_path[fhash] = abs_path

    total = len(hash_to_path)
    fetched = 0
    skipped = 0
    no_doi = 0
    failed = 0

    log.info("Fetching metadata for %d papers...", total)

    from tqdm import tqdm
    for fhash, abs_path in tqdm(hash_to_path.items(), desc="Fetching metadata"):

        # Skip if already fetched (unless forcing)
        if not force and fhash in metadata:
            status = metadata[fhash].get("fetch_status", "")
            if status in ("success", "no_doi"):
                skipped += 1
                continue

        if not os.path.exists(abs_path):
            log.debug("File not found: %s", abs_path)
            skipped += 1
            continue

        entry = fetch_metadata_for_file(abs_path, fhash, force=force)

        # Free OneDrive file after reading (standalone mode safety)
        free_onedrive_file(abs_path)

        if entry["fetch_status"] == "success":
            fetched += 1
        elif entry["fetch_status"] == "no_doi":
            no_doi += 1
        else:
            failed += 1

    log.info(
        "Metadata fetch complete: fetched=%d  no_doi=%d  failed=%d  skipped=%d",
        fetched, no_doi, failed, skipped,
    )


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def search_by_author(name: str) -> list:
    """
    Search metadata for papers by a given author name.

    Returns list of metadata dicts matching the author (case-insensitive,
    partial match).
    """
    metadata = load_metadata()
    name_lower = name.lower()
    results = []

    for fhash, entry in metadata.items():
        authors = entry.get("authors", [])
        for author in authors:
            if name_lower in author.lower():
                results.append(entry)
                break

    return results


def search_by_filename(pattern: str) -> list:
    """
    Search metadata by filename (substring or exact match).

    Returns list of metadata dicts.
    """
    import fnmatch
    metadata = load_metadata()
    pattern_lower = pattern.lower()
    results = []

    for fhash, entry in metadata.items():
        fname = entry.get("filename", "")
        if "*" in pattern or "?" in pattern:
            if fnmatch.fnmatch(fname.lower(), pattern_lower):
                results.append(entry)
        else:
            if pattern_lower in fname.lower():
                results.append(entry)

    return results


def get_all_authors() -> dict:
    """
    Return a dict of {author_name: [list of filenames]}.
    """
    metadata = load_metadata()
    author_map = {}

    for fhash, entry in metadata.items():
        fname = entry.get("filename", "")
        for author in entry.get("authors", []):
            if author not in author_map:
                author_map[author] = []
            author_map[author].append(fname)

    return author_map
