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