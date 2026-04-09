# ocr_scanned.py
import os
import json
import hashlib
import fitz
import ocrmypdf
import config

def get_file_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

scanned_file = os.path.join(config.DB_DIR, "scanned.json")

if not os.path.exists(scanned_file):
    print("No scanned.json found -- run ingest.py first to detect scanned PDFs.")
    exit()

with open(scanned_file) as f:
    scanned_hashes = set(json.load(f))

print("Found {} scanned PDFs to process.".format(len(scanned_hashes)))

# Build a map of hash -> filepath so we can find the files
hash_to_path = {}
for base in config.PAPER_DIRS:
    for root, _, files in os.walk(base):
        for fname in files:
            if fname.lower().endswith(".pdf"):
                path = os.path.join(root, fname)
                h    = get_file_hash(path)
                if h in scanned_hashes:
                    hash_to_path[h] = path

print("Located {}/{} files on disk.".format(
    len(hash_to_path), len(scanned_hashes)))

success        = 0
failed         = 0
hashes_to_remove = set()

for old_hash, path in hash_to_path.items():
    try:
        print("  OCR: {}".format(os.path.basename(path)))
        ocrmypdf.ocr(
            path, path,
            skip_text      = True,
            deskew         = True,
            language       = "eng",
            progress_bar   = False,
        )
        # File content changed after OCR -- old hash is now stale
        hashes_to_remove.add(old_hash)
        success += 1
    except Exception as e:
        print("  FAILED {}: {}".format(os.path.basename(path), e))
        failed += 1

# Remove successfully OCR'd hashes from scanned.json
# ingest.py will encounter them as new files and index them normally
remaining = scanned_hashes - hashes_to_remove
with open(scanned_file, "w") as f:
    json.dump(list(remaining), f)

print("\nOCR complete : {} succeeded, {} failed".format(success, failed))
print("Removed {} hashes from scanned.json".format(len(hashes_to_remove)))
print("Run ingest.py to index the OCR'd files.")
