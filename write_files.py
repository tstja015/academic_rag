#!/usr/bin/env python3
"""
write_files.py -- Paste Claude's output and extract files.

Handles plain-text paste where markdown fencing (```) has been stripped.
Detects files by header lines like:

    config.py
    # config.py

followed by code content, ending when the next file header appears
or a non-code section (like "Summary of all changes") begins.
"""

import os
import re
import sys


# Extensions we recognize as files
EXTENSIONS = {
    "py", "json", "txt", "yaml", "yml", "toml", "cfg", "ini",
    "sh", "bash", "md", "csv", "tsv",
}


def is_filename_line(line: str) -> str | None:
    """
    If the line looks like a standalone filename, return it.
    Otherwise return None.

    Matches:
        config.py
        ## `config.py`
        ## 1. `config.py` — Description
        **`config.py`**
        ## config.py
    """
    stripped = line.strip()
    if not stripped:
        return None

    # Skip lines that are clearly code or prose
    if stripped.startswith(("import ", "from ", "def ", "class ", "    ", "\t")):
        return None
    if stripped.startswith(("- ", "* ", "| ")):
        return None

    # Try to extract a filename
    # Pattern 1: ## `filename.ext` or **`filename.ext`** with optional numbering
    m = re.match(
        r'^(?:#{1,4}\s+)?(?:\d+\.\s*)?[`*]*([a-zA-Z0-9_][a-zA-Z0-9_./-]*\.([a-zA-Z0-9]+))[`*]*(?:\s.*)?$',
        stripped,
    )
    if m and m.group(2) in EXTENSIONS:
        return m.group(1)

    # Pattern 2: bare filename on its own line (no other words)
    m = re.match(
        r'^([a-zA-Z0-9_][a-zA-Z0-9_./-]*\.([a-zA-Z0-9]+))\s*$',
        stripped,
    )
    if m and m.group(2) in EXTENSIONS:
        return m.group(1)

    return None


def is_code_line(line: str) -> bool:
    """Heuristic: does this line look like Python/JSON code?"""
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith("#"):
        return True
    if stripped.startswith(("import ", "from ", "def ", "class ", "return ",
                           "if ", "elif ", "else:", "for ", "while ",
                           "try:", "except ", "finally:", "with ",
                           "raise ", "yield ", "async ", "await ",
                           "print(", "self.", "@", "global ",
                           "logging.", "log.", "os.", "sys.", "re.",
                           '"', "'", "{", "}", "[", "]", "(",
                           )):
        return True
    if line.startswith(("    ", "\t")):
        return True
    if re.match(r'^[A-Za-z_][A-Za-z0-9_.]*\s*[=$$]', stripped):
        return True
    if stripped.startswith("@"):
        return True
    if stripped in (")", "]", "},", ")", "],"):
        return True
    # FIX: use double-quoted string to allow single quotes in pattern
    if re.match(r"^\s*([#]|\w+\s*=|\w+$)", stripped) or stripped.startswith(('"', "'")):
        return True
    # FIX: treat lines with .replace( or similar method calls as code
    if re.search(r'\.\w+$', stripped):
        return True
    return False


def is_section_header(line: str) -> bool:
    """Detect non-code section headers that should stop file extraction."""
    stripped = line.strip()
    # "Summary of all changes", "### Priority order", etc.
    if re.match(r'^#{1,4}\s+\w', stripped):
        # But not if it contains a filename
        if is_filename_line(stripped):
            return False
        return True
    # Table-like markdown
    if stripped.startswith("|") and "---" in stripped:
        return True
    return False


def extract_files(text: str) -> list:
    """Parse pasted text and return list of (filename, content)."""
    lines = text.split("\n")
    files = []

    i = 0
    while i < len(lines):
        fname = is_filename_line(lines[i])
        if fname:
            # Found a file header -- collect code lines
            code_lines = []
            i += 1

            # Skip the first line if it's just "# filename" repeating the header
            if i < len(lines):
                first_stripped = lines[i].strip()
                if first_stripped == "# " + fname or first_stripped == "// " + fname:
                    code_lines.append(lines[i])
                    i += 1

            # Collect code until next file header, section header, or end
            while i < len(lines):
                # Check if this line is a new file header
                next_fname = is_filename_line(lines[i])
                if next_fname:
                    break

                # Check if this is a non-code section
                if is_section_header(lines[i]) and not is_code_line(lines[i]):
                    # Look ahead: if next few lines are not code, stop
                    non_code_count = 0
                    for j in range(i, min(i + 5, len(lines))):
                        if not is_code_line(lines[j]):
                            non_code_count += 1
                    if non_code_count >= 3:
                        break

                code_lines.append(lines[i])
                i += 1

            # Trim trailing blank lines
            while code_lines and not code_lines[-1].strip():
                code_lines.pop()

            # Trim leading blank lines
            while code_lines and not code_lines[0].strip():
                code_lines.pop(0)

            if code_lines:
                content = "\n".join(code_lines) + "\n"
                files.append((fname, content))
        else:
            i += 1

    return files


def main():
    print("=" * 60)
    print("  File Extractor (plain-text mode)")
    print("=" * 60)
    print()
    print("Paste the full response below, then press:")
    print("  Ctrl+D  (Linux/Mac)")
    print("  Ctrl+Z then Enter  (Windows)")
    print()
    print("-" * 60)

    try:
        text = sys.stdin.read()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(0)

    if not text.strip():
        print("No input received.")
        sys.exit(1)

    files = extract_files(text)

    if not files:
        print("\nNo files found in the pasted text.")
        print()
        print("Debug info:")
        lines = text.split("\n")
        print("  Total lines pasted: {}".format(len(lines)))

        # Show lines that almost matched as filenames
        near_misses = []
        for line in lines[:100]:
            stripped = line.strip()
            if re.search(r'\.\w{1,4}$', stripped) and len(stripped) < 80:
                near_misses.append(stripped)
        if near_misses:
            print("  Lines that look like they might be filenames:")
            for nm in near_misses[:10]:
                print("    '{}'".format(nm))
        else:
            print("  No filename-like lines found in first 100 lines.")

        print()
        print("Expected format:")
        print("  config.py")
        print("  # config.py")
        print("  import os")
        print("  ...")
        sys.exit(1)

    # Show what we found
    print("\n" + "=" * 60)
    print("Found {} file(s):".format(len(files)))
    print("=" * 60)
    for fname, content in files:
        lines_count = content.count("\n")
        chars = len(content)
        # Show first non-comment, non-blank line as preview
        preview = ""
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                preview = stripped[:60]
                break
        print("  {:<40} {:>5} lines  {:>7,} chars".format(
            fname, lines_count, chars))
        if preview:
            print("  {}  {}{}".format(
                " " * 40, preview,
                "..." if len(preview) >= 60 else ""))

    # Check for overwrites
    existing = [fname for fname, _ in files if os.path.exists(fname)]
    if existing:
        print("\n  WARNING: these files already exist and will be OVERWRITTEN:")
        for f in existing:
            print("    {}".format(f))

    # Backup option
    if existing:
        print()
        try:
            backup = input("Create .bak backups of existing files? [Y/n] ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled.")
            sys.exit(0)
        if backup not in ("n", "no"):
            for f in existing:
                bak = f + ".bak"
                try:
                    with open(f, "r") as src, open(bak, "w") as dst:
                        dst.write(src.read())
                    print("  Backed up: {} -> {}".format(f, bak))
                except Exception as e:
                    print("  Could not backup {}: {}".format(f, e))

    # Confirm
    print()
    try:
        confirm = input("Write these files? [y/N] ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        sys.exit(0)

    if confirm not in ("y", "yes"):
        print("Aborted.")
        sys.exit(0)

    # Write files
    print()
    for fname, content in files:
        dirpath = os.path.dirname(fname)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        with open(fname, "w", encoding="utf-8") as f:
            f.write(content)

        print("  Wrote: {}".format(fname))

    print("\nDone. {} file(s) written.".format(len(files)))


if __name__ == "__main__":
    main()
