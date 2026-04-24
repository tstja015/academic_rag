# onedrive_utils.py
import os, sys, logging, subprocess, shutil
from pathlib import Path, PurePosixPath
import config

log = logging.getLogger(__name__)

def _is_wsl():
    try:
        with open("/proc/version") as f:
            c = f.read().lower()
            return "microsoft" in c or "wsl" in c
    except OSError:
        return False

IS_WINDOWS = sys.platform == "win32"
IS_WSL     = _is_wsl()
CAN_FREE   = IS_WINDOWS or IS_WSL

def wsl_to_windows_path(path):
    p = PurePosixPath(path)
    parts = p.parts
    if len(parts) < 3 or parts[1] != "mnt":
        raise ValueError(f"Not under /mnt/: {path}")
    drive = parts[2].upper()
    rest  = chr(92).join(parts[3:])
    return drive + ":" + chr(92) + rest

def is_onedrive_path(path):
    return config.ONEDRIVE_PATH_HINT.lower() in path.lower()

def resolve_to_windows_path(path):
    try:
        return wsl_to_windows_path(path), "direct"
    except ValueError:
        pass
    try:
        real = os.path.realpath(path)
        return wsl_to_windows_path(real), "symlink"
    except ValueError:
        pass
    try:
        r = subprocess.run(["wslpath", "-w", path],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip(), "wslpath"
    except Exception:
        pass
    raise ValueError(f"Could not resolve Windows path for: {path}")

def is_local_copy(path):
    if not IS_WSL:
        return True
    try:
        win_path, _ = resolve_to_windows_path(path)
        r = subprocess.run(["fsutil.exe", "reparsepoint", "query", win_path],
                           capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return True
        return "cloud" not in r.stdout.lower()
    except Exception as e:
        log.debug("is_local_copy failed for %s: %s", path, e)
        return True

def get_file_size_mb(path):
    try:
        return os.path.getsize(path) / 1_048_576
    except OSError:
        return 0.0

def cleanup_onedrive_logs():
    """
    Delete accumulated OneDrive .odl log files that can grow to hundreds of GB.
    Safe to call at any time -- OneDrive recreates logs as needed.
    Uses PowerShell natively from WSL to avoid slow /mnt/c rglob.
    """
    if not getattr(config, "ONEDRIVE_CLEANUP_LOGS", True):
        return
    if not IS_WSL and not IS_WINDOWS:
        return

    try:
        if IS_WSL:
            r = subprocess.run(
                ["powershell.exe", "-Command",
                 "Remove-Item '$env:LOCALAPPDATA\\Microsoft\\OneDrive\\logs\\*.odl' "
                 "-Recurse -Force -ErrorAction SilentlyContinue"],
                capture_output=True, text=True, timeout=30,
            )
        else:
            # Native Windows
            log_dir = os.path.join(os.environ.get("LOCALAPPDATA", ""),
                                   "Microsoft", "OneDrive", "logs")
            r = subprocess.run(
                ["powershell.exe", "-Command",
                 f"Remove-Item '{log_dir}\\*.odl' -Recurse -Force "
                 "-ErrorAction SilentlyContinue"],
                capture_output=True, text=True, timeout=30,
            )

        if r.returncode == 0:
            log.info("OneDrive log cleanup: completed via PowerShell")
        else:
            log.debug("OneDrive log cleanup: powershell returned %d", r.returncode)

    except subprocess.TimeoutExpired:
        log.warning("OneDrive log cleanup: timed out after 30s")
    except Exception as e:
        log.debug("OneDrive log cleanup failed: %s", e)

def free_onedrive_file(path):
    if not getattr(config, "ONEDRIVE_FREE_AFTER_INGEST", False):
        return False
    if not CAN_FREE:
        return False
    free_all = getattr(config, "ONEDRIVE_FREE_ALL", False)
    if not free_all and not is_onedrive_path(path):
        log.debug("Skipping non-OneDrive path: %s", path)
        return False
    size_mb = get_file_size_mb(path)
    try:
        win_path, method = resolve_to_windows_path(path)
        log.debug("Resolved %s via %s", path, method)
    except ValueError as e:
        log.warning("Path conversion failed: %s", e)
        return False
    attrib_exe = "attrib.exe" if IS_WSL else "attrib"
    if shutil.which(attrib_exe) is None:
        log.warning("%s not found - WSL interop may be disabled.", attrib_exe)
        return False
    try:
        r = subprocess.run([attrib_exe, "+U", "-P", win_path],
                           capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            log.warning("attrib.exe failed for %s: %s",
                        Path(path).name, r.stdout.strip())
            return False
        log.info("Freed OneDrive file: %s  (%.1f MB reclaimed)",
                 Path(path).name, size_mb)
        return True
    except subprocess.TimeoutExpired:
        log.error("attrib.exe timed out for %s", path)
        return False
    except Exception as e:
        log.error("Unexpected error freeing %s: %s", path, e)
        return False

def free_onedrive_files_bulk(paths):
    summary = {"freed": 0, "skipped": 0, "failed": 0, "mb_reclaimed": 0.0}
    for path in paths:
        size_mb = get_file_size_mb(path)
        ok = free_onedrive_file(path)
        if ok:
            summary["freed"] += 1
            summary["mb_reclaimed"] += size_mb
        else:
            summary["skipped"] += 1
    return summary

def log_drive_free_space(drive="C"):
    try:
        mount = "/mnt/" + drive.lower() if IS_WSL else drive + ":" + chr(92)
        total, used, free = shutil.disk_usage(mount)
        log.info("Drive %s:  free: %.1f GB / %.1f GB total",
                 drive.upper(), free/1_073_741_824, total/1_073_741_824)
    except Exception as e:
        log.warning("Could not read disk usage for %s: %s", drive, e)
