# Download & extract the first OpenVid zip to /path/to/dataset
# Requires: pip install huggingface_hub

import os
import re
import shutil
from typing import List
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
import zipfile

REPO_ID = "nkp37/OpenVid-1M"
REPO_TYPE = "dataset"
DEST_DIR = "/tmp/videos"
CACHE_DIR = "/tmp/hf_cache"


# --- helpers ---------------------------------------------------------------

ZIP_FULL_RE = re.compile(r"^OpenVid_part(\d+)\.zip$")
ZIP_PART_RE = re.compile(r"^OpenVid_part(\d+)_part[\w\d]+$")  # handles aa/ab or 000/001 styles

def natural_int(s: str) -> int:
    try:
        return int(s)
    except Exception:
        return 10**9
    
def choose_first_zip(files, idx) -> str:
    path = f"OpenVid_part{idx}.zip"
    if path not in files:
        raise RuntimeError(f"Could not find {path} in repo files.")
    return path
'''
def choose_first_zip(files: List[str], min_idx) -> str:
    """
    Return the 'base' zip name for the smallest part index present,
    e.g. 'OpenVid_part0.zip'. Works whether we have full zip or split parts.
    """
    candidates = []
    for f in files:
        m = ZIP_FULL_RE.match(f)
        if m:
            counter = natural_int(m.group(1))
            if counter < min_idx:
                continue
            candidates.append(("full", counter, f))
            continue
        m = ZIP_PART_RE.match(f)
        if m:
            # store base name that we'd reconstruct to
            base = f"OpenVid_part{m.group(1)}.zip"
            candidates.append(("split", natural_int(m.group(1)), base))
    if not candidates:
        raise RuntimeError("No OpenVid_part*.zip or split parts found in repo.")
    # choose minimal index; prefer 'full' over 'split' if both exist for same index
    candidates.sort(key=lambda x: (x[1], 0 if x[0] == "full" else 1))
    return candidates[0][2]
'''
def download_or_reconstruct_zip(base_zip: str, files: List[str]) -> str:
    """
    If base_zip exists, download it. Otherwise, download all pieces matching the split pattern
    and concatenate into a full zip in TMP_DIR. Returns local path to the full zip.
    """
    # Case A: full zip exists
    if base_zip in files:
        return hf_hub_download(repo_id=REPO_ID, filename=base_zip, repo_type=REPO_TYPE, cache_dir=CACHE_DIR)

    # Case B: reconstruct from parts
    prefix = base_zip.replace(".zip", "")
    parts = sorted([f for f in files if f.startswith(prefix + "_part")])
    if not parts:
        raise FileNotFoundError(f"Could not find {base_zip} or any {prefix}_part* split files.")

    local_parts = []
    for p in parts:
        local_parts.append(hf_hub_download(repo_id=REPO_ID, filename=p, repo_type=REPO_TYPE, cache_dir=CACHE_DIR))

    out_zip = os.path.join(TMP_DIR, base_zip)
    with open(out_zip, "wb") as out:
        for lp in local_parts:
            with open(lp, "rb") as inp:
                shutil.copyfileobj(inp, out, length=1024 * 1024)  # 1MB chunks
    return out_zip

VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".avi", ".mov", ".m4v"}

def extract_videos(zip_path: str, out_dir: str) -> int:
    count = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            ext = os.path.splitext(name)[1].lower()
            if ext in VIDEO_EXTS:
                # flatten like `unzip -j`
                target = os.path.join(out_dir, os.path.basename(name))
                # avoid overwriting by adding a suffix if needed
                base, ext2 = os.path.splitext(target)
                i = 1
                final = target
                while os.path.exists(final):
                    final = f"{base}__{i}{ext2}"
                    i += 1
                with zf.open(info, "r") as src, open(final, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                count += 1
    return count

# --- main ------------------------------------------------------------------

api = HfApi()
files = list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
for i in range(24, 50):
    VIDEOS_OUT = os.path.join(DEST_DIR, f"videos{i}")
    TMP_DIR = os.path.join(DEST_DIR, f"_tmp_openvid{i}")
    os.makedirs(VIDEOS_OUT, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    first_base_zip = choose_first_zip(files, i)  # e.g., "OpenVid_part0.zip"
    print("Chosen first zip:", first_base_zip)

    zip_path = download_or_reconstruct_zip(first_base_zip, files)
    print("Zip ready at:", zip_path)

    n = extract_videos(zip_path, VIDEOS_OUT)
    print(f"Extracted {n} video files to: {VIDEOS_OUT}")

    # Optional cleanup of reconstructed zip
    if os.path.dirname(zip_path) == TMP_DIR:
        try:
            os.remove(zip_path)
        except Exception:
            pass

    print("Done.")
