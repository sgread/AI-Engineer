"""Download and unzip a MovieLens release into data/raw/<name>/."""

from __future__ import annotations

import zipfile
from pathlib import Path

import httpx
from tqdm import tqdm

from .config import DATASETS, RAW, ensure_dirs


def dataset_dir(name: str) -> Path:
    return RAW / name


def download(name: str) -> Path:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Choose from {list(DATASETS)}")
    ensure_dirs()
    target = dataset_dir(name)
    if (target / "ratings.csv").exists():
        print(f"{name}: already present at {target}")
        return target

    zip_path = RAW / f"{name}.zip"
    if not zip_path.exists():
        url = DATASETS[name]
        tmp = zip_path.with_suffix(".part")
        with httpx.stream("GET", url, follow_redirects=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=name) as bar:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))
        tmp.rename(zip_path)

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW)  # the zip contains a top-level <name>/ folder
    print(f"{name}: extracted to {target}")
    return target
