#!/usr/bin/env python3
"""Unduh dataset kanopi sawit Roboflow DI DALAM VM Colab lewat REST API (zip),
gabungkan semua kelas jadi satu, lalu latih YOLO.

Dipakai lewat `colab exec -f`. Kunci dibaca dari /content/rf_key.txt
(atau env ROBOFLOW_API_KEY) dan dihapus di akhir. Disk lokal tidak disentuh.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

CONTENT = Path("/content")
RAW = CONTENT / "rf_raw"
CLEAN = CONTENT / "rf_palm"
CLASS_NAME = "oil_palm"
KEY_FILE = CONTENT / "rf_key.txt"


def key() -> str:
    if KEY_FILE.exists():
        k = KEY_FILE.read_text().strip()
        if k:
            return k
    k = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not k:
        sys.exit("API key tidak ada: /content/rf_key.txt atau env ROBOFLOW_API_KEY")
    return k


def api(url: str, method: str = "GET") -> dict:
    req = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def sh(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stdout.strip():
        print(r.stdout[-1200:])
    if r.returncode != 0:
        print("GAGAL:", " ".join(cmd), file=sys.stderr)
        print(r.stderr[-1200:], file=sys.stderr)
        sys.exit(r.returncode)


def fetch_zip(ws: str, proj: str, ver: int) -> Path:
    k = key()
    base = f"https://api.roboflow.com/dataset/{ws}/{proj}/{ver}/yolov8"
    info = api(f"{base}?api_key={k}")
    print("--- versi", ver, "| info kunci:", sorted(info.keys()))
    link = (info.get("export") or {}).get("link")
    if not link:
        print("--- belum ada ekspor yolov8, minta dibuat ---")
        try:
            api(f"{base}?api_key={k}", method="POST")
        except Exception as e:
            print("POST ekspor:", repr(e)[:200])
        for i in range(30):
            time.sleep(6)
            info = api(f"{base}?api_key={k}")
            link = (info.get("export") or {}).get("link")
            if link:
                print(f"--- ekspor siap setelah ~{6*(i+1)}s")
                break
    if not link:
        sys.exit("ekspor tidak siap: " + json.dumps(info)[:400])
    z = CONTENT / "rf.zip"
    print("--- unduh zip ---")
    urllib.request.urlretrieve(link, z)
    print("zip:", round(z.stat().st_size / 1e6, 1), "MB")
    return z


def extract(z: Path) -> Path:
    if RAW.exists():
        shutil.rmtree(RAW)
    RAW.mkdir(parents=True)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(RAW)
    n = sum(1 for p in RAW.rglob("*") if p.is_file())
    print(f"--- ekstrak: {n} file ke {RAW}")
    return RAW


def find_yaml(root: Path) -> Path:
    cands = sorted(root.rglob("data.yaml"), key=lambda p: len(p.parts))
    if not cands:
        for p in sorted(root.rglob("*.yaml"), key=lambda p: len(p.parts)):
            print("  yaml lain:", p)
        sys.exit(f"data.yaml tidak ditemukan di {root}")
    print("--- data.yaml:", cands[0])
    return cands[0]


def remap(src_yaml: Path) -> Path:
    import yaml

    meta = yaml.safe_load(src_yaml.read_text()) or {}
    names = meta.get("names") or {}
    print(f"--- kelas asli ({len(names)}): {names}")

    if CLEAN.exists():
        shutil.rmtree(CLEAN)
    stats = {}
    for split in ("train", "valid", "test"):
        src_img = src_yaml.parent / split / "images"
        src_lbl = src_yaml.parent / split / "labels"
        if not src_img.exists():
            print(f"  lewati '{split}'")
            continue
        dst_img = CLEAN / split / "images"
        dst_lbl = CLEAN / split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        n_img = n_box = 0
        for img in src_img.iterdir():
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            shutil.copy2(img, dst_img / img.name)
            n_img += 1
            lab = src_lbl / (img.stem + ".txt")
            out = dst_lbl / (img.stem + ".txt")
            if lab.exists():
                lines = []
                for line in lab.read_text().splitlines():
                    parts = line.split()
                    if len(parts) >= 5:
                        parts[0] = "0"
                        lines.append(" ".join(parts))
                        n_box += 1
                out.write_text("\n".join(lines) + ("\n" if lines else ""))
            else:
                out.write_text("")
        stats[split] = (n_img, n_box)
        print(f"--- {split}: {n_img} gambar, {n_box} kotak")

    out_yaml = CLEAN / "data.yaml"
    out_yaml.write_text(
        f"path: {CLEAN}\ntrain: train/images\nval: valid/images\n"
        f"test: test/images\nnames:\n  0: {CLASS_NAME}\n"
    )
    total = sum(v[0] for v in stats.values())
    print(f"--- total: {total} gambar di {CLEAN}")
    return out_yaml


def train(data_yaml: Path, epochs: int) -> None:
    import torch
    from ultralytics import YOLO

    print("--- GPU:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    YOLO("yolov8n.pt").train(
        data=str(data_yaml), epochs=epochs, imgsz=640, batch=16, device=0,
        workers=4, patience=15, lr0=0.01, project="/content/runs",
        name="roboflow_palm", exist_ok=True,
    )
    best = Path("/content/runs/roboflow_palm/weights/best.pt")
    print("SELESAI | best.pt:", best.exists(),
          best.stat().st_size if best.exists() else "-")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default="mahir-sehmi-fgblg")
    ap.add_argument("--project", default="oil-palm-tree-crown-detection-from-aerial-image")
    ap.add_argument("--version", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=50)
    args, _unknown = ap.parse_known_args()

    try:
        sh([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "pyyaml"])
        z = fetch_zip(args.workspace, args.project, args.version)
        root = extract(z)
        yaml_path = find_yaml(root)
        print("--- data.yaml asli ---")
        print(yaml_path.read_text()[:500])
        clean = remap(yaml_path)
        z.unlink(missing_ok=True)
        train(clean, args.epochs)
    finally:
        KEY_FILE.unlink(missing_ok=True)
        print("--- key di VM dihapus")


if __name__ == "__main__":
    main()
