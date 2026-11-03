#!/usr/bin/env python3
"""Validasi TINGKAT HITUNGAN pada split validasi Roboflow.

Kenapa perlu: `mAP50` tidak sama dengan galat jumlah pohon. Skrip ini
menjalankan pipeline produksi (conf + dedupe jarak minimum yang sama dengan
`predict.py`) pada citra val, lalu melaporkan:

  - bias jumlah  : (Σ prediksi − Σ anotasi) / Σ anotasi
  - MAE per citra
  - recall / precision dengan pencocokan IoU >= 0.5
  - galat relatif median per citra

Pemakaian:
    export RF_KEY=...            # kunci Roboflow, TIDAK ditulis ke disk
    python3 validate_count.py    # unduh dataset ke /tmp, hapus setelahnya

Hasil acuan (2026-11-02, v15, bobot roboflow_palm_best.pt):
    conf 0.25 + dedupe 5 m -> 29.000 titik vs 26.913 anotasi (+7,75%),
    recall 84,6%, precision 78,5%
Catatan: angka precision ini tertekan karena anotasi dataset tidak lengkap
(banyak tajuk tanpa kotak); nilai mAP Ultralytics (P 0,955 R 0,955) dihitung
setelah penyaringan conf sehingga selalu lebih optimistis.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def download(ws: str, proj: str, ver: int, work: Path) -> None:
    if (work / "valid").exists():
        print("dataset sudah ada:", work)
        return
    key = os.environ["RF_KEY"]
    url = f"https://api.roboflow.com/{ws}/{proj}/{ver}/yolov8?api_key={key}"
    info = json.load(urllib.request.urlopen(url, timeout=120))
    link = (info.get("export") or {}).get("link")
    if not link:
        raise SystemExit("ekspor belum siap: " + json.dumps(info)[:300])
    z = Path("/tmp/rf_val.zip")
    print("unduh dataset ...", flush=True)
    urllib.request.urlretrieve(link, z)
    work.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(work)
    size_mb = z.stat().st_size / 1e6
    z.unlink()
    print("zip %.1f MB diekstrak" % size_mb)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def dedupe(boxes: np.ndarray, conf: np.ndarray, min_px: float):
    keep = []
    for i in np.argsort(-conf):
        cx, cy = (boxes[i][0] + boxes[i][2]) / 2, (boxes[i][1] + boxes[i][3]) / 2
        if keep and np.any(np.hypot(np.array(keep)[:, 0] - cx,
                                    np.array(keep)[:, 1] - cy) < min_px):
            continue
        keep.append((cx, cy, i))
    idx = [k[2] for k in keep]
    return (boxes[idx], conf[idx]) if idx else (np.zeros((0, 4)), np.zeros((0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default="mahir-sehmi-fgblg")
    ap.add_argument("--project",
                    default="oil-palm-tree-crown-detection-from-aerial-image")
    ap.add_argument("--version", type=int, default=15)
    ap.add_argument("--weights", default="roboflow_palm_best.pt")
    ap.add_argument("--data", default="/tmp/rf_val")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--dedupe-m", type=float, default=5.0)
    ap.add_argument("--gsd", type=float, default=0.155138,
                    help="m/px citra latih, untuk mengubah dedupe meter -> piksel")
    ap.add_argument("--out", default="validasi_roboflow.json")
    args = ap.parse_args()

    work = Path(args.data)
    download(args.workspace, args.project, args.version, work)
    val = work / "valid"
    imgs = sorted((val / "images").glob("*"))
    min_px = args.dedupe_m / args.gsd
    print(f"citra val: {len(imgs)} | dedupe {args.dedupe_m} m = {min_px:.1f} px")

    model = YOLO(args.weights)
    results = model.predict([str(p) for p in imgs], imgsz=640, conf=args.conf,
                            iou=args.iou, device="cpu", verbose=False, batch=8)
    n_gt = n_pred = tp = 0
    absd, rel = [], []
    for p, r in zip(imgs, results):
        h, w = r.orig_shape
        gt = []
        for line in (val / "labels" / (p.stem + ".txt")).read_text().splitlines():
            c = line.split()
            if len(c) >= 5:
                cx, cy, bw, bh = (float(v) for v in c[1:5])
                gt.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                           (cx + bw / 2) * w, (cy + bh / 2) * h])
        gt = np.array(gt).reshape(-1, 4)
        boxes, conf = dedupe(r.boxes.xyxy.numpy(), r.boxes.conf.numpy(), min_px)

        m = iou_matrix(boxes, gt)
        used, t = set(), 0
        for i in np.argsort(-conf):
            if not len(gt):
                break
            free = np.where(np.isin(np.arange(len(gt)), list(used)), -1.0, m[i])
            j = int(np.argmax(free))
            if free[j] >= 0.5:
                used.add(j)
                t += 1
        n_gt += len(gt)
        n_pred += len(boxes)
        tp += t
        absd.append(abs(len(boxes) - len(gt)))
        if len(gt):
            rel.append((len(boxes) - len(gt)) / len(gt))

    fp, fn = n_pred - tp, n_gt - tp
    out = {"images": len(imgs), "conf": args.conf, "dedupe_m": args.dedupe_m,
           "n_gt": n_gt, "n_pred": n_pred, "tp": tp, "fp": fp, "fn": fn,
           "bias_pct": round((n_pred - n_gt) / n_gt * 100, 2),
           "mae_per_image": round(float(np.mean(absd)), 2),
           "median_rel_err_pct": round(float(np.median(rel)) * 100, 2),
           "recall_pct": round(tp / (tp + fn) * 100, 2),
           "precision_pct": round(tp / (tp + fp) * 100, 2),
           "f1": round(2 * tp / (2 * tp + fp + fn), 3)}
    print(json.dumps(out, indent=2))
    json.dump(out, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
