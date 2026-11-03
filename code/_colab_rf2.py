"""Unduh dataset Roboflow pakai SDK (lokasi default), remap kelas, train.
Memanfaatkan fungsi yang sudah ada di /content/_colab_roboflow.py."""
import os
import sys
from pathlib import Path

sys.path.insert(0, "/content")
os.chdir("/content")

import _colab_roboflow as base
from roboflow import Roboflow

try:
    k = base.key()
    print("--- key terbaca, panjang:", len(k))
    rf = Roboflow(api_key=k)
    proj = rf.workspace("mahir-sehmi-fgblg").project(
        "oil-palm-tree-crown-detection-from-aerial-image")
    vs = proj.versions()
    print("--- versi:", [(getattr(v, "version", None), getattr(v, "images", None)) for v in vs])
    target = 15
    v = [x for x in vs if int(getattr(x, "version", 0) or 0) == target][0]
    print("--- unduh v", target, "...")
    ds = v.download("yolov8")
    print("--- lokasi:", ds.location)
    root = Path(ds.location)
    print("--- isi:", sorted(p.name for p in root.iterdir())[:10])
    yaml_path = base.find_yaml(root)
    print("--- data.yaml ---")
    print(yaml_path.read_text()[:400])
    clean = base.remap(yaml_path)
    base.train(clean, 50)
finally:
    base.KEY_FILE.unlink(missing_ok=True)
    print("--- key di VM dihapus")
