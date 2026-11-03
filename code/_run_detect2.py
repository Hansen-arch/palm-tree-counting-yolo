#!/usr/bin/env python3
"""Jalankan _colab_detect.py di VM Colab untuk PMBN_ORTO2 (3 strip berisi data).

strip3 (baris 10966-16449 dari raster resample) hampir seluruhnya transparan
(319 KB dari 246 MB), jadi dilewati: tidak ada piksel valid di sana.

Pemakaian: colab exec -f /tmp/run_detect2.py --timeout 3000
"""
import runpy
import sys

sys.argv = [
    "_colab_detect.py",
    "--ortho", "/content/orto/strip1.tif",
    "--ortho", "/content/orto/strip2.tif",
    "--ortho", "/content/orto/strip4.tif",
    "--weights", "/content/weights/roboflow_palm_best.pt",
    "--out-dir", "/content/new_ortho2",
    "--tile", "640",
    "--target-gsd", "0.155138",
    "--overlap", "96",
    "--conf", "0.25",
    "--iou", "0.7",
    "--min-dist", "5.0",
    "--device", "0",
    "--batch", "8",
]
runpy.run_path("/content/_colab_detect.py", run_name="__main__")
