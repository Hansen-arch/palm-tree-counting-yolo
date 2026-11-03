#!/usr/bin/env python3
"""
Training wrapper tipis di atas Ultralytics - meniru notebook 02
(`02_model_train_eval.ipynb`) dari repo oilpalm-tree-detection-yolov26, tetapi
tanpa Colab/Roboflow: dataset lokal dari `make_tiles.py`.

    # fine-tune dari backbone COCO (butuh GPU kalau mau cepat)
    python3 train.py --model yolo11n.pt --data dataset/data.yaml --epochs 60

    # kalau sudah punya bobot model sawit (mis. dari penulis repo / Zenodo)
    python3 train.py --model weights/oilpalm_yolov26n.pt --data dataset/data.yaml \
        --epochs 20 --lr0 0.001

CPU-only tetap bisa jalan, tetapi sangat lambat: pakai --epochs 5 --imgsz 512
--batch 2 untuk sekedar menguji pipeline.
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt",
                    help="bobot awal: yolo11n.pt / yolo12n.pt / bobot sawit .pt")
    ap.add_argument("--data", default="dataset/data.yaml")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="cpu | 0 | 0,1 (kosong = auto)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="oilpalm")
    ap.add_argument("--val-only", action="store_true", help="hanya evaluasi bobot")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics belum terpasang: pip install -r requirements.txt")

    if not os.path.exists(args.data):
        sys.exit(f"data.yaml tidak ditemukan: {args.data} (jalankan make_tiles.py)")

    model = YOLO(args.model)
    if args.val_only:
        m = model.val(data=args.data, imgsz=args.imgsz, device=args.device)
    else:
        model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                    batch=args.batch, device=args.device, workers=args.workers,
                    patience=args.patience, lr0=args.lr0,
                    project=args.project, name=args.name, exist_ok=True)
        best = os.path.join(args.project, args.name, "weights", "best.pt")
        print(f"\nbobot terbaik -> {best}")
        print(f"lanjut: python3 predict.py --weights {best}")
        return
    print(m)


if __name__ == "__main__":
    main()
