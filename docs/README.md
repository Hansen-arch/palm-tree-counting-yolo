# yolov26 / YOLO palm detection, separate folder

A YOLO pipeline for detecting oil palm trees, following the notebook recipe from
<https://github.com/mwahyur46/oilpalm-tree-detection-yolov26>
(`01_data_preparation` -> `02_model_train_eval` -> `03_inference_model`), but
rewritten as lightweight scripts that run without Colab and without Roboflow.

This folder is **separate** from the digitising work. It writes nothing to
`../output/` or to `../palm_canopy_digitasi.py`; the only things read from
outside are the orthophoto and the crown polygons as label sources.

## Important: the weights are not in that repo

That repository **ships no `.pt` files**: the `model/` folder is missing on
GitHub (404) and its README states data + weights will be released through
Zenodo (or on request to the authors). So "ready-to-use model" there means the
training recipe, not a detector you can run straight away.

Three routes are available here:

| route | command | notes |
|---|---|---|
| A. you have palm `.pt` weights (from the repo author / Zenodo) | `predict.py --weights weights.pt` | fastest, works immediately |
| B. train from our own labels (13,983 crowns) | `make_tiles.py` -> `train.py` | GPU needed for decent results; CPU only for pipeline tests |
| C. start from a COCO backbone | `train.py --model yolo11n.pt` | same as notebook 02 |

## Flow

```bash
cd yolo_oilpalm
pip install -r requirements.txt          # needs system GDAL (python3-gdal)

# 1. dataset: cut the orthophoto into 640 px tiles + bbox labels from our digitising
python3 make_tiles.py --ortho ../ORTO271026.tif \
    --crowns ../output/palm_crowns.gpkg --out dataset

# 2. training (GPU: 60 epochs; on CPU start with small numbers)
python3 train.py --model yolo11n.pt --data dataset/data.yaml \
    --epochs 5 --imgsz 512 --batch 2 --device cpu      # pipeline test
python3 train.py --model yolo11n.pt --data dataset/data.yaml --epochs 60   # GPU

# 3. inference over the whole orthophoto -> tree points + 5 m filter
python3 predict.py --weights runs/oilpalm/weights/best.pt \
    --ortho ../ORTO271026.tif --out hasil/pohon --device cpu
```

`predict.py` output: `hasil/pohon.gpkg`, `.geojson`, `.csv` (columns `conf`,
`box_w_m`, `box_h_m`, `box_area_m2`), all in the orthophoto CRS (EPSG:32648).

## On Colab: three orthophotos at once

`colab_oilpalm.ipynb` runs everything on Colab (T4 GPU). Required under
`MyDrive/oilpalm/`: the three orthophotos, `output/palm_crowns.gpkg`, and this
`yolo_oilpalm/` folder.

| orthophoto | GSD | size | CRS | notes |
|---|---|---|---|---|
| `ORTO271026.tif` | 0.155 m | 18,708 x 16,672 | 32648 | label source |
| `PMBN_ORTO1.tif` | 0.099 m | 8,756 x 16,684 | 32648 |, |
| `PMBN_ORTO2.tif` | 0.071 m | 28,048 x 47,664 | written as **32748** | actually 48N -> `--assume-srs EPSG:32648` |

Notebook cell order: dependency check -> Drive mount -> GSD/CRS check ->
`make_tiles` (ORTO26) -> `train` -> quick test on 8 tiles -> predict all three
orthophotos -> summary + zip -> visual QA.

**GSD must be made uniform.** The model was trained on 640 px tiles at 0.155 m
(99 m of ground). On PMBN_ORTO2 (0.071 m) a 640 px tile covers only 46 m, so
trees appear 2x larger. `predict.py --target-gsd 0.155138` reads a 1,389 px
window and shrinks it to 640 px, so tree scale is identical across all three
orthophotos, and the ORTO2 tile count drops from ~3,300 to ~735.

Maximum tile counts (before empty tiles outside the footprint are dropped):
ORTO271026 ~810, PMBN_ORTO1 ~153, PMBN_ORTO2 ~735.

Estimated time on a T4: dataset 25 seconds, 60-epoch training 10–15 minutes,
predicting the three orthophotos 5–10 minutes.

If you already have palm weights (from the repo author / Zenodo), skip the
training cells: point `WEIGHTS` at that file and run only the prediction cells.

## Why the labels come from our own digitising

There is no public labelled dataset for this block, and Roboflow would require
manual annotation. The existing digitising (`output/palm_crowns.gpkg`, 13,983
polygons) provides free bounding boxes. These are *pseudo-labels*: the model
will imitate that digitising, including its errors, so this is not independent
proof, but it is far faster than per-tile YOLO annotation as data grows, and it
can be used to check other blocks (PMBN_ORTO1/2) with a consistent appearance.

The most honest way to use it: train the model, then compare its output against
the digitising on blocks that were never used for training.

## Lightweight setup (CPU)

* `make_tiles.py --tile 640` -> **203 tiles** for this orthophoto (only inside
  the flown footprint), 15,436 bounding boxes, 112 MB, no GPU needed. This
  dataset is already built in `dataset/` (169 train / 34 val), so `train.py` can
  be used directly.
* The box count is slightly higher than the crown count (13,983) because crowns
  cut by a tile edge are recorded on both tiles with clipped boxes, which is
  actually good for training, so the model recognises half trees at tile edges.
* `train.py --epochs 5 --imgsz 512 --batch 2 --device cpu` only proves the
  pipeline runs; 13,983 trees need a GPU (a Colab T4 is enough, as in the
  original repo).
* `predict.py --tile 640 --overlap 96` on CPU takes about 0.1 s/tile; the whole
  footprint is ~800 tiles ~ 1–2 minutes (model load time excluded).
* RAM: the orthophoto is 18,708 x 16,672 px; tiles are read chunk by chunk, so
  the full raster is never loaded into memory.

## Folder contents

| file | purpose |
|---|---|
| `make_tiles.py` | orthophoto + crown polygons -> `dataset/` (PNG tiles + YOLO labels + `data.yaml`) |
| `train.py` | thin wrapper around `ultralytics.YOLO(...).train()` / `.val()` |
| `predict.py` | overlapping-tile inference -> GeoJSON/GPKG/CSV + 5 m distance filter |
| `render_points.py` | visual QA: detection points drawn over the orthophoto (no torch) |
| `colab_oilpalm.ipynb` | Colab notebook: dataset -> train -> predict 3 orthophotos -> summary |
| `requirements.txt` | dependencies (ultralytics, torch, scipy, pillow; GDAL from the system) |
