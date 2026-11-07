# Palm Tree Counting, YOLOv8 + Orthomosaic GIS Pipeline

A complete pipeline for counting oil palm trees from aerial orthophotos: imagery
is prepared and merged, trees are detected with a YOLOv8 detector running on a
GPU in Google Colab, detections are converted into georeferenced point layers,
and the points are then analysed against vector zones such as river setbacks.

Every number in the results below comes from real production data (5
orthophotos across UTM 47N/48S, 310+ hectares), not from sample data.

---

## Main results

| Area | Valid area | Trees | Density | Median conf |
|---|---|---|---|---|
| Mosaic of 4 orthos (47N) | 310.8 ha | 24,966 | 80.3 /ha | 0.697 |
| ORTO 47N (run as its own file) | 138.1 ha | 12,864 | 93.1 /ha |, |
| PMBN1 47N | 63.5 ha | 6,008 | 94.6 /ha | 0.755 |
| PMBN3 47N | 48.2 ha | 399 | 8.3 /ha | 0.633 |
| PMBN ORTO2 (UTM 48S) | 90.2 ha | 8,040 | 89.1 /ha | 0.704 |

Accuracy of the detector (YOLOv8n fine-tuned on Roboflow Universe v15, 2,730
images / 180,170 boxes, evaluated on a 280-image validation split):

| Metric | Value |
|---|---|
| mAP@50 | 0.977 |
| Precision / Recall | 0.955 / 0.955 |
| mAP@50-95 | 0.745 |
| Tree count bias in production use | ±8 % |
| Recall / Precision at IoU 0.5 | 84.6 % / 78.5 % (F1 0.814) |
| Agreement with an independent algorithmic digitising pass | 92 % |

Consistency between runs (detecting each ortho separately versus detecting the
merged mosaic once):

| Source | per-ortho | on mosaic | delta |
|---|---|---|---|
| ORTO 47N | 93.1/ha | 93.0/ha | −0.1 % |
| PMBN1 47N | 94.6/ha | 98.8/ha | +4.4 % |
| PMBN ORTO2 (UTM 48S) | 89.1/ha | 86.9/ha | −2.5 % |

River setback analysis (three orthophotos tested against a 500 m setback
shapefile and 5 m wide river polygons):

| Item | Value |
|---|---|
| Trees inside the 500 m setback | 2,253 of 6,495 |
| Trees inside the 5 m river polygons | 100 (across 6 river segments) |
| Setback clip area per ortho | 20.8 / 5.3 / 20.0 ha |
| River clip area per ortho | 1.05 / 0.27 / 1.02 ha |

---

## What the code does

The pipeline takes georeferenced orthophotos as input and produces a point
layer where every detected tree carries its map coordinates, a confidence
value and the physical size of its bounding box. Nothing about the inputs is
assumed beyond what an orthophoto normally provides: an RGB image (optionally
with an alpha band), a coordinate reference system and a ground sample
distance. Any plantation imagery with those properties can be processed the
same way.

Work is split between your machine and a free GPU in Google Colab. Local
scripts handle everything that needs GIS libraries or large disk space, while
the GPU is only used for the neural network inference itself, which is the
slow part. Communication between the two is plain CSV files, so any step can
be re-run or inspected independently.

## Methods used

Mosaicking. When several orthophotos cover the same area with overlapping
edges, detecting on each file separately counts every tree inside the overlap
twice, which inflates the total proportionally to how much the files overlap.
The pipeline therefore merges all inputs into a single raster before
detection, so each location on the ground is visited exactly once. The output
grid is snapped to the pixel grid of the sharpest source image instead of
being derived from the union of all extents, which keeps that source free of
any resampling blur. Coarser sources are resampled with an interpolation that
preserves edges, the result is stored with lossless compression so pixel
values survive a rebuild unchanged, and overview pyramids are generated so
desktop GIS software can display a multi-gigapixel file smoothly. The same
approach works whether the inputs are two files or ten, as long as they share
a CRS.

Resolution matching. A detector only recognises objects at the scale it was
trained on. If the model learned trees that span a certain number of pixels,
feeding it imagery with a different ground resolution makes every tree appear
too large or too small, and accuracy drops even though the image is sharper.
Before inference the imagery is resampled so that one pixel on the ground
matches the pixel size used during training. This also cuts the amount of data
slidethrough the network, since very high resolution imagery would otherwise
be processed at several times the necessary cost.

Sliding-window inference. The raster is too large to fit into GPU memory at
once, so it is cut into strips locally, then processed as overlapping square
tiles on the GPU. Overlap between neighbouring tiles is kept deliberately
large: a tree that straddles a tile edge would otherwise be seen only partially
and could be missed or split into two detections. Every tile goes through the
standard detector post-processing: raw boxes are filtered by a confidence
threshold, then non-maximum suppression removes boxes that overlap each other
beyond a set IoU. The alpha band, when present, is used to skip transparent
gaps between flight corridors, so no detections are invented over empty areas.
Tile results are written as CSV with pixel-independent map coordinates, which
makes them trivial to merge and to verify.

Duplicate suppression. Confidence-based suppression alone is not enough in a
dense, regularly planted orchard: two high-confidence boxes can sit on the
same crown, and near strip boundaries the same crown can be detected once per
strip with slightly shifted centres. After all tile results are merged, a
spatial filter removes any detection closer than a minimum distance to a
stronger one, keeping the higher-confidence detection. The distance threshold
is derived from the expected planting spacing, and it is applied globally
across the whole dataset rather than per tile, which is what closes the
leakage at strip boundaries. The spacing of the surviving points can then be
checked against the known planting distance as a sanity test.

From boxes to points. Each retained box is reduced to its centre point, and
the box dimensions are converted from pixels to metres using the raster
resolution, so downstream users get crown size statistics for free. Points are
written as GeoPackage, GeoJSON and CSV in the CRS of the input imagery, ready
to load in QGIS or any other GIS package.

Validation. Three complementary checks are used. Detection quality is measured
on a labelled validation split with the usual object-detection metrics (mAP at
different IoU thresholds, precision, recall). Counting quality is measured
separately, because a detector can score well on mAP while still over- or
under-counting: predictions are matched against annotations by IoU and the
total bias in the number of trees is reported. Finally, run-to-run consistency
is checked by comparing density per source area when the same imagery is
processed in two different ways, and by comparing the drop in total count
against an independent estimate of how much the source files overlapped. Field
accuracy is not claimed from these numbers alone; narrowing the error further
requires ground plots.

Spatial analysis. The resulting point layers can be intersected with any
polygon layer. The included analysis counts how many detected trees fall
inside a river setback zone and inside narrow river polygons, reports the
count per river segment, and clips the source rasters to those zones so the
area can be inspected visually. Because points and zones share the same CRS,
this is a plain point-in-polygon test and needs no reprojection.

---

## Architecture

```
orthophoto (.tif, any EPSG with RGB + optional alpha)
   |  merge_orto.py            -> single mosaic, aligned grid, lossless, overviews
   |  prep_model_gsd.py        -> resample to the GSD the model was trained at
   |  split_mosaic_strips.py   -> cut into strips small enough to upload
   v
Google Colab (GPU)
   |  _colab_detect.py         -> overlapping tiles, conf threshold, NMS
   |                             min-distance dedupe -> CSV per strip
   v
new_orto_report.py             -> global dedupe -> GPKG/GeoJSON/CSV + QA render
analisis_sempadan_sungai.py    -> point-in-polygon counts + raster clip per zone
validate_count.py              -> IoU matching against labelled validation data
```

The QA renders produced along the way draw every detection back onto the
imagery, so results can be reviewed by eye before anything is published.

---

## Repository layout

| Path | Contents |
|---|---|
| `code/` | the whole pipeline (mosaic, detection, validation, spatial analysis) |
| `colab/colab_oilpalm.ipynb` | Colab notebook for training + inference |
| `docs/` | technical report, QGIS guide, result summaries (JSON), training curves |
| `docs/images/` | detection output and QA visuals |
| `sample/` | sample tree point output (CSV) |

## Running it

```bash
pip install -r requirements.txt

# 1. merge the orthos into one mosaic
python code/merge_orto.py --build

# 2. resample to the model GSD, cut into strips
python code/prep_model_gsd.py
python code/split_mosaic_strips.py

# 3. inference on Colab (GPU) -> CSV per strip, then merge + render QA
python code/new_orto_report.py --dir output/mosaik --name mosaik47n \
    --parts strip1 strip2 ... --raster intermediates/mosaik_155/mosaik47n_155.tif

# 4. setback / river analysis
python code/analisis_sempadan_sungai.py
```

Per-run numbers are in `docs/REPORT.md` and `docs/ringkasan_*.json`.

---

<p align="center"><img src="docs/images/qa_mosaik47n_zoom.jpg" width="49%"> <img src="docs/images/training_curves.png" width="49%"></p>
