# Palm canopy digitising — ORTO271026.tif

## 1. What the ortho actually contains

`ORTO271026.tif` is a plain UAV orthophoto, not a surface model:

| property | value |
|---|---|
| size | 18,708 × 16,672 px |
| ground pixel | 0.155 m |
| extent | 2,902 × 2,586 m ≈ 749 ha of raster box |
| CRS | WGS 84 / UTM zone 48N (EPSG:32648) |
| bands | 1 Red, 2 Green, 3 Blue, 4 Alpha (all `Byte`) |
| elevation | **none** — no Z band, no elevation metadata |

The alpha band shows the flown footprint is 138.1 ha (irregular, follows the
planting blocks and roads).

## 2. Palm canopy per palm (digitasi)

Result of `palm_canopy_digitasi.py` on the whole orthophoto:

| metric | value |
|---|---|
| palms digitised | **13,983** |
| density | **101.3 palms/ha** |
| mean implied spacing | 9.9 m |
| canopy area (sum of crowns) | **32.76 ha** |
| canopy cover | **23.7 %** of the flown footprint |
| crown area | mean 23.4 m², median 23.2 m², p10 9.3 m², p90 37.5 m² |
| crown diameter (equivalent) | mean 5.3 m |
| centre quality | mean 0.16 m from its crown's centroid, p90 0.52 m, 98.4 % of the points on crown pixels |

### How it was digitised

1. **Crown response** — a crown is the brightest, most yellow-green,
   most frond-textured thing in a block, and it sits darker than the bare
   ground and grass next to it. Response = excess green `g − mean(r,b)`, gated
   by local variance *and* by a positive dark response (background − local
   brightness, DoG 9 m vs 1 m). Grass is as green as a frond but has no local
   contrast, so the dark gate is what rejects it.
2. **Crown mask** — response > 12 (excess-green units), morphologically cleaned,
   clipped to the alpha footprint.
3. **Palm centres** — seeds are per-crown maxima of the smoothed response
   (0.5 m smoothing, 2.4 m non-maximum window), deliberately over-seeded.
   Every seed is then **settled onto the response-weighted centroid of the
   crown pixels within 4.2 m**, iterated 5 times (mean movement 0.76 → 0.09 m).
   This is what fixes points that used to land on the edge of a crown: a raw
   peak sits wherever the mask happens to be thickest, whereas the weighted
   centroid sits on the palm. Seeds that collapse onto the same crown are then
   thinned to 4.2 m, keeping the one carrying more crown response.
4. **Crown-under-point check** — for every centre, how much of the 2.5 m disc
   around it is crown (`crown_frac`) and how thick the crown is at the point
   (`centre_dt`, metres). A point on empty ground fails both; a palm passes
   both. Defaults drop `crown_frac < 0.20` or `centre_dt < 0.6 m`, which removed
   2,896 of 16,879 centres (17 %).
5. **Crown polygons** — every mask pixel is assigned to its nearest centre
   (Voronoi delineation, capped at 4.5 m radius), dissolved per palm, then
   simplified at 0.25 m to remove the raster staircase. The stored point is the
   crown's own weighted centroid.

### Accuracy — read this before quoting the numbers

* **Centring is measured, not assumed**: mean offset 0.16 m between the point
  and its crown's centroid, p90 0.52 m, and 98.4 % of points fall on crown
  pixels. `output/qa_centres_dense.jpg` shows them drawn on the full-resolution
  imagery with a circle at their crown radius.
* **Points on empty ground** are handled by the crown-under-point check. Without
  it the detection yields 16,879 points (122/ha) and about a fifth of the lowest
  ones sit on grass or bare soil; with the default cut it yields 13,983
  (101/ha). What remains is mostly where the crown mask itself covers turf, so a
  stricter cut (`--min-crown-frac 0.35 --min-dt-centre 0.9`, 84/ha) is available
  if precision matters more than count.
* **A frond-spoke score was tried and rejected as a filter.** It scores the
  angular variation of brightness at 2–3 m from each point — palms swing, flat
  ground does not — but its absolute level follows the lighting of each block,
  and once normalised against the local background it flags real palms in dense
  rows (the background there is crowns). It is kept as a CSV column only.
* **Recall is still the weak side in the densest rows.** Touching crowns with no
  contrast saddle between them merge into one crown; `--min-sep 3.8` splits them
  harder.
* **Cover (23.7 %) is the firmest number**, because it depends only on the
  crown mask, not on how crowns are split.
* Everything is in the vector outputs, so a missed or merged palm can be added
  or split in QGIS without re-running.
* `output/qa_overlay.png` is whole-block scale (a 5 m crown is one pixel) — use
  `--crop ... --tune` for a readable check, or render the CSV over the ortho as
  in `qa_centres_dense.jpg`.

## 3. DTM — not possible from these files, scaffolding provided

An ortho GeoTIFF holds only visible bands. There is **no elevation data in the
project**: `ORTO271026.tif`, `PMBN_ORTO1.tif` and `PMBN_ORTO2.tif` are all
4-band RGB + alpha orthophotos, and no DSM/point cloud/DTM file exists next to
them. A DTM therefore cannot be derived from the ortho itself — elevation has
to come from the drone flight (point cloud / DSM export), a LiDAR survey, or a
public DEM.

Prepared so a DTM drops straight in:

```bash
# 1. is there elevation in the file?  -> no
python3 add_dtm_to_orto.py --ortho ORTO271026.tif --check

# 2. an empty DTM template on exactly the ortho grid (0.155138 m, EPSG:32648)
#    -> ORTO271026_DTM_template.tif   (use as target grid in QGIS / Metashape)
python3 add_dtm_to_orto.py --ortho ORTO271026.tif --template

# 3. once you have a DTM, put it in the ortho file as band 5
python3 add_dtm_to_orto.py --ortho ORTO271026.tif --dtm my_dtm.tif \
        --out ORTO271026_with_DTM.vrt          # multi-band, mixed types
python3 add_dtm_to_orto.py --ortho ORTO271026.tif --dtm my_dtm.tif \
        --out ORTO271026_with_DTM.tif --cog    # single Cloud-Optimised GeoTIFF
```

The DTM is reprojected/resampled onto the ortho grid and added as band 5
(`DTM_elevation_m`), so QGIS sees one file with imagery + elevation.

## 4. Data issue found in PMBN_ORTO2.tif

`PMBN_ORTO2.tif` is tagged **WGS 84 / UTM zone 48S (EPSG:32748)** with false
northing 10,000,000, but its coordinates (northing ≈ 10,036,343) resolve to
**0°21′ N** — north of the equator, i.e. zone **48N**, the same zone as
`ORTO271026.tif`. As tagged, the file sits in the wrong hemisphere. Set its CRS
to EPSG:32648 before using it with the other layers.

## 5. Files

| file | content |
|---|---|
| `palm_canopy_digitasi.py` | the digitising script (crop mode for QA) |
| `add_dtm_to_orto.py` | DTM check / template / attach-to-ortho helper |
| `output/palm_crowns.gpkg` | 13,983 crown polygons (`crown_id`, `area_m2`, `diam_m`) |
| `output/palm_centres.gpkg` | 13,983 points, one per palm, at the crown centroid, same attributes |
| `output/palm_canopy_digitasi.csv` | per-palm table: id, UTM x/y, crown area, diameter, and the three QA scores (`spoke`, `spoke_rank`, `crown_frac`, `centre_dt`) |
| `output/palm_canopy_stats.json` | the summary numbers above, including the centring metrics |
| `output/qa_centres_dense.jpg` | 15 centres drawn on the full-resolution imagery with their crown circles |
| `output/palm_crown_mask.tif` | binary canopy mask (0/1) on the ortho grid |
| `output/qa_overlay.png` | visual check of the digitising |
| `ORTO271026_DTM_template.tif` | empty band on the ortho grid, ready for a DTM |
| `run_full.log` | log of the production run |

## 6. Re-running / tuning

The run is two steps: the vector step needs its own process, because GDAL keeps
its own label raster open until the interpreter exits.

```bash
python3 palm_canopy_digitasi.py --outdir output --tune        # detection + raster (~2 min)
python3 palm_canopy_digitasi.py --vector-only --outdir output # polygons + points (~1 min)
python3 palm_canopy_digitasi.py --crop 17447 10806 17747 11106 --tune --zoom 4   # QA crop
python3 palm_canopy_digitasi.py --min-sep 3.8    # split touching crowns harder
python3 palm_canopy_digitasi.py --crown-thr 9    # catch fainter crowns (more cover)
python3 palm_canopy_digitasi.py --crown-thr 16   # only obvious, dense crowns
python3 palm_canopy_digitasi.py --resp dark      # dark-rosette response only
python3 palm_canopy_digitasi.py --refine-iters 0 # raw peaks, no centring (for comparison)

# precision vs count: how much crown must sit under a point
python3 palm_canopy_digitasi.py --min-crown-frac 0.35 --min-dt-centre 0.9   # stricter
python3 palm_canopy_digitasi.py --min-crown-frac 0  --min-dt-centre 0       # keep all
python3 palm_canopy_digitasi.py --min-sep 3.8 --crown-thr 9                 # denser, fainter
```
