#!/usr/bin/env python3
"""
Digitasi tajuk sawit per pohon dari ortofoto UAV / per-palm crown digitising.

Pipeline (pixel work at a 2x-decimated overview, ~0.31 m/px)

  1. Crown response : a palm crown is the brightest, most yellow-green,
                      most frond-textured thing in a block, and it is darker
                      than the bare ground and grass around it. The response is
                      excess green (g - mean(r,b)) gated by local variance and
                      by a positive dark response (background - local
                      brightness, difference of Gaussians 9 m vs 1 m). Grass
                      is just as green as a frond but has no local contrast, so
                      the dark gate is what removes it. Cached to a Float32
                      raster so later passes are cheap.
  2. Crown mask     : response above CROWN_THR, morphologically cleaned,
                      clipped to the ortho footprint (alpha band).
  3. Palm centres   : seeds = distance-transform peaks of the mask (which only
                      says "a crown is somewhere in here"), then each seed is
                      settled onto the centroid of the crown pixels around it,
                      weighted by the crown response. A raw distance peak sits
                      wherever the mask is thickest, which for a lopsided or
                      leaky crown is off to one edge; iterated mean-shift on the
                      response puts the point on the palm itself.
  4. Dedupe         : seeds that collapse onto each other after settling are
                      thinned to MIN_SEP_M, keeping the crown with more mass.
  5. Crown polygons : every mask pixel goes to its nearest centre (Voronoi
                      delineation capped at MAX_CROWN_RADIUS_M); the dissolved
                      pixels form the digitised crown, its area, and finally the
                      point is snapped to the crown's own weighted centroid.

Run it in two steps; the vector step needs its own process because GDAL keeps
the label raster open until the interpreter exits:

    python3 palm_canopy_digitasi.py --outdir output --tune
    python3 palm_canopy_digitasi.py --vector-only --outdir output

Outputs
    output/palm_crowns.gpkg, output/palm_centres.gpkg, palm_canopy_digitasi.csv,
    palm_canopy_stats.json, palm_crown_mask.tif, qa_overlay.png
"""
import argparse, csv as csvmod, gc, json, os, sys
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from osgeo import gdal, ogr, osr

gdal.UseExceptions()
ogr.UseExceptions()

OVERVIEW = 1                 # 2x decimation -> ~0.31 m/px
DOG_INNER_M = 1.0            # fine scale (frond clump)
DOG_OUTER_M = 9.0            # background scale (grove / illumination)
MIN_SEP_M = 4.2              # minimum distance between two palm centres
MAX_CROWN_RADIUS_M = 4.5     # cap on the Voronoi crown radius (63 m2 crown)
MIN_CROWN_AREA_M2 = 2.0      # drop slivers
CROWN_THR = 12.0             # crown response cut (excess-green scale)
DT_MIN_M = 1.0               # seed needs this much crown thickness around it
PEAK_WIN_M = 2.4             # seed non-maximum window
SMOOTH_M = 0.5               # response smoothing used for per-crown maxima
PEAK_THR = 3.0               # smoothed-response floor for a seed
DARK_GATE = 3.0              # 'both' mode: crown must also be darker than its background
SPOKE_MIN = 0.0              # frond spoke contrast required at a centre (0 = off)
SPOKE_RADII_M = (2.0, 2.6, 3.2)   # radii where palm fronds still radiate
SPOKE_PX = 0.155             # full-resolution pixel of the ortho, in metres
SPOKE_RANK = 0.0             # keep centres above this local rank (0 = off)
SPOKE_LATTICE_M = 12.0       # background sampling step for the rank
MIN_CROWN_FRAC = 0.20        # crown must cover this much of the 2.5 m disc
MIN_DT_CENTRE = 0.6          # crown must be at least this thick at the centre (m)
REFINE_ITERS = 5             # mean-shift passes that centre the seeds
REFINE_RADIUS_M = 4.2        # crown radius used when settling a seed
SIMPLIFY_M = 0.25            # polygon simplification (staircase removal)
TEX_MIN = 25.0               # local variance gate: fronds are textured, shadow is not
TEX_WIN_M = 1.5
TILE = 512
OVERLAP = 64


class Reader:
    """Windowed reader, works in overview pixels (2x decimated)."""

    def __init__(self, path, x0, y0, x1, y1, mode="both"):
        self.mode = mode
        self.ds = gdal.Open(path)
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        gt = self.ds.GetGeoTransform()
        self.px = abs(gt[1]) * (2 ** OVERVIEW)
        self.gt_o = (gt[0] + x0 * 2 * gt[1], gt[1] * 2, gt[2],
                     gt[3] + y0 * 2 * gt[5], gt[4], gt[5] * 2)
        self.wkt = self.ds.GetProjection()

    def read(self, tx, ty, tw, th, bands=(1, 2, 3, 4)):
        return [self.ds.GetRasterBand(b).ReadAsArray(
            (self.x0 + tx) * 2, (self.y0 + ty) * 2, tw * 2, th * 2,
            buf_xsize=tw, buf_ysize=th) for b in bands]

    def response(self, tx, ty, tw, th):
        """Crown score, gated by frond texture.
        dark  = background minus local brightness (dark rosettes).
        green = excess green, g - mean(r, b): palm fronds are the brightest,
                most yellow-green thing in the block, while the shaded gaps
                between palms are dark but not green."""
        g, r, b = [x.astype(np.float32) for x in self.read(tx, ty, tw, th, (2, 1, 3))]
        if self.mode in ("green", "both"):
            den = np.maximum(g + r + b, 1.0)
            exg = 255.0 * (2.0 * g - r - b) / den
            k = max(3, int(round(TEX_WIN_M / self.px)) | 1)
            m1 = ndi.uniform_filter(g, k)
            tex = ndi.uniform_filter(g * g, k) - m1 * m1
            green = ndi.gaussian_filter(exg, 0.6 / self.px)
            if self.mode == "green":
                return np.where(tex > TEX_MIN, green, -1e4).astype(np.float32)
            # 'both': bright fronds AND darker than the local background. Plain
            # grass is just as green as a frond but has no local contrast, so the
            # dark response gate is what removes it.
            dark = self._dark(g, r, b)
            ok = (tex > TEX_MIN) & (dark > DARK_GATE)
            return np.where(ok, green, -1e4).astype(np.float32)
        return self._dark(g, r, b)

    def _dark(self, g, r, b):
        lum = 0.6 * g + 0.3 * r + 0.1 * b
        s1 = DOG_INNER_M / self.px
        s2 = DOG_OUTER_M / self.px
        # background on a 4x-decimated copy: same result, ~16x cheaper
        th, tw = lum.shape
        bg = ndi.zoom(ndi.gaussian_filter(lum[::4, ::4], s2 / 4.0), 4, order=1)
        bg = bg[:th, :tw]
        if bg.shape != lum.shape:
            bg = ndi.zoom(bg, (th / bg.shape[0], tw / bg.shape[1]), order=1)[:th, :tw]
        resp = bg - ndi.gaussian_filter(lum, s1)
        k = max(3, int(round(TEX_WIN_M / self.px)) | 1)
        m1 = ndi.uniform_filter(lum, k)
        tex = ndi.uniform_filter(lum * lum, k) - m1 * m1
        return np.where(tex > TEX_MIN, resp, -1e4).astype(np.float32)


class Bucket:
    """Greedy non-maximum suppression on centre candidates."""

    def __init__(self, cell):
        self.cell, self.seeds, self.grid = cell, [], {}

    def try_add(self, x, y, s):
        cx, cy = int(x // self.cell), int(y // self.cell)
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for (px_, py_, _) in self.grid.get((gx, gy), ()):
                    if (px_ - x) ** 2 + (py_ - y) ** 2 < self.cell ** 2:
                        return False
        self.grid.setdefault((cx, cy), []).append((x, y, s))
        self.seeds.append((x, y, s))
        return True


def tiles(bw, bh, size, overlap=0):
    for ty in range(0, bh, size - overlap):
        for tx in range(0, bw, size - overlap):
            yield tx, ty, min(size, bw - tx), min(size, bh - ty)


def crown_mask(resp, alpha, thr):
    mask = (resp > thr) & (alpha > 0)
    return ndi.binary_opening(ndi.binary_closing(mask, np.ones((3, 3))),
                              np.ones((3, 3)))


def read_resp(ds, tx, ty, tw, th):
    return ds.GetRasterBand(1).ReadAsArray(tx, ty, tw, th).astype(np.float32)


def refine_seeds(resp_ds, rd, bw, bh, seeds, px, thr, iters, radius_m):
    """Iterated mean-shift: move every seed to the response-weighted centroid of
    the crown pixels within radius_m. Shadow / understory pixels carry a low
    response, so they pull the point only weakly; the palm fronds dominate."""
    R = radius_m / px
    shifts = []
    sw = np.ones(len(seeds))
    for _ in range(iters):
        tree = cKDTree(seeds)
        sw = np.zeros(len(seeds))
        sx = np.zeros(len(seeds))
        sy = np.zeros(len(seeds))
        for tx, ty, tw, th in tiles(bw, bh, TILE):
            resp = read_resp(resp_ds, tx, ty, tw, th)
            m = crown_mask(resp, rd.read(tx, ty, tw, th, (4,))[0], thr)
            yy, xx = np.nonzero(m)
            if not yy.size:
                continue
            pts = np.c_[xx + tx + 0.5, yy + ty + 0.5]
            dist, idx = tree.query(pts, k=1, distance_upper_bound=R)
            ok = np.isfinite(dist)
            if not ok.any():
                continue
            ii = idx[ok].astype(np.int64)
            w = np.clip(resp[yy[ok], xx[ok]].astype(np.float64), 0, None)
            np.add.at(sw, ii, w)
            np.add.at(sx, ii, w * pts[ok, 0])
            np.add.at(sy, ii, w * pts[ok, 1])
        good = sw > 1e-6
        if not good.any():
            break
        moved = seeds.copy()
        moved[good, 0] = sx[good] / sw[good]
        moved[good, 1] = sy[good] / sw[good]
        step = np.hypot(moved[:, 0] - seeds[:, 0], moved[:, 1] - seeds[:, 1]) * px
        shifts.append(float(step.mean()))
        seeds = moved
    return seeds, shifts, sw


def spoke_scores(ortho, pts_full):
    """Frond-spoke score for every centre, sampled on the full-resolution image.

    A palm is a rosette: at 2-3 m from its centre the brightness swings a lot as
    the ring crosses bright fronds and dark gaps between them. Bare ground, grass,
    a shadow or a road edge is flat at that radius, so its angular variation is
    low. Score = mean coefficient of variation around those rings. It is read as
    a local rank (see `spoke_rank`) because its absolute value follows the
    lighting of the block."""
    ds = gdal.Open(ortho)
    arr = ds.GetRasterBand(2).ReadAsArray()
    out = spoke_from_array(arr, pts_full)
    ds = None
    return out


def spoke_from_array(arr, pts_full):
    """Spoke score over an already-loaded green band (points in full-res px)."""
    H, W = arr.shape
    radii = np.array(SPOKE_RADII_M) / SPOKE_PX
    ang = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    dy = np.outer(radii, np.sin(ang))
    dx = np.outer(radii, np.cos(ang))
    R = int(np.ceil(radii.max())) + 1
    out = np.zeros(len(pts_full))
    for i, (x, y) in enumerate(pts_full):
        cx, cy = int(round(x)), int(round(y))
        if cx < R or cy < R or cx >= W - R or cy >= H - R:
            continue
        sub = arr[cy - R:cy + R + 1, cx - R:cx + R + 1].astype(np.float32)
        yy = np.clip((dy + R).astype(int), 0, sub.shape[0] - 1)
        xx = np.clip((dx + R).astype(int), 0, sub.shape[1] - 1)
        vals = sub[yy, xx]
        cv = vals.std(axis=1) / np.maximum(vals.mean(axis=1), 1.0)
        out[i] = float(cv.mean())
    return out


def qa_png(rd, bw, bh, lab, seeds, px, out_png, zoom):
    """Crown outlines in red, palm centres as a yellow cross inside a circle of
    the mean crown radius - so it is obvious whether a point is centred."""
    rgb = np.zeros((bh, bw, 3), dtype=np.uint8)
    for i in range(3):
        rgb[:, :, i] = rd.read(0, 0, bw, bh, (i + 1,))[0]
    edge = (ndi.maximum_filter(lab, 3) != ndi.minimum_filter(lab, 3)) & (lab > 0)
    rgb[edge] = [255, 0, 0]
    res = max(0.6, float(np.sqrt(np.median(np.bincount(lab.ravel())[1:]) / np.pi)) * px)
    r_px = res / px
    yy, xx = np.mgrid[-int(r_px) - 2:int(r_px) + 3, -int(r_px) - 2:int(r_px) + 3]
    ring = np.abs(np.hypot(yy, xx) - r_px) < 0.7
    cross = (np.abs(yy) <= 1) | (np.abs(xx) <= 1)
    for x, y in seeds:
        cy, cx = int(round(y)), int(round(x))
        y0, y1 = cy + yy.min(), cy + yy.max() + 1
        x0, x1 = cx + xx.min(), cx + xx.max() + 1
        if y0 < 0 or x0 < 0 or y1 > bh or x1 > bw:
            continue
        win = rgb[y0:y1, x0:x1]
        win[ring] = [0, 255, 255]
        win[cross] = [255, 255, 0]
    o = gdal.GetDriverByName("MEM").Create("", bw, bh, 3, gdal.GDT_Byte)
    for i in range(3):
        o.GetRasterBand(i + 1).WriteArray(rgb[:, :, i])
    gdal.Translate(out_png, o, options=f"-of PNG -outsize {int(bw*zoom)} {int(bh*zoom)} -r nearest")
    print("QA overlay ->", out_png)


# --------------------------------------------------------------- vector output
def _srs(wkt):
    srs = osr.SpatialReference(wkt=wkt)
    try:
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    except Exception:
        pass
    return srs


def write_crown_polygons(lab_path, out_path, wkt, crowns):
    """Polygonise the label raster, then dissolve to one polygon per palm."""
    srs = _srs(wkt)
    src = gdal.Open(lab_path)
    band = src.GetRasterBand(1)
    tmp = ogr.GetDriverByName("Memory").CreateDataSource("poly")
    raw = tmp.CreateLayer("raw", srs, ogr.wkbPolygon)
    raw.CreateField(ogr.FieldDefn("crown_id", ogr.OFTInteger))
    gdal.Polygonize(band, band, raw, 0, [], callback=None)
    n_raw = raw.GetFeatureCount()
    src = None
    if n_raw == 0:
        raise RuntimeError(f"polygonise found no features in {lab_path}")

    valid = {c["id"] for c in crowns} if crowns else None
    groups = {}
    raw.ResetReading()
    for f in raw:
        g = f.GetGeometryRef()
        cid = int(f.GetField("crown_id"))
        if g is not None and (valid is None or cid in valid):
            groups.setdefault(cid, []).append(g.Clone())

    if os.path.exists(out_path):
        os.remove(out_path)
    gpkg = ogr.GetDriverByName("GPKG").CreateDataSource(out_path)
    lyr = gpkg.CreateLayer("palm_crowns", srs, ogr.wkbMultiPolygon)
    lyr.CreateField(ogr.FieldDefn("crown_id", ogr.OFTInteger))
    lyr.CreateField(ogr.FieldDefn("area_m2", ogr.OFTReal))
    lyr.CreateField(ogr.FieldDefn("diam_m", ogr.OFTReal))
    defn = lyr.GetLayerDefn()
    lyr.StartTransaction()
    for cid, parts in groups.items():
        geom = parts[0]
        for extra in parts[1:]:
            geom = geom.Union(extra)
        # smooth the staircase left by the raster boundary, keep the topology
        geom = geom.SimplifyPreserveTopology(SIMPLIFY_M)
        if geom.GetGeometryType() == ogr.wkbPolygon:
            geom = ogr.ForceToMultiPolygon(geom)
        a = geom.GetArea()
        f = ogr.Feature(defn)
        f.SetField("crown_id", cid)
        f.SetField("area_m2", round(a, 2))
        f.SetField("diam_m", round(2 * np.sqrt(a / np.pi), 2))
        f.SetGeometry(geom)
        lyr.CreateFeature(f)
    lyr.CommitTransaction()
    print(f"wrote {out_path}: {n_raw} raw pieces dissolved into "
          f"{lyr.GetFeatureCount()} crown polygons"
          + (f" ({n_raw - sum(len(v) for v in groups.values())} pieces outside the palm table skipped)"
             if valid is not None else ""))
    gpkg = lyr = tmp = raw = None


def write_palm_points(out_path, wkt, crowns):
    if os.path.exists(out_path):
        os.remove(out_path)
    gpkg = ogr.GetDriverByName("GPKG").CreateDataSource(out_path)
    lyr = gpkg.CreateLayer("palm_centres", _srs(wkt), ogr.wkbPoint)
    for fld, typ in (("crown_id", ogr.OFTInteger), ("area_m2", ogr.OFTReal),
                     ("diam_m", ogr.OFTReal)):
        lyr.CreateField(ogr.FieldDefn(fld, typ))
    defn = lyr.GetLayerDefn()
    lyr.StartTransaction()
    for c in crowns:
        f = ogr.Feature(defn)
        f.SetField("crown_id", c["id"])
        f.SetField("area_m2", c["crown_area_m2"])
        f.SetField("diam_m", c["equiv_diam_m"])
        g = ogr.Geometry(ogr.wkbPoint)
        g.AddPoint_2D(c["map_x"], c["map_y"])
        f.SetGeometry(g)
        lyr.CreateFeature(f)
    lyr.CommitTransaction()
    print(f"wrote {out_path}: {lyr.GetFeatureCount()} palm centres")
    gpkg = lyr = None


def load_crowns(path):
    out = []
    with open(path) as f:
        for row in csvmod.DictReader(f):
            out.append(dict(id=int(row["crown_id"]), map_x=float(row["map_x"]),
                            map_y=float(row["map_y"]),
                            crown_area_m2=float(row["crown_area_m2"]),
                            equiv_diam_m=float(row["equiv_diam_m"])))
    return out


def vector_only(args):
    lab = args.labels or os.path.join(args.outdir, "_crown_labels.tif")
    if not os.path.exists(lab):
        sys.exit(f"label raster not found: {lab}")
    ds = gdal.Open(args.ortho)
    wkt = ds.GetProjection()
    ds = None
    crowns = load_crowns(os.path.join(args.outdir, "palm_canopy_digitasi.csv"))
    write_crown_polygons(lab, os.path.join(args.outdir, "output/palm_crowns.gpkg"), wkt, crowns)
    write_palm_points(os.path.join(args.outdir, "output/palm_centres.gpkg"), wkt, crowns)
    if not args.keep_labels:
        os.remove(lab)


# --------------------------------------------------------------------- pipeline
def main():
    global DARK_GATE, SPOKE_PX
    ap = argparse.ArgumentParser()
    ap.add_argument("--ortho", default="ORTO271026.tif")
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--crop", nargs=4, type=int, default=None,
                    help="x0 y0 x1 y1 in FULL-RESOLUTION pixels")
    ap.add_argument("--crown-thr", type=float, default=CROWN_THR)
    ap.add_argument("--min-sep", type=float, default=MIN_SEP_M)
    ap.add_argument("--dt-min-m", type=float, default=DT_MIN_M)
    ap.add_argument("--resp", choices=("dark", "green", "both"), default="both",
                    help="crown response: dark rosettes, bright fronds, or both "
                         "(fronds must also be darker than their background)")
    ap.add_argument("--dark-gate", type=float, default=DARK_GATE)
    ap.add_argument("--min-spoke", type=float, default=SPOKE_MIN,
                    help="drop centres whose frond-spoke score is below this "
                         "absolute value; 0 disables")
    ap.add_argument("--spoke-rank", type=float, default=SPOKE_RANK,
                    help="drop centres that are not more crown-like than this "
                         "fraction of the ground around them (0.75 recommended)")
    ap.add_argument("--spoke-lattice", type=float, default=SPOKE_LATTICE_M,
                    help="spacing of the background reference lattice, metres")
    ap.add_argument("--min-crown-frac", type=float, default=MIN_CROWN_FRAC,
                    help="drop a centre if less than this fraction of the 2.5 m "
                         "disc around it is crown (0 = off; 0.2 recommended)")
    ap.add_argument("--min-dt-centre", type=float, default=MIN_DT_CENTRE,
                    help="drop a centre if the crown there is thinner than this "
                         "many metres (0 = off; 0.6 recommended)")
    ap.add_argument("--peak-mode", choices=("resp", "dt"), default="resp",
                    help="resp = per-crown maxima of the smoothed crown response "
                         "(one seed per crown even when crowns touch); dt = peaks "
                         "of the mask distance transform")
    ap.add_argument("--peak-thr", type=float, default=PEAK_THR)
    ap.add_argument("--smooth-m", type=float, default=SMOOTH_M,
                    help="response smoothing for per-crown maxima; small values "
                         "over-seed, which the centring pass then collapses")
    ap.add_argument("--peak-win-m", type=float, default=PEAK_WIN_M,
                    help="seed non-maximum window")
    ap.add_argument("--refine-iters", type=int, default=REFINE_ITERS)
    ap.add_argument("--refine-radius", type=float, default=REFINE_RADIUS_M)
    ap.add_argument("--tune", action="store_true", help="write qa_overlay.png")
    ap.add_argument("--zoom", type=float, default=None,
                    help="QA PNG scale (default 1 for the whole ortho, 4 for crops)")
    ap.add_argument("--keep-resp", action="store_true",
                    help="keep the cached crown-response raster")
    ap.add_argument("--vector-only", action="store_true",
                    help="build the polygons/points from an existing label raster")
    ap.add_argument("--labels", default=None, help="label raster for --vector-only")
    ap.add_argument("--keep-labels", action="store_true")
    args = ap.parse_args()

    if args.vector_only:
        return vector_only(args)

    DARK_GATE = args.dark_gate

    ds = gdal.Open(args.ortho)
    W, H = ds.RasterXSize, ds.RasterYSize
    ds = None
    if args.crop:
        x0 = max(0, args.crop[0] // 2); y0 = max(0, args.crop[1] // 2)
        x1 = min(W >> OVERVIEW, args.crop[2] // 2); y1 = min(H >> OVERVIEW, args.crop[3] // 2)
    else:
        x0, y0, x1, y1 = 0, 0, W >> OVERVIEW, H >> OVERVIEW
    rd = Reader(args.ortho, x0, y0, x1, y1, args.resp)
    px = rd.px
    bw, bh = x1 - x0, y1 - y0
    os.makedirs(args.outdir, exist_ok=True)
    print(f"ortho {W}x{H}, analysing {bw}x{bh} px at {px:.4f} m/px "
          f"= {bw*px:.0f} x {bh*px:.0f} m")

    gdrv = gdal.GetDriverByName("GTiff")
    opts = ["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"]
    resp_path = os.path.join(args.outdir, "_crown_response.tif")
    mask_path = os.path.join(args.outdir, "palm_crown_mask.tif")
    lab_path = os.path.join(args.outdir, "_crown_labels.tif")
    for p in (resp_path, mask_path, lab_path):
        if os.path.exists(p):
            os.remove(p)

    # ---- pass A: crown response cache + centre candidates ------------------
    rds = gdrv.Create(resp_path, bw, bh, 1, gdal.GDT_Float32, options=opts)
    rds.SetGeoTransform(rd.gt_o); rds.SetProjection(rd.wkt)
    cand = []
    win = max(3, int(round(args.peak_win_m / px)) | 1)
    for tx, ty, tw, th in tiles(bw, bh, TILE, OVERLAP):
        a = rd.read(tx, ty, tw, th, (4,))[0]
        resp = rd.response(tx, ty, tw, th)
        rds.GetRasterBand(1).WriteArray(resp, tx, ty)
        if a is None or a.max() == 0:
            continue
        mask = crown_mask(resp, a, args.crown_thr)
        if not mask.any():
            continue
        if args.peak_mode == "resp":
            sm = ndi.gaussian_filter(np.clip(resp, 0, None), args.smooth_m / px)
            score = sm
            peaks = (sm >= ndi.maximum_filter(sm, size=win)) & \
                    (sm > args.peak_thr) & mask & (a > 0)
        else:
            dt = ndi.distance_transform_edt(mask)
            score = dt
            peaks = (ndi.maximum_filter(dt, size=win) <= dt) & \
                    (dt > args.dt_min_m / px) & (a > 0)
        lab, nl = ndi.label(peaks)
        if not nl:
            continue
        cen = np.array(ndi.center_of_mass(score, lab, range(1, nl + 1))).reshape(-1, 2)
        for (yy, xx), sc in zip(cen, ndi.maximum(score, lab, range(1, nl + 1))):
            cand.append((tx + xx, ty + yy, float(sc)))
    rds = None
    gc.collect()
    print(f"centre candidates: {len(cand)}")
    if not cand:
        sys.exit("no crown candidates - lower --crown-thr")

    cand.sort(key=lambda c: -c[2])
    bk = Bucket(args.min_sep / px)
    for x, y, s in cand:
        bk.try_add(x, y, s)
    seeds = np.array([(x, y) for x, y, _ in bk.seeds], dtype=float).reshape(-1, 2)
    n_seed = len(seeds)
    print(f"palm centres after {args.min_sep} m thinning: {n_seed}")

    # ---- pass A2: settle every seed onto its crown centre ------------------
    rds = gdal.Open(resp_path)
    seeds, shifts, mass = refine_seeds(rds, rd, bw, bh, seeds, px, args.crown_thr,
                                       args.refine_iters, args.refine_radius)
    rds = None
    gc.collect()
    print("mean seed shift per refinement pass (m): "
          + ", ".join(f"{s:.2f}" for s in shifts))

    # seeds may have converged onto the same crown - thin them again, keeping
    # the seed that carries more crown response (the better-substantiated palm)
    bk2 = Bucket(args.min_sep / px)
    for i in np.argsort(-mass):
        bk2.try_add(seeds[i, 0], seeds[i, 1], float(mass[i]))
    seeds = np.array([(x, y) for x, y, _ in bk2.seeds], dtype=float).reshape(-1, 2)
    print(f"palm centres after refinement + {args.min_sep} m dedupe: {len(seeds)}")
    if not len(seeds):
        sys.exit("all seeds collapsed - raise --min-sep or lower --crown-thr")

    # ---- pass A3: frond-spoke validation ---------------------------------
    SPOKE_PX = rd.px / (2 ** OVERVIEW)
    pts_full = (seeds + np.array([x0, y0])) * (2 ** OVERVIEW)
    ods = gdal.Open(args.ortho)
    arr = ods.GetRasterBand(2).ReadAsArray()
    ods = None
    spoke = spoke_from_array(arr, pts_full)
    # background reference: a lattice over the analysed window. The score is read
    # as a rank against it, because its absolute level follows the block lighting.
    step = max(2.0, args.spoke_lattice / rd.px)
    gy, gx = np.mgrid[0.5:bh:step, 0.5:bw:step]
    lat = (np.c_[gx.ravel(), gy.ravel()] + np.array([x0, y0])) * (2 ** OVERVIEW)
    lat_s = spoke_from_array(arr, lat)
    arr = None
    kk = max(1, min(200, len(lat_s)))
    _, ii = cKDTree(lat).query(pts_full, k=kk)
    rank = (lat_s[np.atleast_2d(ii)] < spoke[:, None]).mean(axis=1)
    pct = lambda v: ", ".join(f"{q:.2f}" for q in np.percentile(v, [10, 50, 90]))
    print("spoke score p10/p50/p90: " + pct(spoke))
    print("spoke rank  p10/p50/p90: " + pct(rank))
    if args.min_spoke > 0:
        ok = spoke >= args.min_spoke
        print(f"spoke score >= {args.min_spoke}: kept {int(ok.sum())} of {len(seeds)}"
              f" centres")
        seeds, spoke, rank = seeds[ok], spoke[ok], rank[ok]
    if args.spoke_rank > 0:
        ok = rank >= args.spoke_rank
        print(f"spoke rank >= {args.spoke_rank}: kept {int(ok.sum())} of {len(seeds)}"
              f" centres")
        seeds, spoke, rank = seeds[ok], spoke[ok], rank[ok]
    if not len(seeds):
        sys.exit("spoke filters removed every centre - lower --min-spoke / --spoke-rank")

    # ---- pass B: mask + nearest-centre crown delineation -------------------
    mds = gdrv.Create(mask_path, bw, bh, 1, gdal.GDT_Byte,
                      options=["TILED=YES", "COMPRESS=DEFLATE",
                               "SPARSE_OK=TRUE", "BIGTIFF=IF_SAFER"])
    lds = gdrv.Create(lab_path, bw, bh, 1, gdal.GDT_Int32,
                      options=["TILED=YES", "COMPRESS=DEFLATE",
                               "SPARSE_OK=TRUE", "BIGTIFF=IF_SAFER"])
    for d in (mds, lds):
        d.SetGeoTransform(rd.gt_o); d.SetProjection(rd.wkt)

    rds = gdal.Open(resp_path)
    tree = cKDTree(seeds)
    npalm = len(seeds)
    area_px = np.zeros(npalm, dtype=np.int64)
    wsum = np.zeros(npalm)
    cx_sum = np.zeros(npalm)
    cy_sum = np.zeros(npalm)
    n_mask = n_valid = 0
    # how much crown actually sits under each centre: fraction of a 2.5 m disc
    # that is crown, and the crown thickness (mask distance transform) there.
    # A point on empty ground fails both; a palm passes both.
    R_u = max(2, int(round(2.5 / px)))
    uy, ux = np.mgrid[-R_u:R_u + 1, -R_u:R_u + 1]
    udisc = (uy * uy + ux * ux) <= R_u * R_u
    crown_frac = np.ones(npalm)
    centre_dt = np.full(npalm, 9.0)
    for tx, ty, tw, th in tiles(bw, bh, TILE):
        a = rd.read(tx, ty, tw, th, (4,))[0]
        valid = a > 0
        n_valid += int(valid.sum())
        resp = read_resp(rds, tx, ty, tw, th)
        mask = crown_mask(resp, a, args.crown_thr)
        n_mask += int(mask.sum())
        mds.GetRasterBand(1).WriteArray(mask.astype(np.uint8), tx, ty)
        lab = np.zeros((th, tw), dtype=np.int32)
        yy, xx = np.nonzero(mask)
        if yy.size:
            pts = np.c_[xx + tx + 0.5, yy + ty + 0.5]
            dist, idx = tree.query(pts, k=1,
                                   distance_upper_bound=MAX_CROWN_RADIUS_M / px)
            ok = np.isfinite(dist)
            if ok.any():
                ii = idx[ok].astype(np.int64)
                lab[yy[ok], xx[ok]] = ii + 1
                w = np.clip(resp[yy[ok], xx[ok]].astype(np.float64), 0, None) + 1e-3
                np.add.at(area_px, ii, 1)
                np.add.at(wsum, ii, w)
                np.add.at(cx_sum, ii, w * pts[ok, 0])
                np.add.at(cy_sum, ii, w * pts[ok, 1])
        lds.GetRasterBand(1).WriteArray(lab, tx, ty)
        sel = np.nonzero((seeds[:, 0] >= tx) & (seeds[:, 0] < tx + tw) &
                         (seeds[:, 1] >= ty) & (seeds[:, 1] < ty + th))[0]
        if sel.size:
            dtl = ndi.distance_transform_edt(mask)
            for i in sel:
                cx = int(round(seeds[i, 0])) - tx
                cy = int(round(seeds[i, 1])) - ty
                if (cx - R_u < 0) or (cy - R_u < 0) or \
                   (cx + R_u >= tw) or (cy + R_u >= th):
                    continue                      # edge of the tile: keep it
                crown_frac[i] = float(mask[cy - R_u:cy + R_u + 1,
                                            cx - R_u:cx + R_u + 1][udisc].mean())
                centre_dt[i] = float(dtl[cy, cx]) * px
    rds = mds = lds = None
    gc.collect()

    valid_ha = n_valid * px * px / 10000.0
    print(f"canopy mask {n_mask*px*px/10000:.3f} ha = "
          f"{100.0*n_mask*px*px/10000/valid_ha:.1f} % of the footprint")

    keep = (area_px * px * px) >= MIN_CROWN_AREA_M2
    if args.min_crown_frac > 0 or args.min_dt_centre > 0:
        under = (crown_frac >= args.min_crown_frac) & (centre_dt >= args.min_dt_centre)
        print(f"crown-under-point filter: dropped {int((keep & ~under).sum())} of "
              f"{int(keep.sum())} centres with too little crown beneath them")
        keep = keep & under
    new_ids = np.nonzero(keep)[0]
    if not len(new_ids):
        sys.exit("no crowns survived - lower --min-sep / --crown-thr")
    remap = np.zeros(npalm + 1, dtype=np.int32)
    remap[new_ids + 1] = np.arange(1, len(new_ids) + 1, dtype=np.int32)

    lds = gdal.Open(lab_path, gdal.GA_Update)
    for tx, ty, tw, th in tiles(bw, bh, TILE):
        lds.GetRasterBand(1).WriteArray(
            remap[lds.GetRasterBand(1).ReadAsArray(tx, ty, tw, th)], tx, ty)
    lds = None
    gc.collect()

    # ---- centring QA: how far is a point from its own crown centroid? -----
    with np.errstate(invalid="ignore", divide="ignore"):
        ctr_x = cx_sum / np.maximum(wsum, 1e-9)
        ctr_y = cy_sum / np.maximum(wsum, 1e-9)
    off_m = np.hypot(ctr_x[new_ids] - seeds[new_ids, 0],
                     ctr_y[new_ids] - seeds[new_ids, 1]) * px

    mds2 = gdal.Open(mask_path)
    msk = mds2.GetRasterBand(1).ReadAsArray()
    mds2 = None
    pts_x = np.clip(np.round(ctr_x[new_ids]).astype(int), 0, bw - 1)
    pts_y = np.clip(np.round(ctr_y[new_ids]).astype(int), 0, bh - 1)
    inside_pct = 100.0 * float(msk[pts_y, pts_x].mean())
    msk = None

    crowns = []
    for i in new_ids:
        gxp = rd.gt_o[0] + (ctr_x[i] + 0.5) * rd.gt_o[1]
        gyp = rd.gt_o[3] + (ctr_y[i] + 0.5) * rd.gt_o[5]
        a_m2 = float(area_px[i] * px * px)
        crowns.append(dict(id=int(remap[i + 1]), map_x=gxp, map_y=gyp,
                           crown_area_m2=round(a_m2, 2),
                           equiv_diam_m=round(2 * np.sqrt(a_m2 / np.pi), 2),
                           spoke=round(float(spoke[i]), 3),
                           spoke_rank=round(float(rank[i]), 3),
                           crown_frac=round(float(crown_frac[i]), 3),
                           centre_dt=round(float(centre_dt[i]), 2)))

    csv_path = os.path.join(args.outdir, "palm_canopy_digitasi.csv")
    with open(csv_path, "w") as f:
        f.write("crown_id,map_x,map_y,crown_area_m2,equiv_diam_m,spoke,spoke_rank,"
                "crown_frac,centre_dt\n")
        for c in crowns:
            f.write(f"{c['id']},{c['map_x']:.2f},{c['map_y']:.2f},"
                    f"{c['crown_area_m2']},{c['equiv_diam_m']},{c['spoke']},"
                    f"{c['spoke_rank']},{c['crown_frac']},{c['centre_dt']}\n")

    ar = np.array([c["crown_area_m2"] for c in crowns])
    diam = 2 * np.sqrt(ar / np.pi)
    stats = dict(ortho=os.path.basename(args.ortho),
                 analysis_pixel_size_m=round(px, 4),
                 flown_footprint_ha=round(valid_ha, 3),
                 palm_count=int(len(crowns)),
                 palms_per_ha=round(len(crowns) / valid_ha, 1),
                 mean_palm_spacing_m=round(float(np.sqrt(10000.0 / (len(crowns) / valid_ha))), 2),
                 canopy_area_ha=round(float(ar.sum()) / 10000, 3),
                 canopy_cover_pct_of_footprint=round(100.0 * float(ar.sum()) / 10000 / valid_ha, 1),
                 crown_area_m2=dict(mean=round(float(ar.mean()), 2),
                                    median=round(float(np.median(ar)), 2),
                                    std=round(float(ar.std()), 2),
                                    min=round(float(ar.min()), 2),
                                    max=round(float(ar.max()), 2),
                                    p10=round(float(np.percentile(ar, 10)), 2),
                                    p90=round(float(np.percentile(ar, 90)), 2)),
                 crown_diameter_m=dict(mean=round(float(diam.mean()), 2),
                                       median=round(float(np.median(diam)), 2),
                                       min=round(float(diam.min()), 2),
                                       max=round(float(diam.max()), 2)),
                 centre_refinement=dict(iters=args.refine_iters,
                                        radius_m=args.refine_radius,
                                        mean_shift_final_pass_m=round(float(shifts[-1]), 2) if shifts else None,
                                        mean_offset_to_crown_centroid_m=round(float(off_m.mean()), 2),
                                        p90_offset_to_crown_centroid_m=round(float(np.percentile(off_m, 90)), 2),
                                        max_offset_to_crown_centroid_m=round(float(off_m.max()), 2),
                                        centres_inside_crown_pct=round(inside_pct, 1)),
                 crown_threshold=args.crown_thr,
                 min_separation_m=args.min_sep,
                 spoke=dict(min_score=args.min_spoke,
                            min_rank=args.spoke_rank,
                            score_p10=round(float(np.percentile(spoke, 10)), 3),
                            score_median=round(float(np.median(spoke)), 3),
                            score_p90=round(float(np.percentile(spoke, 90)), 3),
                            rank_p10=round(float(np.percentile(rank, 10)), 3),
                            rank_median=round(float(np.median(rank)), 3)),
                 crown_under_point=dict(min_crown_frac=args.min_crown_frac,
                                        min_centre_thickness_m=args.min_dt_centre,
                                        frac_p10=round(float(np.percentile(crown_frac[new_ids], 10)), 3),
                                        frac_median=round(float(np.median(crown_frac[new_ids])), 3),
                                        thickness_median_m=round(float(np.median(centre_dt[new_ids])), 2)))
    with open(os.path.join(args.outdir, "palm_canopy_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))

    if args.tune:
        labds = gdal.Open(lab_path)
        lab = labds.GetRasterBand(1).ReadAsArray()
        labds = None
        allpts = np.c_[ctr_x, ctr_y][keep]
        zoom = args.zoom or (4.0 if args.crop else 1.0)
        qa_png(rd, bw, bh, lab, allpts, px,
               os.path.join(args.outdir, "qa_overlay.png"), zoom)

    if not args.keep_resp:
        if os.path.exists(resp_path):
            os.remove(resp_path)

    print("\nnext: build the vector layers in a fresh process\n"
          f"  python3 {os.path.basename(__file__)} --vector-only --outdir {args.outdir}")


if __name__ == "__main__":
    main()
