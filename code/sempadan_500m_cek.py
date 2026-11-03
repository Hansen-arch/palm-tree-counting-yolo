#!/usr/bin/env python3
"""Sempadan 500 m yang SEBENARNYA: buffer 500 m dari polygon sungai.

Shapefile SEMPADAN_SEGMEN_500M ternyata pita 50 m (2,5 ha per segmen 500 m),
jadi skrip ini menghitung ulang dengan sempadan 500 m dari tepi sungai,
sekaligus menyimpan polygon hasil buffer untuk dipakai di QGIS.

Keluaran:
  output/sempadan/sempadan_500m_dari_sungai.gpkg
  output/sempadan/pohon_dalam_sempadan_500m_asli.gpkg
  output/sempadan/ringkasan_sempadan_500m.json
"""
from __future__ import annotations

import json
import os

from osgeo import ogr

import analisis_sempadan_sungai as A

BUF = 500.0
OUT = A.OUT
BUF_SHP = os.path.join(OUT, "sempadan_500m_dari_sungai.gpkg")


def main() -> None:
    ds = ogr.Open(A.SUNGAI)
    lyr = ds.GetLayer(0)
    srs = lyr.GetSpatialRef()
    geoms = []
    for f in lyr:
        g = f.GetGeometryRef().Clone()
        if not g.IsValid():
            g = g.MakeValid()
        geoms.append(g)

    print(f"buffer {BUF} m dari {len(geoms)} segmen sungai ...", flush=True)
    buf = None
    for g in geoms:
        b = g.Buffer(BUF, 24)
        buf = b if buf is None else buf.Union(b)
    print(f"  luas sempadan 500 m: {buf.GetArea() / 1e4:.1f} ha", flush=True)

    # simpan polygon hasil buffer
    if os.path.exists(BUF_SHP):
        os.remove(BUF_SHP)
    out = ogr.GetDriverByName("GPKG").CreateDataSource(BUF_SHP)
    lyr_out = out.CreateLayer("sempadan_500m", srs, ogr.wkbPolygon)
    lyr_out.CreateField(ogr.FieldDefn("buffer_m", ogr.OFTInteger))
    fo = ogr.Feature(lyr_out.GetLayerDefn())
    fo.SetField("buffer_m", int(BUF))
    fo.SetGeometry(buf)
    lyr_out.CreateFeature(fo)
    out = None

    rows, summary = [], {"buffer_m": BUF, "orto": {}}
    for name, cfg in A.ORTHOS.items():
        pts = A.read_pts(cfg["pts"])
        hit = []
        for x, y, c in pts:
            g = A.point(x, y)
            if buf.Intersects(g):
                hit.append(([name, round(c, 3)], x, y))
        a_buf = A.masked_area_ha(cfg["grid"], BUF_SHP)
        a_tot = A.valid_area_ha(cfg["grid"])
        rows.extend(hit)
        summary["orto"][name] = {
            "pohon_total": len(pts),
            "pohon_dalam_sempadan_500m": len(hit),
            "persen": round(len(hit) / len(pts) * 100, 1) if pts else None,
            "luas_valid_total_ha": round(a_tot, 2),
            "luas_valid_sempadan_500m_ha": round(a_buf, 2),
            "kepadatan_dalam_per_ha": round(len(hit) / a_buf, 1) if a_buf else None,
            "kepadatan_luar_per_ha": round((len(pts) - len(hit)) / (a_tot - a_buf), 1)
            if a_tot - a_buf > 0 else None,
        }
        print(f"{name}: {len(hit)}/{len(pts)} pohon dalam 500 m "
              f"({len(hit)/len(pts)*100:.1f}%), luas {a_buf:.2f} ha -> "
              f"{len(hit)/a_buf if a_buf else 0:.1f} pohon/ha", flush=True)

    A.write_gpkg(os.path.join(OUT, "pohon_dalam_sempadan_500m_asli.gpkg"),
                 [(r + [None], x, y) for r, x, y in rows], srs)
    summary["total"] = {
        "pohon_semua_orto": sum(v["pohon_total"] for v in summary["orto"].values()),
        "pohon_dalam_sempadan_500m": sum(v["pohon_dalam_sempadan_500m"]
                                         for v in summary["orto"].values()),
    }
    with open(os.path.join(OUT, "ringkasan_sempadan_500m.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("TOTAL dalam sempadan 500 m:", summary["total"]["pohon_dalam_sempadan_500m"])


if __name__ == "__main__":
    main()
