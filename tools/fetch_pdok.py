"""Download a real area of the Netherlands (PDOK open data) to fly the simulator over.

    python -m tools.fetch_pdok --centre 85300 446400 --size 300 --out data/nl

Fetches, for a square around --centre (Dutch RD New / EPSG:28992 coordinates, metres):
    dsm.tif, dtm.tif   AHN4 surface and ground models, 0.5 m (CC0)
    ortho.png          aerial orthophoto, 8 cm (open data, CC BY; PDOK "Beeldmateriaal Nederland")
    area.json          the bounding box and resolutions

The drone never uses these: the orthophoto is what the cameras see, the elevation model
is the shape it is painted on, and the difference DSM - DTM is the ground truth.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

WCS = ("https://service.pdok.nl/rws/ahn/wcs/v1_0?service=WCS&version=2.0.1&request=GetCoverage"
       "&format=image/tiff&subset=x({x0},{x1})&subset=y({y0},{y1})&CoverageId={cov}")
WMS = ("https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0?service=WMS&version=1.3.0&request=GetMap"
       "&layers={layer}&styles=&crs=EPSG:28992&bbox={x0},{y0},{x1},{y1}&width={w}&height={h}&format=image/jpeg")
MAX_PX = 2500  # the WMS refuses larger images
BGT = ("https://api.pdok.nl/lv/bgt/ogc/v1/collections/{coll}/items?bbox={x0},{y0},{x1},{y1}"
       "&bbox-crs=http://www.opengis.net/def/crs/EPSG/0/28992&crs=http://www.opengis.net/def/crs/EPSG/0/28992"
       "&limit=1000&f=json")
BGT_COLLECTIONS = ("pand", "waterdeel", "wegdeel", "begroeidterreindeel", "onbegroeidterreindeel")


def fetch(url: str, path: Path) -> None:
    subprocess.run(["curl", "-sS", "-f", "-o", str(path), url], check=True)


def fetch_bgt(coll: str, bbox, path: Path) -> None:
    """All features of one BGT collection in the box (following 'next' pages) as one GeoJSON file.
    BGT: Basisregistratie Grootschalige Topografie via PDOK, open data (CC0)."""
    url = BGT.format(coll=coll, x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3])
    features = []
    while url:
        page = json.loads(subprocess.run(["curl", "-sS", "-f", url], check=True, capture_output=True).stdout)
        features += page.get("features", [])
        url = next((ln["href"] for ln in page.get("links", []) if ln.get("rel") == "next"), None)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--centre", type=float, nargs=2, default=(85300.0, 446400.0), help="RD x y (default: TU Delft)")
    ap.add_argument("--size", type=float, default=300.0, help="side of the square, m")
    ap.add_argument("--ortho-layer", default="2022_orthoHR", help="WMS layer; match the AHN4 years (2020-2022)")
    ap.add_argument("--ortho-res", type=float, default=0.08)
    ap.add_argument("--out", default="data/nl")
    ap.add_argument("--bgt", action="store_true",
                    help="also fetch BGT topography (buildings, water, roads, green space) as GeoJSON, for labels")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cx, cy = args.centre
    x0, y0 = cx - args.size / 2, cy - args.size / 2
    x1, y1 = x0 + args.size, y0 + args.size
    for cov in ("dsm", "dtm"):
        fetch(WCS.format(x0=x0, x1=x1, y0=y0, y1=y1, cov=f"{cov}_05m"), out / f"{cov}.tif")

    px = int(round(args.size / args.ortho_res))
    n = math.ceil(px / MAX_PX)
    tile_m, tile_px = args.size / n, px // n
    rows = []
    for j in reversed(range(n)):           # image rows run north to south
        row = []
        for i in range(n):
            tile = out / f"_ortho_{i}_{j}.jpg"
            fetch(WMS.format(layer=args.ortho_layer, x0=x0 + i * tile_m, y0=y0 + j * tile_m,
                             x1=x0 + (i + 1) * tile_m, y1=y0 + (j + 1) * tile_m, w=tile_px, h=tile_px), tile)
            img = cv2.imread(str(tile))
            if img is None:
                raise SystemExit(f"tile {tile.name} isn't an image: {tile.read_text(errors='replace')[:300]}")
            row.append(img)
            tile.unlink()
        rows.append(np.hstack(row))
    cv2.imwrite(str(out / "ortho.png"), np.vstack(rows))
    if args.bgt:
        for coll in BGT_COLLECTIONS:
            fetch_bgt(coll, (x0, y0, x1, y1), out / f"bgt_{coll}.json")
    (out / "area.json").write_text(json.dumps({
        "crs": "EPSG:28992", "bbox": [x0, y0, x1, y1], "dem_resolution": 0.5,
        "ortho_resolution": args.size / (tile_px * n), "ortho_layer": args.ortho_layer,
        "sources": {"elevation": "AHN4 via PDOK (CC0)", "ortho": "Beeldmateriaal Nederland via PDOK (CC BY)"}},
        indent=2))
    print(f"wrote {out}/: dsm.tif, dtm.tif, ortho.png ({tile_px * n} px), area.json")


if __name__ == "__main__":
    main()
