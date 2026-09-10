"""
generate_multiangle_sets.py

Generates synthetic "photos" of lunar terrain at different Sun incidence angles
from a single DEM (elevation) tile, using hillshading. This sidesteps hunting
for real repeat NAC imagery entirely: one DEM download gives you unlimited
angle variants of the exact same terrain, which is what you actually need to
stress-test illumination invariance in your matcher.

Requires:
    pip install rasterio numpy pillow --break-system-packages

Get a DEM tile first (pick ONE):
  - LOLA 118m/px global DEM (smallest, easiest to start with):
      https://astrogeology.usgs.gov/search/map/moon_lro_lola_dem_118m
      Download the full GeoTIFF, or use a GIS tool to clip a small region
      (a few hundred km across is plenty) before using it here to keep file
      size manageable.
  - SLDEM2015 (LOLA+Kaguya merged, higher resolution ~512 px/degree):
      https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/sldem2015/
      Pick one regional tile (these are gridded by lon/lat range in the filename).

Usage:
    # One set (one location), 4 incidence angles, from a DEM you already have:
    python generate_multiangle_sets.py --dem moon_tile.tif --outdir out/site1 \
        --altitudes 80 60 40 20 --azimuth 270

    # 4 sets automatically, tiling across the DEM in a grid of crops:
    python generate_multiangle_sets.py --dem moon_tile.tif --outdir out \
        --altitudes 80 60 40 20 --azimuth 270 --num-sets 4 --crop-size 512

Notes:
  - "altitude" here = Sun elevation above horizon in degrees.
    incidence angle = 90 - altitude (so altitude=20 means incidence=70, grazing light).
  - Same azimuth across a set = same Sun direction, only elevation changes
    (this isolates incidence-angle sensitivity, matching how LROC's own
    illumination studies are structured). Vary --azimuth too if you also want
    to test Sun-direction robustness, not just angle.
  - Output is grayscale PNG per angle, named like altXX_incYY.png, plus a
    metadata.json per set recording exact azimuth/altitude/crop bounds.
"""

import argparse
import json
import os

import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image


def hillshade(dem: np.ndarray, azimuth: float, altitude: float, z_factor: float = 1.0) -> np.ndarray:
    """Classic hillshade algorithm (same math QGIS/ArcGIS use). Returns 0..1 float array."""
    x, y = np.gradient(dem.astype(np.float64) * z_factor)
    slope = np.pi / 2.0 - np.arctan(np.hypot(x, y))
    aspect = np.arctan2(-x, y)

    az_rad = np.radians(360.0 - azimuth)
    alt_rad = np.radians(altitude)

    shaded = (np.sin(alt_rad) * np.sin(slope) +
              np.cos(alt_rad) * np.cos(slope) * np.cos(az_rad - aspect))
    return np.clip(shaded, 0, 1)


def to_uint8_png(arr01: np.ndarray) -> Image.Image:
    arr = (arr01 * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def make_set(dem: np.ndarray, out_dir: str, azimuth: float, altitudes, set_name: str,
             extra_meta: dict = None):
    os.makedirs(out_dir, exist_ok=True)
    meta = {"set_name": set_name, "azimuth": azimuth, "angles": [], **(extra_meta or {})}

    for altitude in altitudes:
        incidence = 90.0 - altitude
        shaded = hillshade(dem, azimuth=azimuth, altitude=altitude)
        img = to_uint8_png(shaded)
        fname = f"alt{int(altitude):02d}_inc{int(round(incidence)):02d}.png"
        img.save(os.path.join(out_dir, fname))
        meta["angles"].append({"altitude": altitude, "incidence": incidence, "file": fname})
        print(f"  wrote {os.path.join(out_dir, fname)}")

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem", required=True, help="Path to DEM GeoTIFF")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--altitudes", type=float, nargs="+", default=[80, 60, 40, 20],
                     help="Sun elevation angles in degrees (incidence = 90 - this)")
    ap.add_argument("--azimuth", type=float, default=270.0, help="Sun azimuth in degrees")
    ap.add_argument("--num-sets", type=int, default=1,
                     help="If >1, tiles the DEM into this many non-overlapping crops, "
                          "each becoming its own 'same location, different angle' set")
    ap.add_argument("--crop-size", type=int, default=512, help="Crop width/height in pixels")
    args = ap.parse_args()

    with rasterio.open(args.dem) as src:
        full_h, full_w = src.height, src.width

        if args.num_sets == 1:
            dem = src.read(1)
            print(f"Generating set '{os.path.basename(args.outdir)}' "
                  f"({dem.shape[1]}x{dem.shape[0]} px)")
            make_set(dem, args.outdir, args.azimuth, args.altitudes, "site1")
        else:
            cs = args.crop_size
            cols = max(1, full_w // cs)
            rows = max(1, full_h // cs)
            placed = 0
            for r in range(rows):
                for c in range(cols):
                    if placed >= args.num_sets:
                        break
                    window = Window(c * cs, r * cs, cs, cs)
                    dem = src.read(1, window=window)
                    if dem.size == 0 or np.all(dem == src.nodata):
                        continue
                    set_dir = os.path.join(args.outdir, f"site{placed+1}")
                    print(f"Generating {set_dir} (crop row={r} col={c})")
                    make_set(dem, set_dir, args.azimuth, args.altitudes, f"site{placed+1}",
                             extra_meta={"crop_row": r, "crop_col": c, "crop_size": cs})
                    placed += 1
                if placed >= args.num_sets:
                    break
            if placed < args.num_sets:
                print(f"Warning: only found {placed} valid (non-empty) crops, "
                      f"wanted {args.num_sets}. Try a smaller --crop-size or a bigger DEM.")

    print("\nDone.")


if __name__ == "__main__":
    main()