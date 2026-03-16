#!/bin/bash
# Executes locally. Requires GDAL.

TARGET_PROJ="+proj=laea +lat_0=55.0 +lon_0=10.0 +x_0=1950000.0 +y_0=-2100000.0 +units=m +ellps=WGS84"  # From Ophelia Mirrales
BBOX="-10.43 31.75 57.81 67.62"  # Check (*) for code to compute 
TARGET_RES="2000"  # The metadata in the dataset is wrong.

# OpenTopography Copernicus GLO-30 public VRT
VRT_URL="/vsicurl/https://opentopography.s3.sdsc.edu/raster/COP30/COP30_hh.vrt"
OUTPUT_FILE="/home/fquareng/work/data/extremes/OPERA/europe_dem_laea.tif"

echo "Initiating retrieval and reprojection..."
gdalwarp -t_srs "$TARGET_PROJ" \
         -te_srs EPSG:4326 -te $BBOX \
         -tr $TARGET_RES $TARGET_RES \
         -r bilinear \
         -wm 2048 \
         -multi \
         -co COMPRESS=DEFLATE \
         -co TILED=YES \
         "$VRT_URL" "$OUTPUT_FILE"

echo "Process complete: $OUTPUT_FILE"

###### Notes
# ```Python:
# >>> from pyproj import Proj, Transformer
# >>> target_proj = "+proj=laea +lat_0=55.0 +lon_0=10.0 +x_0=1950000.0 +y_0=-2100000.0 +units=m +ellps=WGS84"
# >>> xmin, ymin = -0.000245, -4400000.001
# >>> xmax, ymax = 3800000.0, -0.000426
# >>> transformer = Transformer.from_proj(target_proj, "EPSG:4326", always_xy=True)
# >>> lon_min, lat_min = transformer.transform(xmin, ymin)
# >>> lon_max, lat_max = transformer.transform(xmax, ymax)
# >>> print(f"BBOX (Degrees): {lon_min:.2f} {lat_min:.2f} {lon_max:.2f} {lat_max:.2f}")
# BBOX (Degrees): -10.43 31.75 57.81 67.62
# ```