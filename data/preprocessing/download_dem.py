import ee
import requests
import xarray as xr
import rioxarray
import os
import argparse
import yaml
import glob

# 1. Parse configuration FIRST to avoid hardcoding
parser = argparse.ArgumentParser(description="Download and align static DEM.")
parser.add_argument("config", type=str, help="Path to the configuration YAML file.")
args = parser.parse_args()

with open(args.config, "r") as file:
    config = yaml.safe_load(file)

# 2. Initialize Earth Engine explicitly with the Project ID
ee_project = config.get("EE_PROJECT_ID")
if not ee_project:
    raise ValueError("EE_PROJECT_ID is missing from config.yaml. Earth Engine requires a Google Cloud Project.")

try:
    ee.Initialize(project=ee_project)
except Exception as e:
    print(f"Failed to initialize Earth Engine: {e}")
    raise

# 3. Dynamically locate a valid OPERA reference folder
raw_dir = config["RAW_OPERA_DATA_DIR"]
zarr_folders = sorted(glob.glob(os.path.join(raw_dir, "[0-9]" * 8)))
if not zarr_folders:
    raise FileNotFoundError(f"No daily Zarr folders found in {raw_dir} to use as a spatial reference.")

opera_zarr_path = zarr_folders[0]
print(f"Using {opera_zarr_path} as spatial reference...")

ds_opera = xr.open_zarr(opera_zarr_path, consolidated=True)

# 4. Extract coordinates and projection details
x_coords = ds_opera.x.values
y_coords = ds_opera.y.values
epsg_code = "EPSG:3035" # Verify this matches OPERA's actual CRS

opera_geom = ee.Geometry.Rectangle([
    float(x_coords.min()), float(y_coords.min()), 
    float(x_coords.max()), float(y_coords.max())
], proj=epsg_code, evenOdd=False)

# 5. Query Copernicus DEM and resample from pyramids
print("Querying Copernicus DEM from Google Servers...")
dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").select('DEM').mosaic()

# Bypass the 2^31 pixel limit by relying on GEE's internal pyramids
# rather than forcing a base-layer conservative reduction.
dem_opera = dem.resample('bilinear').reproject(
    crs=epsg_code,
    scale=2000.0
)

# 6. Download the GeoTIFF
print("Requesting download URL from Earth Engine...")
url = dem_opera.getDownloadURL({
    'region': opera_geom,
    'scale': 2000.0,
    'crs': epsg_code,
    'format': 'GEO_TIFF'
})

response = requests.get(url)

# Strictly validate the HTTP response
if response.status_code != 200:
    raise RuntimeError(f"Earth Engine API Error {response.status_code}: {response.text}")

tiff_path = "temp_elevation.tif"
with open(tiff_path, 'wb') as f:
    f.write(response.content)

# 7. Strictly align to OPERA grid and save
print("Aligning raster to OPERA grid...")
da_elev = rioxarray.open_rasterio(tiff_path).squeeze("band", drop=True)
da_elev = da_elev.rename({'x': 'x', 'y': 'y'})

da_aligned = da_elev.interp(x=x_coords, y=y_coords, method="nearest")
output_path = config["STATIC_DEM_PATH"].replace(".nc", ".zarr")
da_aligned.name = "elevation"
da_aligned.to_dataset().to_zarr(output_path, mode="w")
os.remove(tiff_path)
print(f"Static elevation saved to {output_path}")