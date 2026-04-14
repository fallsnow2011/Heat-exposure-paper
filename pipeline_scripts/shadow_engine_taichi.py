import argparse
import taichi as ti
import numpy as np
import rasterio
from rasterio.transform import xy
import pysolar.solar as solar
import datetime
import os
import math
from tqdm import tqdm
from pyproj import Transformer
import pytz

# Initialize Taichi
try:
    ti.init(arch=ti.cuda)
except:
    ti.init(arch=ti.cpu)

@ti.func
def interpolate_dsm(dsm: ti.template(), x: float, y: float, w: int, h: int) -> float:
    """
    Bilinear interpolation with boundary checks and NoData handling.

    Returns:
    --------
    height : float
        Interpolated height, or -9999.0 if any corner is NoData or out of bounds
    """
    # Clamp coordinates to valid range [0, w-1] and [0, h-1]
    x_clamped = ti.max(0.0, ti.min(x, float(w - 1)))
    y_clamped = ti.max(0.0, ti.min(y, float(h - 1)))

    x0 = int(x_clamped)
    y0 = int(y_clamped)
    x1 = ti.min(x0 + 1, w - 1)
    y1 = ti.min(y0 + 1, h - 1)

    # Read corner heights
    h00 = dsm[x0, y0]
    h10 = dsm[x1, y0]
    h01 = dsm[x0, y1]
    h11 = dsm[x1, y1]

    # Use result variable instead of early return (Taichi requirement)
    result = -9999.0

    # If ANY corner is NoData, keep NoData result
    # Otherwise compute bilinear interpolation
    if h00 >= -1000.0 and h10 >= -1000.0 and h01 >= -1000.0 and h11 >= -1000.0:
        # Bilinear interpolation
        dx = x_clamped - float(x0)
        dy = y_clamped - float(y0)
        result = h00 * (1 - dx) * (1 - dy) + h10 * dx * (1 - dy) + h01 * (1 - dx) * dy + h11 * dx * dy

    return result

@ti.kernel
def compute_shadow_kernel(
    dsm: ti.template(),
    shadow_mask: ti.template(), # output: 1=Lit, 0=Shadow, 255=NoData
    width: int,
    height: int,
    res_x: float,
    res_y: float,
    sun_vec_x: float,
    sun_vec_y: float,
    sun_vec_z: float,
    max_ray_dist: float
):
    for i, j in shadow_mask:
        start_x = float(i)
        start_y = float(j)
        start_z = dsm[i, j]
        
        # Check NoData in DSM (Assuming -9999.0 is sentinel)
        if start_z < -1000.0: 
            shadow_mask[i, j] = 255
            continue

        xy_len = ti.sqrt(sun_vec_x**2 + sun_vec_y**2)
        
        if xy_len < 1e-6: # Zenith
            shadow_mask[i, j] = 1
            continue
            
        step_size = 1.0
        dx = (sun_vec_x / xy_len) * step_size
        dy = (sun_vec_y / xy_len) * step_size
        pixel_size = (res_x + res_y) / 2.0
        dz = (sun_vec_z / xy_len) * pixel_size 

        is_shadowed = 0
        curr_x = start_x + dx
        curr_y = start_y + dy
        curr_z = start_z + dz
        dist_traveled = 0.0
        
        while (curr_x >= 0 and curr_x < width and
               curr_y >= 0 and curr_y < height and
               dist_traveled < max_ray_dist):

            terrain_z = interpolate_dsm(dsm, curr_x, curr_y, width, height)

            # If interpolation hits NoData region, stop ray (assume no occlusion)
            if terrain_z < -1000.0:
                break

            if terrain_z > curr_z - 0.1:
                is_shadowed = 1
                break

            curr_x += dx
            curr_y += dy
            curr_z += dz
            dist_traveled += pixel_size
            
        if is_shadowed == 1:
            shadow_mask[i, j] = 0
        else:
            shadow_mask[i, j] = 1

class ShadowCalculator:
    def __init__(self, dsm_path):
        self.dsm_path = dsm_path
        self._load_dsm()
        self._init_taichi_fields()

    def _load_dsm(self):
        print(f"Loading DSM from {self.dsm_path}...")
        with rasterio.open(self.dsm_path) as src:
            self.dsm_data = src.read(1).astype(np.float32)
            self.profile = src.profile
            self.transform = src.transform
            self.crs = src.crs
            self.res_x = src.res[0]
            self.res_y = src.res[1]
            self.height, self.width = self.dsm_data.shape
            
            # Handle nodata: set to -9999.0 for internal kernel check
            if src.nodata is not None:
                self.dsm_data[self.dsm_data == src.nodata] = -9999.0
            else:
                self.dsm_data[self.dsm_data < -1000] = -9999.0
        
        # Calculate Center Lat/Lon dynamically
        cx, cy = self.width // 2, self.height // 2
        wx, wy = rasterio.transform.xy(self.transform, cy, cx)
        
        if self.crs.is_projected:
            transformer = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
            self.lon, self.lat = transformer.transform(wx, wy)
        else:
            self.lon, self.lat = wx, wy
            
        print(f"DSM Center: Lat {self.lat:.4f}, Lon {self.lon:.4f}")
        print(f"Size: {self.width}x{self.height}, Res: {self.res_x}m")

    def _init_taichi_fields(self):
        print("Initializing Taichi fields...")
        self.ti_dsm = ti.field(dtype=ti.f32, shape=(self.width, self.height))
        self.ti_shadow = ti.field(dtype=ti.u8, shape=(self.width, self.height)) # Using u8
        self.ti_dsm.from_numpy(self.dsm_data.T)

    def calculate_shadow(self, dt: datetime.datetime, max_ray_dist_meters=500.0):
        altitude_deg = solar.get_altitude(self.lat, self.lon, dt)
        azimuth_deg = solar.get_azimuth(self.lat, self.lon, dt)
        
        print(f"Time: {dt}, Sun Alt: {altitude_deg:.2f}, Az: {azimuth_deg:.2f}")
        
        # Nighttime / Twilight Check
        if altitude_deg <= 5.0: # Filter low angles too
            print("Sun too low/set. Returning NoData (255).")
            # Or return 0 if you want strict night=shadow, but better separate.
            return np.full((self.height, self.width), 255, dtype=np.uint8)

        theta = math.radians(90 - azimuth_deg)
        phi = math.radians(altitude_deg)
        sun_x = math.cos(phi) * math.cos(theta)
        sun_y = math.cos(phi) * math.sin(theta)
        sun_z = math.sin(phi)
        
        compute_shadow_kernel(
            self.ti_dsm, self.ti_shadow, self.width, self.height,
            self.res_x, self.res_y, sun_x, sun_y, sun_z, max_ray_dist_meters
        )
        ti.sync()
        return self.ti_shadow.to_numpy().T.astype(np.uint8)

    def save_result(self, shadow_mask, output_path):
        meta = self.profile.copy()
        meta.update({
            "dtype": "uint8", "count": 1, "nodata": 255, "compress": "lzw"
        })
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(shadow_mask, 1)
        print(f"Saved {output_path}")

def batch_process_day(
    dsm_path,
    output_dir,
    date_str="2024-06-21",
    hours=range(9, 18), # 9am-5pm
    max_ray_dist_meters=500.0,
    timezone_str="Europe/London"  # Use local timezone
):
    """
    Process shadow maps for multiple hours in a day.

    Parameters:
    -----------
    timezone_str : str
        IANA timezone string (e.g., 'Europe/London', 'America/New_York')
        IMPORTANT: Uses local time, not UTC. For London in summer (BST),
        this is UTC+1. The sun position will be calculated correctly.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    calc = ShadowCalculator(dsm_path)
    year, month, day = map(int, date_str.split("-"))

    # Use local timezone
    tz = pytz.timezone(timezone_str)

    for h in tqdm(hours, desc="Processing hours"):
        # Create timezone-aware datetime in local time
        dt_naive = datetime.datetime(year, month, day, h, 0, 0)
        dt = tz.localize(dt_naive)

        mask = calc.calculate_shadow(dt, max_ray_dist_meters=max_ray_dist_meters)
        out_path = os.path.join(output_dir, f"shadow_{date_str}_{h:02d}00.tif")
        calc.save_result(mask, out_path)

def parse_hours(spec: str):
    spec = spec.strip()
    if "-" in spec:
        try:
            start, end = spec.split("-", 1)
            start_i, end_i = int(start), int(end)
            if start_i > end_i:
                raise ValueError
            return range(start_i, end_i + 1)
        except Exception:
            raise argparse.ArgumentTypeError(f"Invalid hour range: {spec}")
    if not spec:
        raise argparse.ArgumentTypeError("Hours string is empty")
    try:
        return [int(h.strip()) for h in spec.split(",") if h.strip() != ""]
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid hour list: {spec}")

def main():
    parser = argparse.ArgumentParser(description="Taichi-based shadow casting for DSM raster.")
    parser.add_argument("--dsm", default="../london_dsm_2m.tif", help="Path to DSM raster.")
    parser.add_argument("--output-dir", default="../paper_urban_cool_network/shadow_maps",
                        dest="output_dir", help="Directory to save shadow rasters.")
    parser.add_argument("--date", default="2024-06-21", help="Date string YYYY-MM-DD.")
    parser.add_argument("--hours", default="9-17",
                        help="Hours to process, e.g., '9-17' or '9,12,15'.")
    parser.add_argument("--max-ray-dist", type=float, default=500.0,
                        dest="max_ray_dist", help="Maximum ray length in meters.")
    args = parser.parse_args()

    hours = parse_hours(args.hours)

    if not os.path.exists(args.dsm):
        print(f"DSM not found at {args.dsm}")
        return

    batch_process_day(
        args.dsm,
        args.output_dir,
        date_str=args.date,
        hours=hours,
        max_ray_dist_meters=args.max_ray_dist
    )

if __name__ == "__main__":
    main()



