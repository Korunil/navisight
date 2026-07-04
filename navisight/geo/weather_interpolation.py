# navisight/geo/weather_interpolation.py
import logging
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

ZERO = np.float32(0.0)
ONE = np.float32(1.0)

class RectilinearMeshInterpolator:
    """
    Production-Grade Rectilinear Weather Grid Interpolation Engine.
    Tailored for regional AIS track feature engineering (e.g., Piraeus Domain).
    Derives meteorologically accurate wind direction and speed fields directly 
    from trilinearly interpolated horizontal momentum components (UGRD, VGRD).
    
    ENGINE ARCHITECTURAL CONSTRAINTS:
      - Supports irregular grid spacing (rectilinear grids).
      - Requires strictly monotonic, unique coordinate axes.
      - Tracks wind fields using UGRD/VGRD components to avoid directional interpolation artifacts and preserve vector consistency.
    """
    def __init__(
        self, 
        df_weather: pl.DataFrame, 
        boundary_mode: str = "clamp", 
        run_mode: str = "interpolated",
        is_cyclic: bool = False
    ):
        if boundary_mode not in ("clamp", "nan", "raise"):
            raise ValueError(f"Boundary policy error: mode must be 'clamp', 'nan', or 'raise'. Found: {boundary_mode}")
        self.boundary_mode = boundary_mode
        
        if run_mode not in ("causal", "interpolated"):
            raise ValueError(f"Temporal mode error: must be 'causal' or 'interpolated'. Found: {run_mode}")
        self.run_mode = run_mode
        
        # Default to non-cyclic longitude handling since the weather grid
        # represents a regional domain rather than a global field.
        self.is_cyclic = is_cyclic

        self.required_source_columns = [
            "timestamp_", "lon", "lat", "TMP", "GUST", "PRMSL", "RH", "UGRD", "VGRD"
        ]
        
         # Dense weather feature layout stored in the interpolation cube.
        self.cube_features = [
            "TMP", "GUST", "VIS", "PRMSL", "RH", "UGRD", "VGRD"
        ]
        self._idx = {feat: i for i, feat in enumerate(self.cube_features)}
        
        # STAGE 1: Standardize incoming data column strings
        df_clean = self._normalize_to_noaa_short_codes(df_weather)
        
        for col in self.required_source_columns:
            if col not in df_clean.columns:
                raise KeyError(f"Malformed weather mesh: missing required weather column: {col}")

        # STAGE 2: Coordinate Cleanup
        # Preserve original float64 coordinate precision by avoiding coordinate rounding
        # during grid construction and lookup operations.
        df_clean = df_clean.with_columns([
            pl.col("lon").alias("lon_clean"),
            pl.col("lat").alias("lat_clean"),
            pl.col("timestamp_").cast(pl.Int64).alias("timestamp_clean")
        ])

        # STAGE 3: Deterministic Data Update Sorting Pass
        sort_keys = ["timestamp_clean", "lon_clean", "lat_clean"]
        if "ingestion_epoch" in df_clean.columns:
            sort_keys.append("ingestion_epoch")
        elif "version_id" in df_clean.columns:
            sort_keys.append("version_id")
        
        df_clean = df_clean.sort(sort_keys)
        df_unique = df_clean.unique(subset=["timestamp_clean", "lon_clean", "lat_clean"], keep="last")

        # Extract dimension coordinate arrays using exact bit widths
        base_timestamps = sorted(df_unique["timestamp_clean"].unique().to_list())
        base_longitudes = sorted(df_unique["lon_clean"].unique().to_list())
        base_latitudes  = sorted(df_unique["lat_clean"].unique().to_list())

        if len(base_timestamps) < 2 or len(base_longitudes) < 2 or len(base_latitudes) < 2:
            raise ValueError("Malformed weather mesh: grid axes dimensions must contain at least 2 points for trilinear interpolation.")

        # Extract authoritative coordinate axes directly from the source grid 
        # without coordinate quantization or resampling.
        self.timestamps = np.array(base_timestamps, dtype=np.int64)
        self.latitudes  = np.array(base_latitudes, dtype=np.float64)

        # STAGE 4: Strict Spatial Monotonicity Checks (Enables Irregular Rectilinear Spacing)
        if np.any(np.diff(self.timestamps) <= 0) or np.any(np.diff(self.latitudes) <= 0) or np.any(np.diff(base_longitudes) <= 0):
            raise ValueError("Malformed weather mesh: Grid coordinate axes contain non-monotonic steps or duplicate intervals.")

        self.lon_axis_min = float(base_longitudes[0])
        self.lon_axis_max = float(base_longitudes[-1])
        
        # STAGE 5: Conditional Global Cyclic Storage Configuration
        if self.is_cyclic:
            self.median_dlon = np.median(np.diff(base_longitudes))
            self.longitudes = np.concatenate([base_longitudes, [self.lon_axis_min + 360.0]])
            logger.info(
                f"Weather Mesh Engine: Cyclic longitude mode active. \n"
                f"Bounds: [{self.lon_axis_min:.2f}, {self.lon_axis_min + 360.0:.2f}]"
            )
        else:
            self.longitudes = np.array(base_longitudes, dtype=np.float64)
            logger.info(f"Weather Mesh Engine: Local regional mode initialized.")
            logger.info(
                f"Grid Limits: Lon[{self.lon_axis_min:.2f}, {self.lon_axis_max:.2f}], "
                f"Lat[{self.latitudes[0]:.2f}, {self.latitudes[-1]:.2f}]"
            )

        # STAGE 6: Validate that the weather grid contains every
        # expected (timestamp, lon, lat) lattice point.
        lattice_contract = (
            pl.DataFrame({"timestamp_clean": self.timestamps})
            .join(pl.DataFrame({"lon_clean": base_longitudes}), how="cross")
            .join(pl.DataFrame({"lat_clean": self.latitudes}), how="cross")
        )
        
        missing_cells = lattice_contract.join(
            df_unique.select(["timestamp_clean", "lon_clean", "lat_clean"]),
            on=["timestamp_clean", "lon_clean", "lat_clean"],
            how="anti"
        )
        if missing_cells.height > 0:
            raise ValueError(f"Input regional weather grid contract breached! Missing tracking tuples details:\n{missing_cells.head(5)}")

        # STAGE 7: Allocate dense weather cube.
        self.cube = np.full(
            (len(self.timestamps), len(self.longitudes), len(self.latitudes), len(self.cube_features)),
            fill_value=np.nan, dtype=np.float32
        )

        # STAGE 8: Load weather variables into the interpolation cube.
        self.has_vis = "VIS" in df_unique.columns
        if self.has_vis:
            df_unique = df_unique.with_columns(
                pl.when(pl.col("VIS").is_null() | pl.col("VIS").is_nan()).then(pl.lit(10000.0)).otherwise(pl.col("VIS")).alias("VIS")
            )

        self._populate_mesh_cube_vectorized(df_unique, base_longitudes)

    def _normalize_to_noaa_short_codes(self, df: pl.DataFrame) -> pl.DataFrame:
        translation_table = {
            "temperature": "TMP", "gust_factor": "GUST", "visibility": "VIS",
            "wind_speed": "WSPD", "wind_direction": "WDIRMET", "pressure": "PRMSL",
            "humidity": "RH", "zonal_wind": "UGRD", "meridional_wind": "VGRD",
            "timestamp_sec": "timestamp_"
        }
        rename_actions = {}
        for src, target in translation_table.items():
            if src in df.columns:
                if target in df.columns and src != target:
                    if df[src].equals(df[target]):
                        df = df.drop(src)
                        continue
                    else:
                        raise ValueError(f"Conflicting weather column layout variations detected: {src} and {target}")
                rename_actions[src] = target

        if rename_actions:
            df = df.rename(rename_actions)
        return df

    def _populate_mesh_cube_vectorized(self, df: pl.DataFrame, base_longitudes: np.ndarray):
        """Populate the dense interpolation cube using searchsorted-based axis indexing."""
        t_idx = np.searchsorted(self.timestamps, df["timestamp_clean"].to_numpy().astype(np.int64))
        lon_idx = np.searchsorted(base_longitudes, df["lon_clean"].to_numpy().astype(np.float64))
        lat_idx = np.searchsorted(self.latitudes, df["lat_clean"].to_numpy().astype(np.float64))
        
        fill_features = [f for f in self.cube_features if f != "VIS" or self.has_vis]
        
        feature_matrix = np.empty((df.height, len(fill_features)), dtype=np.float32)
        for i, col in enumerate(fill_features):
            feature_matrix[:, i] = df[col].to_numpy().astype(np.float32)

        if not np.isfinite(feature_matrix).all():
            raise ValueError("Weather Data Initialization Fault: Non-finite values (NaN/Inf) detected inside processed feature blocks.")

        if not self.has_vis:
            self.cube[..., self._idx["VIS"]] = 10000.0

        for src_idx, col_name in enumerate(fill_features):
            f_idx = self._idx[col_name]
            self.cube[t_idx, lon_idx, lat_idx, f_idx] = feature_matrix[:, src_idx]

        if self.is_cyclic:
            self.cube[:, -1, :, :] = self.cube[:, 0, :, :]
            
        if not np.isfinite(self.cube).all():
            raise ValueError("Weather Cube Compilation Error: Found unpopulated or missing nodes inside your mesh layouts.")

    def interpolate_slice(self, df_slice: pl.DataFrame, chunk_size: int = 50000) -> pl.DataFrame:
        """
        Executes chunked trilinear interpolation over input dataframes
        to reduce peak memory consumption for large AIS trajectory batches.
        """
        if df_slice.is_empty():
            return df_slice
            
        if df_slice.height > chunk_size:
            chunks = []
            for i in range(0, df_slice.height, chunk_size):
                df_chunk = df_slice.slice(i, chunk_size)
                chunks.append(self._interpolate_kernel(df_chunk))
            return pl.concat(chunks)
            
        return self._interpolate_kernel(df_slice)

    def _interpolate_kernel(self, df_chunk: pl.DataFrame) -> pl.DataFrame:
        num_points = len(df_chunk)
        raw_lons = df_chunk["lon_raw"].to_numpy().astype(np.float64)
        
        if self.is_cyclic:
            query_lon = ((raw_lons - self.lon_axis_min) % 360.0) + self.lon_axis_min
            query_lon = np.where(query_lon < self.longitudes[0], query_lon + 360.0, query_lon)
        else:
            query_lon = raw_lons.copy()

        query_lat = df_chunk["lat_raw"].to_numpy().astype(np.float64)
        query_time = df_chunk["timestamp_sec"].to_numpy().astype(np.int64)
        vessel_courses = df_chunk["course_raw"].to_numpy().astype(np.float32)

        out_of_mesh_mask = (
            (query_lon < self.longitudes[0]) | (query_lon > self.longitudes[-1]) |
            (query_lat < self.latitudes[0]) | (query_lat > self.latitudes[-1]) |
            (query_time < self.timestamps[0]) | (query_time > self.timestamps[-1])
        )
        
        if np.any(out_of_mesh_mask):
            if self.boundary_mode == "raise":
                raise ValueError(f"Geospatial Field Contract Breach: {np.sum(out_of_mesh_mask)} tracking samples sit outside regional weather grid limits.")
            elif self.boundary_mode == "clamp":
                query_lon = np.clip(query_lon, self.longitudes[0], self.longitudes[-1])
                query_lat = np.clip(query_lat, self.latitudes[0], self.latitudes[-1])
                query_time = np.clip(query_time, self.timestamps[0], self.timestamps[-1])

        # ── STEP 1: CAUSAL TEMPORAL LOOKUP ALIGNMENT ──────────────────────────
        if self.run_mode == "causal":
            t_idx = np.searchsorted(self.timestamps, query_time, side="right") - 1
            t_lo = np.clip(t_idx, 0, len(self.timestamps) - 1)
            t_hi = t_lo
            t_weight = np.zeros(num_points, dtype=np.float32)
        else:
            t_hi = np.searchsorted(self.timestamps, query_time, side="right")
            t_hi = np.where(query_time == self.timestamps[-1], len(self.timestamps) - 1, t_hi)
            t_hi = np.clip(t_hi, 1, len(self.timestamps) - 1)
            t_lo = t_hi - 1
            
            dt = self.timestamps[t_hi] - self.timestamps[t_lo]
            t_weight = np.divide(
                (query_time - self.timestamps[t_lo]).astype(np.float32), dt.astype(np.float32),
                out=np.zeros_like(query_time, dtype=np.float32), where=dt > 0
            )

        # ── STEP 2: SPATIAL VECTOR RECTILINEAR LOOKUPS ────────────────────────
        lon_hi = np.searchsorted(self.longitudes, query_lon, side="right")
        lon_hi = np.where(query_lon == self.longitudes[-1], len(self.longitudes) - 1, lon_hi)
        lon_hi = np.clip(lon_hi, 1, len(self.longitudes) - 1)
        lon_lo = lon_hi - 1
        
        dlon = self.longitudes[lon_hi] - self.longitudes[lon_lo]
        lon_weight = np.divide(
            (query_lon - self.longitudes[lon_lo]).astype(np.float32), dlon.astype(np.float32),
            out=np.zeros_like(query_lon, dtype=np.float32), where=dlon > 0
        )

        lat_hi = np.searchsorted(self.latitudes, query_lat, side="right")
        lat_hi = np.where(query_lat == self.latitudes[-1], len(self.latitudes) - 1, lat_hi)
        lat_hi = np.clip(lat_hi, 1, len(self.latitudes) - 1)
        lat_lo = lat_hi - 1
        
        dlat = self.latitudes[lat_hi] - self.latitudes[lat_lo]
        lat_weight = np.divide(
            (query_lat - self.latitudes[lat_lo]).astype(np.float32), dlat.astype(np.float32),
            out=np.zeros_like(query_lat, dtype=np.float32), where=dlat > 0
        )

        # ── STEP 3: TRILINEAR INTERPOLATION OVER LOCAL SPATIO-TEMPORAL CELL ──
        c000 = self.cube[t_lo, lon_lo, lat_lo]
        c001 = self.cube[t_lo, lon_lo, lat_hi]
        c010 = self.cube[t_lo, lon_hi, lat_lo]
        c011 = self.cube[t_lo, lon_hi, lat_hi]
        c100 = self.cube[t_hi, lon_lo, lat_lo]
        c101 = self.cube[t_hi, lon_lo, lat_hi]
        c110 = self.cube[t_hi, lon_hi, lat_lo]
        c111 = self.cube[t_hi, lon_hi, lat_hi]

        w_lon = lon_weight[:, np.newaxis]
        w_lat = lat_weight[:, np.newaxis]
        w_t   = t_weight[:, np.newaxis]

        c00 = c000 * (ONE - w_lon) + c010 * w_lon
        c01 = c001 * (ONE - w_lon) + c011 * w_lon
        c0  = c00 * (ONE - w_lat) + c01 * w_lat

        c10 = c100 * (ONE - w_lon) + c110 * w_lon
        c11 = c101 * (ONE - w_lon) + c111 * w_lon
        c1  = c10 * (ONE - w_lat) + c11 * w_lat

        interp_matrix = c0 * (ONE - w_t) + c1 * w_t

        # ── STEP 4: METEOROLOGICAL DIRECTION ANALYSIS AND ANGLE RECOVERY ─────
        # Derive wind speed and direction from interpolated UGRD/VGRD components
        # rather than interpolating angular quantities directly.        
        ugrd = interp_matrix[:, self._idx["UGRD"]]
        vgrd = interp_matrix[:, self._idx["VGRD"]]
        
        wind_speed = np.sqrt(ugrd**2 + vgrd**2)
        norm_uv = np.hypot(ugrd, vgrd)
        
        # Convert UGRD/VGRD vectors to meteorological wind direction
        # (direction from which the wind originates).
        recovered_wdir = np.where(
            norm_uv > 1e-6,
            (np.arctan2(-ugrd, -vgrd) * (180.0 / np.pi) + 360.0) % 360.0,
            np.nan
        )

        course_rad = np.deg2rad(np.mod(vessel_courses, 360.0))
        ship_east  = np.sin(course_rad)
        ship_north = np.cos(course_rad)

        headwind = -(ugrd * ship_east + vgrd * ship_north)
        crosswind = ugrd * ship_north - vgrd * ship_east

        if self.has_vis:
            visibility_series = interp_matrix[:, self._idx["VIS"]]
        else:
            visibility_series = np.full(num_points, 10000.0, dtype=np.float32)

        translated_frame_data = {
            "wind_speed":          wind_speed.astype(np.float32),
            "wind_direction":      recovered_wdir.astype(np.float32),
            "temperature":         interp_matrix[:, self._idx["TMP"]],
            "visibility":          visibility_series.astype(np.float32),
            "gust_factor":         interp_matrix[:, self._idx["GUST"]],
            "pressure":            interp_matrix[:, self._idx["PRMSL"]],
            "humidity":            interp_matrix[:, self._idx["RH"]],
            "headwind_component":  headwind.astype(np.float32),
            "crosswind_component": crosswind.astype(np.float32),
            "wind_dir_valid":      (norm_uv > 1e-6).astype(np.int32)
        }

        df_out = df_chunk.with_columns([pl.Series(k, v) for k, v in translated_frame_data.items()])
        
        if self.boundary_mode == "nan" and np.any(out_of_mesh_mask):
            mask_series = pl.Series("_mask_flag_space", out_of_mesh_mask.astype(np.int32))
            df_out = df_out.with_columns(mask_series)
            for k in translated_frame_data.keys():
                if k == "wind_dir_valid":
                    df_out = df_out.with_columns(pl.when(pl.col("_mask_flag_space") == 1).then(pl.lit(0, dtype=pl.Int32)).otherwise(pl.col(k)).alias(k))
                else:
                    df_out = df_out.with_columns(pl.when(pl.col("_mask_flag_space") == 1).then(pl.lit(np.nan, dtype=pl.Float32)).otherwise(pl.col(k)).alias(k))
            df_out = df_out.drop("_mask_flag_space")
                
        return df_out.with_columns(pl.Series("out_of_mesh_flag", out_of_mesh_mask.astype(np.int32)))