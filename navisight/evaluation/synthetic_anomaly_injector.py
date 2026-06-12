# navisight/evaluation/synthetic_anomaly_injector.py
import numpy as np
import polars as pl
from navisight.pipeline.feature_registry import REGISTRY
from navisight.pipeline.ingestion.kinematics import RASTER_ENGINE, compute_kinematic_features

class ManifoldAwareAnomalyPerturbationEngine:
    """
    Multi-Modal Maritime Counterfactual Generator (M3CG-v8).
    Formulated as a Unified Canonical ENU State-Space Stochastic Hybrid System
    equipped with Isotropic Geodetic Linearization and 25m Spatial Memoization.
    """
    def __init__(self, random_state: int = 42):
        self.rng = np.random.default_rng(random_state)

    def inject_loitering_smuggling_drift(self, df_vessel_voyage: pl.DataFrame, severity: float = 0.85) -> pl.DataFrame:
        """Backward-compatible entry point routing directly to the updated multi-modal behavioral suite."""
        return self.inject_counterfactual_scenario(df_vessel_voyage, mode="dead_reckoning", severity=severity)

    def inject_counterfactual_scenario(self, df_vessel_voyage: pl.DataFrame, mode: str = "dead_reckoning", severity: float = 0.85) -> pl.DataFrame:
        """
        Synthesizes a counterfactual route scenario by executing continuous state-space dynamics
        entirely within a unified, canonical local metric ENU tangent plane frame.
        """
        if df_vessel_voyage.is_empty() or df_vessel_voyage.height < 25:
            return df_vessel_voyage
            
        df_proc = df_vessel_voyage.sort("timestamp_sec")
        total_points = len(df_proc)
        
        start_anomaly_idx = int(total_points * 0.35)
        end_anomaly_idx = int(total_points * 0.75)
        anomaly_duration = end_anomaly_idx - start_anomaly_idx
        
        if anomaly_duration <= 5:
            return df_proc

        # Array Isolation: Explicit numpy copies isolate states from shared memory references
        orig_lons = df_proc["lon_raw"].to_numpy().astype(np.float64).copy()
        orig_lats = df_proc["lat_raw"].to_numpy().astype(np.float64).copy()
        
        timestamps = df_proc["timestamp_sec"].to_numpy().astype(np.float64)
        speeds = df_proc["speed_raw"].to_numpy().astype(np.float64)
        courses = df_proc["course_raw"].to_numpy().astype(np.float64)

        # ── 1. CANONICAL ENU METRIC TANGENT FRAME INITIALIZATION ──
        anchor_lat = orig_lats[start_anomaly_idx]
        anchor_lon = orig_lons[start_anomaly_idx]
        
        # Establish exact WGS-84 metric grid conversion coefficients evaluated at our anchor point
        lat_to_meters = 111132.92 - 559.82 * np.cos(2.0 * np.radians(anchor_lat))
        lon_to_meters = 111412.84 * np.cos(np.radians(anchor_lat))
        meters_to_lat = 1.0 / lat_to_meters
        meters_to_lon = 1.0 / lon_to_meters
        
        # Convert ALL original paths to canonical ENU meters relative to the anomaly anchor point
        orig_x_meters = (orig_lons - anchor_lon) * lon_to_meters
        orig_y_meters = (orig_lats - anchor_lat) * lat_to_meters
        
        sim_x_meters = orig_x_meters.copy()
        sim_y_meters = orig_y_meters.copy()

        # Initialize continuous angular manifold parameters
        base_course_rad = np.radians(courses[max(0, start_anomaly_idx - 1)])
        theta_angle = base_course_rad
        
        drift_speed = 0.0
        rho = 0.92  # Autoregressive correlation persistence
        max_speed_clip = 6.0 * severity
        base_speed = max(2.0, speeds[max(0, start_anomaly_idx - 1)])

        # Continuous latent logit parameters tracking current tracking volatility
        regime_logits = np.array([2.0, 0.0, -1.0]) # [Nominal, Deviation, Stabilization]
        
        # Track active coordinate states entirely inside meter space
        current_x_m = orig_x_meters[start_anomaly_idx]
        current_y_m = orig_y_meters[start_anomaly_idx]
        NM_TO_M = 1852.0

        t_theta_series = np.linspace(0, 2.5 * np.pi, anomaly_duration)

        # ADJUSTMENT 1: 25-Meter ENU Cache Hashing protects against geometric and curvature noise
        spatial_lookup_cache = {}
        def _get_cached_land_distance_nm(x_m: float, y_m: float) -> float:
            spatial_grid_key = (int(x_m / 25.0), int(y_m / 25.0))
            if spatial_grid_key not in spatial_lookup_cache:
                lat_q = anchor_lat + (y_m * meters_to_lat)
                lon_q = anchor_lon + (x_m * meters_to_lon)
                spatial_lookup_cache[spatial_grid_key] = RASTER_ENGINE.lookup_single(lat_q, lon_q)
            return spatial_lookup_cache[spatial_grid_key]

        # ── 2. CONTINUOUS-TIME STOCHASTIC SIMULATION LOOP ──
        harbor = df_proc["harbor_basin_proximity"].to_numpy() if "harbor_basin_proximity" in df_proc.columns else None
        for i, idx in enumerate(range(start_anomaly_idx, end_anomaly_idx)):
            step_dt = max(1.0, timestamps[idx] - timestamps[max(0, idx - 1)]) if idx > 0 else 30.0
            progress_fraction = i / anomaly_duration
            
            land_dist_nm = _get_cached_land_distance_nm(current_x_m, current_y_m)

            # Leaky integrator updates
            harbor_prox = float(harbor[idx]) if harbor is not None and not np.isnan(harbor[idx]) else 0.0
            instantaneous_signal = (abs(drift_speed) * (step_dt / 60.0)) + (harbor_prox * 0.1)
            regime_logits[1] = (0.95 * regime_logits[1]) + (instantaneous_signal * 0.08 * severity)
            
            # The path-biased decay curve introduces intentional non-stationarity for adversarial scenarios
            if progress_fraction > 0.65:
                decay_ramp = (progress_fraction - 0.65) / 0.35
                regime_logits[2] = (0.94 * regime_logits[2]) + (0.12 * decay_ramp * severity)
                regime_logits[0] *= (1.0 - 0.04 * decay_ramp)

            # Softmax blending isolates smooth mixture weights
            exp_logits = np.exp(regime_logits - np.max(regime_logits))
            pi_weights = exp_logits / np.sum(exp_logits)

            # Blended innovations across active regimes
            speed_innov = (pi_weights[0] * self.rng.normal(0.02, 0.03) + 
                           pi_weights[1] * self.rng.normal(-0.08, 0.18) + 
                           pi_weights[2] * self.rng.normal(0.0, 0.01)) * severity
                           
            sin_innov = (pi_weights[0] * self.rng.normal(0.0, 0.005) + 
                         pi_weights[1] * self.rng.normal(0.02, 0.06) + 
                         pi_weights[2] * self.rng.normal(0.0, 0.002)) * severity
                         
            cos_innov = (pi_weights[0] * self.rng.normal(0.0, 0.005) + 
                         pi_weights[1] * self.rng.normal(0.02, 0.06) + 
                         pi_weights[2] * self.rng.normal(0.0, 0.002)) * severity

            # Clamped angular torque damping protects against numerical freezing artifacts
            theta_damping = np.clip(pi_weights[2], 0.0, 1.0) * 0.25
            drift_speed = (0.92 * drift_speed) + speed_innov - (theta_damping * drift_speed)
            drift_speed = np.clip(drift_speed, -5.0 * severity, 5.0 * severity)
            simulated_knots = max(1.0, base_speed + drift_speed)
            
            # ADJUSTMENT 3: Apply the Ornstein-Uhlenbeck damping pull BEFORE running angular wrapping steps
            theta_angle -= theta_damping * np.sin(theta_angle - base_course_rad)
            theta_angle += np.arctan2(sin_innov, cos_innov)
            theta_angle = (theta_angle + np.pi) % (2.0 * np.pi) - np.pi
            
            # Translate scalar velocity into an ENU metric vector (meters/sec)
            speed_mps = simulated_knots * 0.514444
            v_attract_x = speed_mps * np.sin(theta_angle)
            v_attract_y = speed_mps * np.cos(theta_angle)

            # ── ADJUSTMENT 2: ISOTROPIC FIRST-ORDER GEODETIC LINEARIZATION STENCIL ──
            # Samples finite differences exclusively using metric ENU displacement offsets
            ds_m = 100.0  
            d_north = _get_cached_land_distance_nm(current_x_m, current_y_m + ds_m) * NM_TO_M
            d_south = _get_cached_land_distance_nm(current_x_m, current_y_m - ds_m) * NM_TO_M
            d_east  = _get_cached_land_distance_nm(current_x_m + ds_m, current_y_m) * NM_TO_M
            d_west  = _get_cached_land_distance_nm(current_x_m - ds_m, current_y_m) * NM_TO_M
            
            # Gradient scales cleanly to meters per meter (dimensionless direction components)
            grad_y = (d_north - d_south) / (2.0 * ds_m)
            grad_x = (d_east - d_west) / (2.0 * ds_m)
            grad_norm = np.sqrt(grad_x**2 + grad_y**2) + 1e-9
            
            # Sigmoid-style soft saturation curve prevents sharp numerical cliffs near coastlines
            repulsion_magnitude = speed_mps * (1.0 / (1.0 + np.exp(15.0 * (land_dist_nm - 0.10))))

            v_repulse_y = (grad_y / grad_norm) * repulsion_magnitude
            v_repulse_x = (grad_x / grad_norm) * repulsion_magnitude

            # ── CONSTRAINED CANONICAL METRIC PROPAGATION ENGINES ──
            if mode == "loitering":
                base_radius = 600.0 * severity
                t_theta_val = t_theta_series[i]
                radius_wobble = base_radius + self.rng.normal(0.0, 45.0 * severity)
                
                target_x = orig_x_meters[start_anomaly_idx] + (np.cos(t_theta_val) * radius_wobble)
                target_y = orig_y_meters[start_anomaly_idx] + (np.sin(t_theta_val) * radius_wobble)
                
                if _get_cached_land_distance_nm(target_x, target_y) > 0.02:
                    current_x_m = target_x
                    current_y_m = target_y
                else:
                    current_x_m = orig_x_meters[start_anomaly_idx] + (np.cos(t_theta_val) * (radius_wobble * 0.3))
                    current_y_m = orig_y_meters[start_anomaly_idx] + (np.sin(t_theta_val) * (radius_wobble * 0.3))
            
            elif mode == "dead_reckoning":
                current_x_m += (v_attract_x + v_repulse_x) * step_dt
                current_y_m += (v_attract_y + v_repulse_y) * step_dt
                
            elif mode == "coastal_creep":
                target_boundary_x = (23.6010 - anchor_lon) * lon_to_meters
                target_boundary_y = (37.9360 - anchor_lat) * lat_to_meters
                
                dir_x = target_boundary_x - current_x_m
                dir_y = target_boundary_y - current_y_m
                dir_norm = np.sqrt(dir_x**2 + dir_y**2) + 1e-9
                
                base_bearing = np.arctan2(dir_x, dir_y)
                stochastic_bearing_noise = self.rng.normal(0.0, 0.04) * severity
                stabilized_bearing = base_bearing + stochastic_bearing_noise
                
                creep_speed = 3.5 * severity
                distance_m = (creep_speed * 0.514444) * step_dt
                
                step_x = current_x_m + (np.sin(stabilized_bearing) * distance_m)
                step_y = current_y_m + (np.cos(stabilized_bearing) * distance_m)
                
                if 0.02 <= _get_cached_land_distance_nm(step_x, step_y) <= 1.5:
                    current_x_m = step_x
                    current_y_m = step_y
                else:
                    # ADJUSTMENT 4: Normalized spatial diffusion maps noise consistently along the velocity vector
                    current_x_m += self.rng.normal(0.0, 0.1) * distance_m * np.sin(stabilized_bearing + np.pi/2)
                    current_y_m += self.rng.normal(0.0, 0.1) * distance_m * np.cos(stabilized_bearing + np.pi/2)

            sim_x_meters[idx] = current_x_m
            sim_y_meters[idx] = current_y_m

        # ── 3. QUINTIC SPLINE VECTOR OFFSET BLENDING ──
        if end_anomaly_idx < total_points:
            denom = max(1, total_points - end_anomaly_idx)
            
            # Extract spatial deflection offsets at the conclusion of the anomaly window
            lat_offset_m = sim_y_meters[end_anomaly_idx - 1] - orig_y_meters[end_anomaly_idx - 1]
            lon_offset_m = sim_x_meters[end_anomaly_idx - 1] - orig_x_meters[end_anomaly_idx - 1]
            
            for idx in range(end_anomaly_idx, total_points):
                linear_weight = (idx - end_anomaly_idx) / denom
                smooth_decay = 1.0 - (6 * (linear_weight**5) - 15 * (linear_weight**4) + 10 * (linear_weight**3))
                
                sim_y_meters[idx] = orig_y_meters[idx] + (lat_offset_m * smooth_decay)
                sim_x_meters[idx] = orig_x_meters[idx] + (lon_offset_m * smooth_decay)

        # ── 4. CANONICAL GEODETIC COORD RE-MAPPING PASS ──
        final_lats = anchor_lat + (sim_y_meters / lat_to_meters)
        final_lons = anchor_lon + (sim_x_meters / lon_to_meters)

        # Pack the modified coordinates into a clean Polars dataframe shell
        df_counterfactual_base = df_proc.with_columns([
            pl.Series("lon_raw", final_lons, dtype=pl.Float64),
            pl.Series("lat_raw", final_lats, dtype=pl.Float64)
        ])
        
        # Completely regenerate features via the authoritative kinematics layer
        df_counterfactual = compute_kinematic_features(df_counterfactual_base)
        
        # Verification Firewall: Hard fail fast on any schema mutations
        for col in REGISTRY.all_features:
            if col not in df_counterfactual.columns:
                raise RuntimeError(
                    f"CRITICAL COUNTERFACTUAL SCHEMA DRIFT: The required feature column token '{col}' "
                    f"is missing from the dataframe post-simulation recomputation. Verify alignment contracts."
                )
                
        return df_counterfactual