# -*- coding: utf-8 -*-
"""
core/imm_ekf.py - Batch-Vectorized 3-Mode Interacting Multiple Model Extended
Kalman Filter (IMM-EKF) in local tangent plane ENU (East-North-Up, km).

Modes (SDD Section 3.1):
  0: Mode_CV     (Constant Velocity cruise, omega = 0)
  1: Mode_CTRV   (Coordinated Turn Rate & Velocity maneuver)
  2: Mode_LOITER (Acoustic orbiting / loitering with sigma^2_pos = 18.0 km^2/min)

Strictly single-thread 3D/4D NumPy tensor operations across all N tracks simultaneously.
Guarantees numerical stability (Taylor series limit as |omega| -> 0, Joseph-form covariance
update, and eigenvalue clipping) so NaN/Inf never occur.
"""
import config.settings  # Enforce single-thread BLAS/OMP before numpy import
import math
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import h3

from core.models import (
    Track,
    Observation,
    wgs84_to_enu,
    enu_to_wgs84,
    enu_to_wgs84_vec,
    DEFAULT_AIRSPEED_KMH,
)

# Markov mode transition matrix Pi_imm (SDD Section 3.1)
PI_IMM_STANDARD = np.array([
    [0.85, 0.12, 0.03],  # from CV
    [0.25, 0.70, 0.05],  # from CTRV
    [0.20, 0.15, 0.65],  # from LOITER
], dtype=np.float64)

# Decoy transition matrix (Gerbera/Parodiya decoys maneuver and loiter more frequently)
PI_IMM_DECOY = np.array([
    [0.72, 0.18, 0.10],
    [0.20, 0.65, 0.15],
    [0.15, 0.15, 0.70],
], dtype=np.float64)

# Ballistic transition matrix: quasi-ballistic missiles never loiter (LOITER prob = 0)
PI_IMM_BALLISTIC = np.array([
    [0.96, 0.04, 0.00],
    [0.60, 0.40, 0.00],
    [1.00, 0.00, 0.00],
], dtype=np.float64)

_BALLISTIC_HUB_COORDS: Dict[str, Tuple[float, float]] = {
    "bryansk": (53.24, 34.36),
    "kursk": (51.73, 36.19),
    "belgorod": (50.60, 36.59),
    "voronezh": (51.67, 39.21),
    "millerovo": (48.95, 40.29),
    "rostov": (47.23, 39.70),
    "taganrog": (47.21, 38.92),
    "yeisk": (46.68, 38.21),
    "mozyr": (52.05, 29.25),
    "crimea": (45.70, 34.42),
    "dzhankoi": (45.70, 34.42),
    "gvardiiske": (45.12, 33.98),
    "chauda": (45.01, 35.84),
    "tot_zap": (46.90, 35.37),
    "tot_donbas": (48.01, 37.80),
}

BALLISTIC_MIN_KMH = 2000.0
BALLISTIC_MAX_KMH = 5800.0

MODE_NAMES = ("CV", "CTRV", "LOITER")

# --- Robust update parameters (v9.2) ---
HUBER_GAMMA = 1.85          # Mahalanobis threshold above which the measurement is down-weighted
HUBER_MAX_R_MULT = 25.0     # hard cap for R_eff / R so the track is never frozen
OOSM_ALONG_MIN_KM = -18.0   # late-report window (target already passed this point)
OOSM_ALONG_MAX_KM = -2.0
OOSM_CROSS_MAX_KM = 6.0
OOSM_CROSS_GAIN = 0.5       # fraction of lateral offset applied on a suppressed single backward fix
WIND_DRIFT_ALPHA_NOMINAL = 0.05   # extra cross-wind drift coefficient while COASTING with GPS
WIND_DRIFT_ALPHA_EW = 0.35        # ... under strong EW (jamming_score >= EW_THRESHOLD)
EW_THRESHOLD = 0.60
HUBER_GROSS_D = 3.0         # d_M vs IMM-combined prior above which a FIRST suspect fix cannot change velocity/course
                            # (prior cov is wide, a 15 km jump is only d_M~3.9; 2 consecutive gross fixes = real manoeuvre)


class BatchIMMEKF:
    """
    Vectorized 3-mode IMM-EKF operating on a batch of Track objects in 1 thread.
    State vector per mode: x = [px (km), py (km), v (km/s), psi (rad, CW from North), omega (rad/s)]^T
    """

    def __init__(self):
        # Process noise variance rates (per second)
        # Note: LOITER has spatial variance 18.0 km^2/min = 0.30 km^2/s (SDD Section 3.1)
        self.q_pos_rate = np.array([0.015, 0.045, 0.300], dtype=np.float64)  # km^2 / s
        self.q_vel_rate = np.array([1e-6, 4e-6, 8e-6], dtype=np.float64)     # (km/s)^2 / s
        self.q_psi_rate = np.array([3e-5, 4e-4, 2e-3], dtype=np.float64)     # rad^2 / s
        self.q_omg_rate = np.array([1e-8, 2e-6, 1e-6], dtype=np.float64)     # (rad/s)^2 / s

    @staticmethod
    def _default_tactical_heading(
        lat: float, lon: float, threat_class: str, from_zone: Optional[str] = None
    ) -> float:
        """
        Computes a physically realistic tactical attack bearing (degrees CW from North)
        from explicit from_zone launch hubs or regional operational corridors.
        """
        if from_zone:
            z_key = str(from_zone).strip().lower()
            hub = _BALLISTIC_HUB_COORDS.get(z_key)
            if hub is not None:
                hx, hy = wgs84_to_enu(hub[0], hub[1])
                tx, ty = wgs84_to_enu(lat, lon)
                dx, dy = tx - hx, ty - hy
                if math.hypot(dx, dy) > 15.0:
                    return (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0

        if threat_class == "KAB":
            if lat >= 49.6:
                return 205.0
            if lat < 47.85 and lon < 36.4:
                return 340.0
            return 278.0

        if threat_class == "BALLISTIC":
            # Southern Ukraine (from Crimea / Dzhankoi / Chauda northward)
            if lat < 47.6 and lon >= 30.0:
                return 350.0 if lon < 33.5 else 10.0
            # Northern / Central Ukraine (from Bryansk / Kursk / Voronezh SSW/SW)
            if lat >= 49.6:
                return 205.0 if lon < 33.0 else 225.0
            # Eastern / Dnieper basin (from Belgorod / Rostov / Taganrog WSW/WNW)
            return 245.0 if lon >= 34.5 else 215.0

        if lat >= 50.0 and lon >= 33.0:
            return 245.0
        if lat < 48.2 and lon >= 32.0:
            return 315.0
        return 285.0

    @staticmethod
    def initialize_track_from_obs(obs: Observation, track_uuid: str) -> Track:
        px, py = wgs84_to_enu(obs.lat, obs.lon)
        spd_kmh = obs.speed_kmh if (obs.speed_kmh and obs.speed_kmh > 10.0) else DEFAULT_AIRSPEED_KMH.get(obs.threat_class, 185.0)
        if obs.is_decoy and not obs.speed_kmh:
            spd_kmh = DEFAULT_AIRSPEED_KMH["DECOY"]
        if obs.threat_class == "BALLISTIC":
            spd_kmh = float(np.clip(spd_kmh, BALLISTIC_MIN_KMH, BALLISTIC_MAX_KMH))
        v_kms = spd_kmh / 3600.0
        hdg_deg = (
            obs.heading_deg
            if obs.heading_deg is not None
            else BatchIMMEKF._default_tactical_heading(
                obs.lat, obs.lon, obs.threat_class, from_zone=getattr(obs, "from_zone", None)
            )
        )
        psi_rad = math.radians(hdg_deg % 360.0)

        x0 = np.array([px, py, v_kms, psi_rad, 0.0], dtype=np.float64)
        mode_states = np.tile(x0, (3, 1))  # shape (3, 5)

        sig_r = obs.sigma_r_km
        P0 = np.diag([sig_r ** 2, sig_r ** 2, (0.015) ** 2, (0.25) ** 2, (0.002) ** 2]).astype(np.float64)
        mode_covs = np.tile(P0, (3, 1, 1))  # shape (3, 5, 5)

        if obs.threat_class == "BALLISTIC":
            mode_probs = np.array([0.96, 0.04, 0.00], dtype=np.float64)
        elif obs.kinematic_hint == "TURN":
            mode_probs = np.array([0.10, 0.85, 0.05], dtype=np.float64)
        elif obs.kinematic_hint == "LOITER":
            mode_probs = np.array([0.05, 0.10, 0.85], dtype=np.float64)
        else:
            mode_probs = np.array([0.80, 0.15, 0.05], dtype=np.float64)

        dom_idx = int(np.argmax(mode_probs))
        h3_cell = obs.h3_r6 or h3.latlng_to_cell(obs.lat, obs.lon, 6)

        trk = Track(
            track_uuid=track_uuid,
            threat_class=obs.threat_class,
            subkind=obs.subkind or ("decoy" if obs.is_decoy else None),
            status="CONFIRMED" if obs.source_id in ("sensor_ext_1", "feed_primary_a") else "TENTATIVE",
            imm_mode=MODE_NAMES[dom_idx],
            count=max(1, obs.target_count),
            lat=obs.lat,
            lon=obs.lon,
            ground_speed_kmh=spd_kmh,
            course_deg=hdg_deg % 360.0,
            turn_rate_rad_s=0.0,
            cov_major_km=round(sig_r * 1.4, 2),
            cov_minor_km=round(sig_r * 0.9, 2),
            cov_angle_deg=hdg_deg % 360.0,
            h3_cell=h3_cell,
            locality=obs.raw_text or h3_cell,
            created_ts=obs.timestamp,
            last_updated_ts=obs.timestamp,
            last_seen_ts=obs.timestamp,
            last_obs_lat=obs.lat,
            last_obs_lon=obs.lon,
            hit_updates=1,
            missed_updates=0,
            mode_states=mode_states,
            mode_covs=mode_covs,
            mode_probs=mode_probs,
            sources_history=[(obs.source_id, obs.timestamp, h3_cell)],
            trail=list(obs.trail) if obs.trail else [[round(obs.lon, 5), round(obs.lat, 5)]],
        )
        return trk

    def predict_batch(
        self,
        tracks: List[Track],
        target_ts: float,
        wind_speed_kmh: float = 0.0,
        wind_dir_deg: float = 0.0,
    ) -> None:
        """
        Vectorized IMM interaction + 3-mode state & covariance time prediction
        across all N tracks simultaneously using 3D/4D NumPy arrays.
        """
        n = len(tracks)
        if n == 0:
            return

        # Ensure internal tensors are initialized
        for t in tracks:
            if t.mode_states is None or t.mode_covs is None or t.mode_probs is None:
                px, py = wgs84_to_enu(t.lat, t.lon)
                v_kms = max(0.01, t.ground_speed_kmh / 3600.0)
                psi = math.radians(t.course_deg % 360.0)
                x0 = np.array([px, py, v_kms, psi, t.turn_rate_rad_s], dtype=np.float64)
                t.mode_states = np.tile(x0, (3, 1))
                t.mode_covs = np.tile(np.diag([36.0, 36.0, 2e-4, 0.06, 4e-6]), (3, 1, 1))
                t.mode_probs = np.array([0.80, 0.15, 0.05], dtype=np.float64)

        # Stack batch tensors: X shape (N, 3, 5), P shape (N, 3, 5, 5), mu shape (N, 3)
        X = np.stack([t.mode_states for t in tracks], axis=0)
        P = np.stack([t.mode_covs for t in tracks], axis=0)
        mu = np.stack([t.mode_probs for t in tracks], axis=0)
        dt = np.array([max(0.0, min(600.0, target_ts - t.last_updated_ts)) for t in tracks], dtype=np.float64)

        # Skip if all dt == 0
        if np.all(dt <= 1e-6):
            return

        # 1. IMM Mixing step (vectorized across N tracks)
        is_decoy_mask = np.array([t.is_decoy for t in tracks], dtype=bool)
        is_ballistic_mask = np.array([t.threat_class == "BALLISTIC" for t in tracks], dtype=bool)
        Pi = np.where(
            is_ballistic_mask[:, None, None],
            PI_IMM_BALLISTIC[None, :, :],
            np.where(is_decoy_mask[:, None, None], PI_IMM_DECOY[None, :, :], PI_IMM_STANDARD[None, :, :]),
        )  # (N, 3, 3)

        # c_bar_j = sum_i mu_i * Pi_ij -> shape (N, 3)
        c_bar = np.einsum("ni,nij->nj", mu, Pi)
        c_bar = np.maximum(c_bar, 1e-12)
        # Mixing probabilities mu_{i|j} = (mu_i * Pi_ij) / c_bar_j -> shape (N, 3, 3)
        mu_mix = (mu[:, :, None] * Pi) / c_bar[:, None, :]

        # Mixed state X_mix[n, j] = sum_i mu_mix[n, i, j] * X[n, i] -> shape (N, 3, 5)
        X_mix = np.einsum("nij,nik->njk", mu_mix, X)
        # Circular weighted mean for course angle psi (index 3) to prevent 0/360 wraparound flips
        sin_mix = np.einsum("nij,ni->nj", mu_mix, np.sin(X[:, :, 3]))
        cos_mix = np.einsum("nij,ni->nj", mu_mix, np.cos(X[:, :, 3]))
        X_mix[:, :, 3] = np.mod(np.arctan2(sin_mix, cos_mix), 2.0 * np.pi)
        # Mixed covariance P_mix[n, j] = sum_i mu_mix[n, i, j] * (P[n, i] + dX dX^T)
        dX = X[:, :, None, :] - X_mix[:, None, :, :]  # (N, 3, 3, 5)
        # Wrap angle difference in state index 3 (psi)
        dX[..., 3] = (dX[..., 3] + np.pi) % (2.0 * np.pi) - np.pi
        outer_dX = np.einsum("nijk,nijl->nijkl", dX, dX)
        P_mix = np.einsum("nij,nijkl->njkl", mu_mix, P[:, :, None, :, :] + outer_dX)

        # 2. 100m Wind vector components in km/s: Open-Meteo wind_dir_deg is meteorological ("FROM" direction),
        # so convert to downwind direction (+180 deg) before projecting onto East/North velocity components.
        # High-altitude ballistic missiles in upper stratosphere are unaffected by 100m surface winds.
        v_wind_kms = np.asarray(wind_speed_kmh, dtype=np.float64) / 3600.0
        theta_wind_to_rad = np.radians((np.asarray(wind_dir_deg, dtype=np.float64) + 180.0) % 360.0)
        wx = np.where(is_ballistic_mask, 0.0, v_wind_kms * np.sin(theta_wind_to_rad))
        wy = np.where(is_ballistic_mask, 0.0, v_wind_kms * np.cos(theta_wind_to_rad))

        # Extra cross-wind drift for blind-extrapolated (COASTING) propeller UAVs (v9.2).
        # alpha = 0.05 nominal, 0.35 when the sector EW score >= EW_THRESHOLD (INS-only flight).
        coast_mask = np.array(
            [(t.status == "COASTING") and ("UAV" in str(t.threat_class).upper()) for t in tracks],
            dtype=bool,
        )
        jam = np.array([float(getattr(t, "jamming_score", 0.0)) for t in tracks], dtype=np.float64)
        alpha_drift = np.where(coast_mask, np.where(jam >= EW_THRESHOLD, WIND_DRIFT_ALPHA_EW, WIND_DRIFT_ALPHA_NOMINAL), 0.0)

        def _cross_drift(psi_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            nx, ny = np.cos(psi_arr), -np.sin(psi_arr)
            w_cross = wx * nx + wy * ny
            return alpha_drift * w_cross * nx, alpha_drift * w_cross * ny

        X_pred = np.copy(X_mix)
        P_pred = np.zeros_like(P_mix)

        # --- Mode 0: Mode_CV (omega = 0) ---
        px0, py0, v0, psi0 = X_mix[:, 0, 0], X_mix[:, 0, 1], X_mix[:, 0, 2], X_mix[:, 0, 3]
        sin_psi0 = np.sin(psi0)
        cos_psi0 = np.cos(psi0)
        dxw0, dyw0 = _cross_drift(psi0)
        X_pred[:, 0, 0] = px0 + (v0 * sin_psi0 + wx + dxw0) * dt
        X_pred[:, 0, 1] = py0 + (v0 * cos_psi0 + wy + dyw0) * dt
        X_pred[:, 0, 2] = v0
        X_pred[:, 0, 3] = psi0 % (2.0 * np.pi)
        X_pred[:, 0, 4] = 0.0

        F0 = np.broadcast_to(np.eye(5, dtype=np.float64), (n, 5, 5)).copy()
        F0[:, 0, 2] = sin_psi0 * dt
        F0[:, 0, 3] = v0 * cos_psi0 * dt
        F0[:, 1, 2] = cos_psi0 * dt
        F0[:, 1, 3] = -v0 * sin_psi0 * dt
        F0[:, 4, 4] = 0.0

        # --- Mode 1: Mode_CTRV (Coordinated Turn Rate & Velocity, safe limit as |omega| -> 0) ---
        px1, py1, v1, psi1, omg1 = (
            X_mix[:, 1, 0], X_mix[:, 1, 1], X_mix[:, 1, 2], X_mix[:, 1, 3], X_mix[:, 1, 4]
        )
        psi1_new = psi1 + omg1 * dt
        small_omg = np.abs(omg1) <= 1e-4

        # Safe denominator for non-small omega
        omg_safe = np.where(small_omg, 1.0, omg1)
        # Exact turn integral vs 2nd-order Taylor expansion (zero division-by-zero!)
        dx_exact = (v1 / omg_safe) * (-np.cos(psi1_new) + np.cos(psi1))
        dy_exact = (v1 / omg_safe) * (np.sin(psi1_new) - np.sin(psi1))
        dx_taylor = v1 * dt * (np.sin(psi1) + 0.5 * omg1 * dt * np.cos(psi1))
        dy_taylor = v1 * dt * (np.cos(psi1) - 0.5 * omg1 * dt * np.sin(psi1))

        dxw1, dyw1 = _cross_drift(psi1)
        X_pred[:, 1, 0] = px1 + np.where(small_omg, dx_taylor, dx_exact) + (wx + dxw1) * dt
        X_pred[:, 1, 1] = py1 + np.where(small_omg, dy_taylor, dy_exact) + (wy + dyw1) * dt
        X_pred[:, 1, 2] = v1
        X_pred[:, 1, 3] = psi1_new % (2.0 * np.pi)
        X_pred[:, 1, 4] = omg1

        F1 = np.broadcast_to(np.eye(5, dtype=np.float64), (n, 5, 5)).copy()
        sin_p1 = np.sin(psi1)
        cos_p1 = np.cos(psi1)
        F1[:, 0, 2] = sin_p1 * dt
        F1[:, 0, 3] = v1 * cos_p1 * dt
        F1[:, 0, 4] = 0.5 * v1 * cos_p1 * (dt ** 2)
        F1[:, 1, 2] = cos_p1 * dt
        F1[:, 1, 3] = -v1 * sin_p1 * dt
        F1[:, 1, 4] = -0.5 * v1 * sin_p1 * (dt ** 2)
        F1[:, 3, 4] = dt

        # --- Mode 2: Mode_LOITER (Acoustic orbiting, sigma^2_pos = 18.0 km^2/min) ---
        # Keep airframe speed v2 intact in state index 2 so IMM mode mixing never drains cruise speed!
        px2, py2, v2, psi2 = X_mix[:, 2, 0], X_mix[:, 2, 1], X_mix[:, 2, 2], X_mix[:, 2, 3]
        X_pred[:, 2, 0] = px2 + 0.15 * wx * dt
        X_pred[:, 2, 1] = py2 + 0.15 * wy * dt
        X_pred[:, 2, 2] = v2
        X_pred[:, 2, 3] = (psi2 + 0.015 * dt) % (2.0 * np.pi)
        X_pred[:, 2, 4] = 0.0

        F2 = np.broadcast_to(np.eye(5, dtype=np.float64), (n, 5, 5)).copy()
        F2[:, 2, 2] = 1.0
        F2[:, 4, 4] = 0.0

        F_all = np.stack([F0, F1, F2], axis=1)  # shape (N, 3, 5, 5)

        # Build process noise covariance Q_all shape (N, 3, 5, 5)
        Q_all = np.zeros((n, 3, 5, 5), dtype=np.float64)
        for m in range(3):
            Q_all[:, m, 0, 0] = self.q_pos_rate[m] * dt
            Q_all[:, m, 1, 1] = self.q_pos_rate[m] * dt
            Q_all[:, m, 2, 2] = self.q_vel_rate[m] * dt
            Q_all[:, m, 3, 3] = self.q_psi_rate[m] * dt
            Q_all[:, m, 4, 4] = self.q_omg_rate[m] * dt

        # Q_EW = Q * (1 + 0.5 * jamming_score): widen process noise inside active EW zones
        Q_all *= (1.0 + 0.5 * np.clip(jam, 0.0, 1.0))[:, None, None, None]

        # P_pred = F * P_mix * F^T + Q
        P_pred = np.einsum("nmij,nmjk,nmlk->nmil", F_all, P_mix, F_all) + Q_all
        # Symmetrize to eliminate floating-point drift
        P_pred = 0.5 * (P_pred + np.swapaxes(P_pred, -1, -2))

        # 3. Combine IMM output state and covariance across modes
        mu_pred = c_bar / np.sum(c_bar, axis=1, keepdims=True)
        X_comb, P_comb = self._combine_imm_estados(X_pred, P_pred, mu_pred)

        # Write back to Track objects
        lats, lons = enu_to_wgs84_vec(X_comb[:, 0], X_comb[:, 1])
        dom_modes = np.argmax(mu_pred, axis=1)

        for i, trk in enumerate(tracks):
            if dt[i] <= 1e-6:
                continue
            if is_ballistic_mask[i]:
                X_pred[i, :, 2] = np.clip(X_pred[i, :, 2], BALLISTIC_MIN_KMH / 3600.0, BALLISTIC_MAX_KMH / 3600.0)
                X_pred[i, :, 4] = 0.0
                mu_pred[i, 2] = 0.0
                s_mu = float(np.sum(mu_pred[i, :2]))
                mu_pred[i] = (mu_pred[i] / s_mu) if s_mu > 1e-12 else np.array([0.96, 0.04, 0.00], dtype=float)
                dom_modes[i] = int(np.argmax(mu_pred[i]))
            trk.mode_states = X_pred[i]
            trk.mode_covs = P_pred[i]
            trk.mode_probs = mu_pred[i]
            trk.imm_mode = MODE_NAMES[int(dom_modes[i])]
            trk.lat = float(lats[i])
            trk.lon = float(lons[i])
            if is_ballistic_mask[i]:
                trk.ground_speed_kmh = float(np.clip(X_comb[i, 2] * 3600.0, BALLISTIC_MIN_KMH, BALLISTIC_MAX_KMH))
                trk.turn_rate_rad_s = 0.0
            else:
                trk.ground_speed_kmh = float(max(25.0, X_comb[i, 2] * 3600.0))
                trk.turn_rate_rad_s = float(X_comb[i, 4])
            trk.course_deg = float(math.degrees(X_comb[i, 3]) % 360.0)
            trk.last_updated_ts = target_ts
            try:
                trk.h3_cell = h3.latlng_to_cell(trk.lat, trk.lon, 6)
            except Exception:
                pass
            maj, minr, ang = self.extract_ellipse_km(P_comb[i, :2, :2])
            trk.cov_major_km = maj
            trk.cov_minor_km = minr
            trk.cov_angle_deg = ang

    @staticmethod
    def _combine_imm_estados(
        X_modes: np.ndarray, P_modes: np.ndarray, mu: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Combines 3-mode state vectors and covariances using circular mean for course angle psi."""
        X_comb = np.einsum("nm,nmk->nk", mu, X_modes)
        # Circular mean for angle psi (index 3)
        sin_psi = np.einsum("nm,nm->n", mu, np.sin(X_modes[:, :, 3]))
        cos_psi = np.einsum("nm,nm->n", mu, np.cos(X_modes[:, :, 3]))
        X_comb[:, 3] = np.mod(np.arctan2(sin_psi, cos_psi), 2.0 * np.pi)

        diff = X_modes - X_comb[:, None, :]
        diff[..., 3] = (diff[..., 3] + np.pi) % (2.0 * np.pi) - np.pi
        outer = np.einsum("nmi,nmj->nmij", diff, diff)
        P_comb = np.einsum("nm,nmij->nij", mu, P_modes + outer)
        return X_comb, P_comb

    def update_track_with_obs(self, trk: Track, obs: Observation) -> None:
        """
        Updates a single track's 3-mode IMM-EKF state with a matched Observation.
        Uses Joseph-form covariance update and analytical 2x2/3x3 innovation inverse.
        """
        zx, zy = wgs84_to_enu(obs.lat, obs.lon)
        sig_r2 = obs.sigma_r_km ** 2
        is_ballistic = trk.threat_class == "BALLISTIC"
        is_stationary_bal = False
        if is_ballistic and trk.last_obs_lat is not None and trk.last_obs_lon is not None and trk.hit_updates >= 1:
            lx, ly = wgs84_to_enu(trk.last_obs_lat, trk.last_obs_lon)
            if math.hypot(zx - lx, zy - ly) < 1.5:
                is_stationary_bal = True

        # Override mode probabilities if explicit kinematic hint is present in OSINT text (SDD Section 3.1)
        if not is_ballistic:
            if obs.kinematic_hint == "TURN":
                trk.mode_probs = np.array([0.10, 0.85, 0.05], dtype=np.float64)
            elif obs.kinematic_hint == "LOITER":
                trk.mode_probs = np.array([0.05, 0.10, 0.85], dtype=np.float64)

        likelihoods = np.ones(3, dtype=np.float64)
        I5 = np.eye(5, dtype=np.float64)

        # --- OOSM guard: a single late/backward fix must not roll the target back ---
        zx, zy = self._apply_oosm_guard(trk, zx, zy, sig_r2, obs)
        # --- Huber robust scaling of measurement noise (one factor per track) ---
        r_mult, d_m_prior = self._huber_r_multiplier(trk, zx, zy, sig_r2)
        sig_r2_eff = sig_r2 * r_mult
        gross_now = d_m_prior > HUBER_GROSS_D
        suppress_vel_hdg = bool(
            is_stationary_bal
            or (r_mult > 1.0 and d_m_prior > HUBER_GROSS_D and trk.oosm_streak < 2 and trk.gross_streak < 1)
        )
        vel_hdg_scale = 0.0 if suppress_vel_hdg else (1.0 / r_mult if r_mult > 1.0 else 1.0)

        for m in range(3):
            xm = trk.mode_states[m]
            Pm = trk.mode_covs[m]

            # Position innovation y = z - H*x (2D ENU position)
            y = np.array([zx - xm[0], zy - xm[1]], dtype=np.float64)
            S = Pm[:2, :2] + np.diag([sig_r2, sig_r2])  # unscaled S: mode likelihoods stay discriminative

            # Analytical 2x2 determinant & inverse
            det_S = float(S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0])
            if det_S <= 1e-9 or not np.isfinite(det_S):
                det_S = 1e-3
                S_inv = np.diag([1.0 / max(1.0, S[0, 0]), 1.0 / max(1.0, S[1, 1])])
            else:
                S_inv = np.array([[S[1, 1], -S[0, 1]], [-S[1, 0], S[0, 0]]], dtype=np.float64) / det_S


            # Kalman gain with Huber-scaled R_eff
            S_eff = Pm[:2, :2] + np.diag([sig_r2_eff, sig_r2_eff])
            det_eff = float(S_eff[0, 0] * S_eff[1, 1] - S_eff[0, 1] * S_eff[1, 0])
            if det_eff <= 1e-9 or not np.isfinite(det_eff):
                S_inv_eff = S_inv
            else:
                S_inv_eff = np.array([[S_eff[1, 1], -S_eff[0, 1]], [-S_eff[1, 0], S_eff[0, 0]]], dtype=np.float64) / det_eff
            K = Pm[:, :2] @ S_inv_eff
            if is_stationary_bal:
                # Repeated static destination coordinates (e.g., sensor_ext_1 pinning) must not
                # pull ballistic velocity downward or rotate heading backward as the prior passes.
                K[2:, :] = 0.0
            elif r_mult > 1.0:
                # Position is pulled with the Huber-scaled gain, but velocity/course/turn-rate
                # corrections inferred from a suspect fix are attenuated by w^2 = 1/r_mult
                # (Joseph form below stays valid for any gain K).
                K[2:, :] /= r_mult
                if d_m_prior > HUBER_GROSS_D and trk.oosm_streak < 2 and trk.gross_streak < 1:
                    K[2:, :] = 0.0  # first gross fix: do not let it rotate course / change speed
            # Mode likelihood from the Huber-scaled innovation covariance: an outlier must not
            # flip IMM mode probabilities (CV -> CTRV/LOITER) and thereby rotate the course.
            d2 = float(y.T @ S_inv_eff @ y)
            det_l = det_eff if (det_eff > 1e-9 and np.isfinite(det_eff)) else det_S
            likelihoods[m] = max(1e-12, math.exp(-0.5 * min(50.0, d2)) / (2.0 * math.pi * math.sqrt(abs(det_l))))
            xm_new = xm + (K @ y)

            # Softly fuse observed heading and speed if provided (blocked on first gross outliers / pinned ballistic)
            if obs.heading_deg is not None and not suppress_vel_hdg and (is_ballistic or obs.kinematic_hint != "LOITER"):
                obs_psi = math.radians(obs.heading_deg % 360.0)
                d_psi = (obs_psi - xm_new[3] + math.pi) % (2.0 * math.pi) - math.pi
                xm_new[3] = (xm_new[3] + 0.45 * vel_hdg_scale * d_psi) % (2.0 * math.pi)
                if not is_ballistic and m == 1 and vel_hdg_scale > 0.25 and trk.last_seen_ts > 0 and (obs.timestamp - trk.last_seen_ts) > 2.0:
                    dt_obs = min(180.0, obs.timestamp - trk.last_seen_ts)
                    xm_new[4] = np.clip(d_psi / dt_obs, -0.035, 0.035)
            elif obs.heading_deg is None and trk.threat_class == "KAB":
                # Keep un-headed frontline KAB markers aligned with their tactical drop vector instead of drifting into Russia
                tac_psi = math.radians(self._default_tactical_heading(obs.lat, obs.lon, "KAB"))
                xm_new[3] = tac_psi
                xm_new[4] = 0.0
            elif obs.heading_deg is None and is_ballistic and obs.from_zone:
                # Reinforce OTRK launch azimuth when source omits heading
                tac_psi = math.radians(self._default_tactical_heading(obs.lat, obs.lon, "BALLISTIC", from_zone=obs.from_zone))
                d_psi = (tac_psi - xm_new[3] + math.pi) % (2.0 * math.pi) - math.pi
                xm_new[3] = (xm_new[3] + 0.25 * d_psi) % (2.0 * math.pi)

            if obs.speed_kmh is not None and obs.speed_kmh > 20.0 and not suppress_vel_hdg:
                w_spd = 0.5 * vel_hdg_scale
                if is_ballistic:
                    clamped_obs_kms = float(np.clip(float(obs.speed_kmh), BALLISTIC_MIN_KMH, BALLISTIC_MAX_KMH)) / 3600.0
                    xm_new[2] = (1.0 - w_spd) * xm_new[2] + w_spd * clamped_obs_kms
                else:
                    xm_new[2] = (1.0 - w_spd) * xm_new[2] + w_spd * (float(obs.speed_kmh) / 3600.0)
            elif (trk.threat_class == "UAV_JET" or obs.threat_class == "UAV_JET") and xm_new[2] * 3600.0 < 260.0:
                xm_new[2] = 420.0 / 3600.0

            if is_ballistic:
                xm_new[2] = float(np.clip(xm_new[2], BALLISTIC_MIN_KMH / 3600.0, BALLISTIC_MAX_KMH / 3600.0))
                xm_new[4] = 0.0
            elif m in (0, 2):
                xm_new[4] = 0.0

            # Joseph-form covariance update: P = (I - K H) P (I - K H)^T + K R K^T
            IKH = I5.copy()
            IKH[:, :2] -= K
            R2 = np.diag([sig_r2_eff, sig_r2_eff])
            Pm_new = IKH @ Pm @ IKH.T + K @ R2 @ K.T
            Pm_new = 0.5 * (Pm_new + Pm_new.T)
            # Ensure positive diagonal floor
            np.fill_diagonal(Pm_new, np.maximum(np.diag(Pm_new), np.array([1.0, 1.0, 1e-6, 1e-4, 1e-8])))

            trk.mode_states[m] = xm_new
            trk.mode_covs[m] = Pm_new

        trk.gross_streak = (trk.gross_streak + 1) if gross_now else 0

        if is_ballistic:
            likelihoods[2] = 0.0
            post_mu = trk.mode_probs * likelihoods
            post_mu[2] = 0.0
            s_mu = float(np.sum(post_mu))
            trk.mode_probs = (post_mu / s_mu) if s_mu > 1e-12 else np.array([0.96, 0.04, 0.00], dtype=np.float64)
        elif obs.kinematic_hint not in ("TURN", "LOITER"):
            post_mu = trk.mode_probs * likelihoods
            s_mu = float(np.sum(post_mu))
            trk.mode_probs = (post_mu / s_mu) if s_mu > 1e-12 else np.array([0.80, 0.15, 0.05])

        X_comb, P_comb = self._combine_imm_estados(
            trk.mode_states[None, :, :], trk.mode_covs[None, :, :, :], trk.mode_probs[None, :]
        )
        xc = X_comb[0]
        pc = P_comb[0]
        lat_new, lon_new = enu_to_wgs84(float(xc[0]), float(xc[1]))

        trk.lat = lat_new
        trk.lon = lon_new
        if is_ballistic:
            trk.ground_speed_kmh = float(np.clip(xc[2] * 3600.0, BALLISTIC_MIN_KMH, BALLISTIC_MAX_KMH))
            trk.turn_rate_rad_s = 0.0
        else:
            trk.ground_speed_kmh = float(max(25.0, xc[2] * 3600.0))
            trk.turn_rate_rad_s = float(xc[4])
        trk.course_deg = float(math.degrees(xc[3]) % 360.0)
        trk.imm_mode = MODE_NAMES[int(np.argmax(trk.mode_probs))]
        trk.last_updated_ts = max(trk.last_updated_ts, obs.timestamp)
        trk.last_seen_ts = max(trk.last_seen_ts, obs.timestamp)
        trk.last_obs_lat = obs.lat
        trk.last_obs_lon = obs.lon
        trk.hit_updates += 1
        trk.missed_updates = 0
        trk.count = max(trk.count, obs.target_count)
        if obs.raw_text:
            trk.locality = obs.raw_text
        if obs.threat_class == "UAV_JET" or (obs.subkind or "").lower() in ("drone_jet", "banderol") or (obs.speed_kmh is not None and obs.speed_kmh >= 280.0 and trk.threat_class == "UAV_PROP"):
            trk.threat_class = "UAV_JET"
            trk.subkind = obs.subkind if obs.subkind in ("banderol", "drone_jet") else "drone_jet"
        elif obs.threat_class == "UAV_PROP" and trk.threat_class != "UAV_JET" and (obs.speed_kmh is None or obs.speed_kmh < 260.0):
            trk.threat_class = "UAV_PROP"
            trk.subkind = obs.subkind or "drone_piston"
        elif obs.threat_class == "UAV_PROP" and trk.threat_class == "UAV_JET" and obs.source_id == "sensor_ext_1" and obs.speed_kmh is not None and obs.speed_kmh < 230.0:
            trk.threat_class = "UAV_PROP"
            trk.subkind = obs.subkind or "drone_piston"
        elif obs.is_decoy and trk.threat_class != "UAV_JET":
            trk.subkind = "decoy"
        if trk.status == "TENTATIVE" and (trk.hit_updates >= 2 or obs.source_trust >= 0.80):
            trk.status = "CONFIRMED"
        elif trk.status == "COASTING":
            trk.status = "CONFIRMED"
        if obs.kinematic_hint == "NEUTRALIZED":
            trk.status = "DELETED"

        try:
            trk.h3_cell = h3.latlng_to_cell(trk.lat, trk.lon, 6)
        except Exception:
            pass

        maj, minr, ang = self.extract_ellipse_km(pc[:2, :2])
        trk.cov_major_km = maj
        trk.cov_minor_km = minr
        trk.cov_angle_deg = ang

    @staticmethod
    def _combined_prior(trk: Track) -> Tuple[np.ndarray, float]:
        """Mode-probability-weighted prior position (km) and circular-mean course psi (rad)."""
        mu = trk.mode_probs
        pos = np.einsum("m,mk->k", mu, trk.mode_states[:, :2])
        psi = math.atan2(float(np.dot(mu, np.sin(trk.mode_states[:, 3]))),
                         float(np.dot(mu, np.cos(trk.mode_states[:, 3]))))
        return pos, psi

    def _apply_oosm_guard(
        self, trk: Track, zx: float, zy: float, sig_r2: float, obs: Observation
    ) -> Tuple[float, float]:
        """
        Out-of-sequence guard (v9.2). Detects a fix lying 2..18 km BEHIND the extrapolated
        position (|cross| <= 6 km). The first such fix (oosm_streak == 1) is treated as a late
        publication: the along-track component is discarded and only 0.5 * cross is applied.
        Two or more consecutive backward fixes (oosm_streak >= 2) are treated as a real
        reversal / braking and applied in full.
        """
        if obs.kinematic_hint in ("TURN", "LOITER") or trk.hit_updates < 2:
            trk.oosm_streak = 0
            return zx, zy
        pos, psi = self._combined_prior(trk)
        dp = np.array([zx - pos[0], zy - pos[1]], dtype=np.float64)
        u = np.array([math.sin(psi), math.cos(psi)])
        n = np.array([math.cos(psi), -math.sin(psi)])
        along = float(dp @ u)
        cross = float(dp @ n)
        if OOSM_ALONG_MIN_KM <= along < OOSM_ALONG_MAX_KM and abs(cross) <= OOSM_CROSS_MAX_KM:
            trk.oosm_streak += 1
            if trk.oosm_streak == 1:
                z_new = pos + OOSM_CROSS_GAIN * cross * n
                return float(z_new[0]), float(z_new[1])
            return zx, zy  # streak >= 2: genuine reversal -> full update
        trk.oosm_streak = 0
        return zx, zy

    def _huber_r_multiplier(self, trk: Track, zx: float, zy: float, sig_r2: float) -> Tuple[float, float]:
        """
        Huber weight w = gamma / d_M (d_M > gamma), R_eff = R / w^2 = R * (d_M/gamma)^2,
        capped at HUBER_MAX_R_MULT. d_M is measured against the IMM-combined prior
        (mode-probability-weighted mean + spread-of-means covariance). A best-fitting-mode rule
        would let the wide LOITER covariance 'explain' any outlier and disable the filter.
        """
        X_c, P_c = self._combine_imm_estados(
            trk.mode_states[None, :, :], trk.mode_covs[None, :, :, :], trk.mode_probs[None, :]
        )
        S = P_c[0, :2, :2] + np.diag([sig_r2, sig_r2])
        det_S = float(S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0])
        if det_S <= 1e-9 or not np.isfinite(det_S):
            return 1.0, 0.0
        y = np.array([zx - X_c[0, 0], zy - X_c[0, 1]])
        S_inv = np.array([[S[1, 1], -S[0, 1]], [-S[1, 0], S[0, 0]]]) / det_S
        d_m = math.sqrt(max(0.0, float(y @ S_inv @ y)))
        if d_m <= HUBER_GAMMA:
            return 1.0, d_m
        return float(min(HUBER_MAX_R_MULT, (d_m / HUBER_GAMMA) ** 2)), d_m

    @staticmethod
    def extract_ellipse_km(cov_2x2: np.ndarray) -> Tuple[float, float, float]:
        """
        Extracts 95% confidence Mahalanobis ellipse semi-axes (major_km, minor_km)
        and orientation angle (degrees CW from North) with strict numerical guards.
        """
        if not np.all(np.isfinite(cov_2x2)):
            return 8.0, 5.0, 0.0
        vals, vecs = np.linalg.eigh(cov_2x2)
        vals = np.clip(vals, 0.5, 900.0)
        # chi2(df=2, 95%) = 5.991 -> scale factor sqrt(5.991) = 2.4477
        scale = 2.4477
        minor_km = float(np.clip(scale * math.sqrt(float(vals[0])), 2.0, 45.0))
        major_km = float(np.clip(scale * math.sqrt(float(vals[1])), minor_km, 60.0))
        v_maj = vecs[:, 1]
        # Angle clockwise from North (py is North, px is East)
        angle_deg = (math.degrees(math.atan2(float(v_maj[0]), float(v_maj[1]))) + 360.0) % 360.0
        return round(major_km, 2), round(minor_km, 2), round(angle_deg, 1)

    @staticmethod
    def compute_ballistic_cone(
        origin_lat: float, origin_lon: float, course_deg: float, range_km: float = 280.0, half_angle_deg: float = 18.0
    ) -> List[Tuple[float, float]]:
        """
        Generates a polygonal threat cone for BALLISTIC missile launches (Iskander-M / Kinzhal).
        """
        pts = [(origin_lat, origin_lon)]
        ox, oy = wgs84_to_enu(origin_lat, origin_lon)
        for a in np.linspace(course_deg - half_angle_deg, course_deg + half_angle_deg, 9):
            rad = math.radians(float(a))
            px = ox + range_km * math.sin(rad)
            py = oy + range_km * math.cos(rad)
            pts.append(enu_to_wgs84(px, py))
        pts.append((origin_lat, origin_lon))
        return pts
