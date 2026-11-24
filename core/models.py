# -*- coding: utf-8 -*-
"""
core/models.py - Canonical data structures for RBC-C4ISR v2.1:
Observation, Track, SectorState, KillChainState, and local tangent plane (ENU)
coordinate transformations anchored at Ukraine reference meridian (49.5° N, 30.0° E).
"""
import config.settings  # Enforce single-thread BLAS
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Reference tangent plane origin for Ukraine (SDD Section 3.1: 49.5° N, 30.0° E)
ENU_LAT0_DEG = 49.5
ENU_LON0_DEG = 30.0
KM_PER_DEG_LAT = 111.13295
KM_PER_DEG_LON = 111.31949 * math.cos(math.radians(ENU_LAT0_DEG))


def wgs84_to_enu(lat: float, lon: float) -> Tuple[float, float]:
    """Converts WGS-84 (lat, lon) to local tangent plane ENU (px_east_km, py_north_km)."""
    px = (lon - ENU_LON0_DEG) * KM_PER_DEG_LON
    py = (lat - ENU_LAT0_DEG) * KM_PER_DEG_LAT
    return px, py


def enu_to_wgs84(px_km: float, py_km: float) -> Tuple[float, float]:
    """Converts local tangent plane ENU (px_east_km, py_north_km) back to WGS-84 (lat, lon)."""
    lat = ENU_LAT0_DEG + (py_km / KM_PER_DEG_LAT)
    lon = ENU_LON0_DEG + (px_km / KM_PER_DEG_LON)
    return lat, lon


def wgs84_to_enu_vec(lats: np.ndarray, lons: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized WGS-84 to ENU conversion for NumPy arrays."""
    px = (lons - ENU_LON0_DEG) * KM_PER_DEG_LON
    py = (lats - ENU_LAT0_DEG) * KM_PER_DEG_LAT
    return px, py


def enu_to_wgs84_vec(px: np.ndarray, py: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized ENU to WGS-84 conversion for NumPy arrays."""
    lats = ENU_LAT0_DEG + (py / KM_PER_DEG_LAT)
    lons = ENU_LON0_DEG + (px / KM_PER_DEG_LON)
    return lats, lons


def latlon_to_enu_km(lat: Any, lon: Any) -> Tuple[Any, Any]:
    """Universal scalar/vector alias for wgs84_to_enu / wgs84_to_enu_vec."""
    if isinstance(lat, np.ndarray) or isinstance(lon, np.ndarray):
        return wgs84_to_enu_vec(np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64))
    return wgs84_to_enu(float(lat), float(lon))


def enu_km_to_latlon(px_km: Any, py_km: Any) -> Tuple[Any, Any]:
    """Universal scalar/vector alias for enu_to_wgs84 / enu_to_wgs84_vec."""
    if isinstance(px_km, np.ndarray) or isinstance(py_km, np.ndarray):
        return enu_to_wgs84_vec(np.asarray(px_km, dtype=np.float64), np.asarray(py_km, dtype=np.float64))
    return enu_to_wgs84(float(px_km), float(py_km))



# Default cruise / ground-projection airspeeds by threat class (km/h)
DEFAULT_AIRSPEED_KMH: Dict[str, float] = {
    "UAV_PROP": 185.0,
    "DECOY": 165.0,
    "UAV_JET": 460.0,
    "CRUISE_MISSILE": 820.0,
    "BALLISTIC": 4200.0,
    "KAB": 650.0,
}

# Default measurement error standard deviation sigma_r (km) by source (SDD Section 3.2)
SOURCE_SIGMA_R_KM: Dict[str, float] = {
    "api_mapa_ua": 5.0,
    "api_neptun_threats": 8.0,
    "api_neptun_messages": 10.0,
    "tg_kpszsu": 12.0,
    "tg_vanek_nikolaev": 9.0,
    "tg_war_monitor": 10.0,
    "tg_ppo_vinnytsia": 6.0,
    "tg_vinnytsiarealll": 6.0,
    "tg_zhytomyr_operative": 6.0,
}


@dataclass
class Observation:
    obs_id: str
    timestamp: float
    lat: float
    lon: float
    source_id: str = "api_mapa_ua"
    threat_class: str = "UAV_PROP"
    subkind: Optional[str] = None        # e.g. 'decoy', 'drone_piston', 'drone_jet', 'kalibr', 'x101'
    heading_deg: Optional[float] = None  # [0..360) clockwise from North
    speed_kmh: Optional[float] = None
    uncertainty_km: Optional[float] = None
    target_count: int = 1
    kinematic_hint: str = "TRANSIT"      # 'TRANSIT', 'TURN', 'LOITER', 'NEUTRALIZED'
    source_trust: float = 0.85
    h3_r6: Optional[str] = None
    raw_text: str = ""
    from_zone: Optional[str] = None
    trail: List[List[float]] = field(default_factory=list)

    @property
    def is_decoy(self) -> bool:
        return (self.subkind or "").lower() == "decoy" or self.threat_class == "DECOY"

    @property
    def x_km(self) -> float:
        return (self.lon - ENU_LON0_DEG) * KM_PER_DEG_LON

    @property
    def y_km(self) -> float:
        return (self.lat - ENU_LAT0_DEG) * KM_PER_DEG_LAT

    @property
    def course_rad(self) -> float:
        return math.radians(self.course_deg % 360.0)

    @property
    def ellipse_minor_km(self) -> float:
        return float(self.cov_minor_km * self.ellipse_scale)

    @property
    def ellipse_major_km(self) -> float:
        return float(self.cov_major_km * self.ellipse_scale)


    @property
    def sigma_r_km(self) -> float:
        if self.uncertainty_km is not None and self.uncertainty_km > 0:
            return max(2.5, min(25.0, float(self.uncertainty_km)))
        return SOURCE_SIGMA_R_KM.get(self.source_id, 8.0)


@dataclass
class Track:
    track_uuid: str
    threat_class: str = "UAV_PROP"
    subkind: Optional[str] = None
    status: str = "TENTATIVE"            # 'TENTATIVE', 'CONFIRMED', 'COASTING', 'DELETED'
    imm_mode: str = "CV"                 # 'CV', 'CTRV', 'LOITER'
    count: int = 1
    lat: float = 49.2331
    lon: float = 28.4682
    ground_speed_kmh: float = 185.0
    course_deg: float = 300.0
    turn_rate_rad_s: float = 0.0
    cov_major_km: float = 8.0
    cov_minor_km: float = 5.0
    cov_angle_deg: float = 0.0
    h3_cell: str = ""
    locality: str = ""
    created_ts: float = 0.0
    last_updated_ts: float = 0.0
    last_seen_ts: float = 0.0
    last_obs_lat: float = 0.0
    last_obs_lon: float = 0.0
    hit_updates: int = 1
    missed_updates: int = 0
    ellipse_scale: float = 1.0
    belief_threat: float = 0.85
    conflict_mass: float = 0.0
    projected_path_h3: List[str] = field(default_factory=list)
    sector_eta_min: Dict[str, Optional[int]] = field(default_factory=dict)
    sources_history: List[Tuple[str, float, str]] = field(default_factory=list)  # (source_id, ts, h3_cell)
    trail: List[List[float]] = field(default_factory=list)
    # Internal IMM-EKF 3-mode state tensors: (3, 5) and (3, 5, 5) + mode probabilities (3,)
    mode_states: Optional[np.ndarray] = None
    mode_covs: Optional[np.ndarray] = None
    mode_probs: Optional[np.ndarray] = None
    # Robust-filter state (v9.2): consecutive backward (OOSM-like) observations & sector EW score
    oosm_streak: int = 0
    gross_streak: int = 0
    jamming_score: float = 0.0

    @property
    def is_decoy(self) -> bool:
        return (self.subkind or "").lower() == "decoy" or self.threat_class == "DECOY"

    @property
    def x_km(self) -> float:
        return (self.lon - ENU_LON0_DEG) * KM_PER_DEG_LON

    @property
    def y_km(self) -> float:
        return (self.lat - ENU_LAT0_DEG) * KM_PER_DEG_LAT

    @property
    def course_rad(self) -> float:
        return math.radians(self.course_deg % 360.0)

    @property
    def ellipse_minor_km(self) -> float:
        return float(self.cov_minor_km * self.ellipse_scale)

    @property
    def ellipse_major_km(self) -> float:
        return float(self.cov_major_km * self.ellipse_scale)


    @property
    def lethality_weight(self) -> float:
        """Decoy drones (Gerbera/Parodiya) carry reduced kinetic lethality weight while still triggering PVO."""
        return 0.42 if self.is_decoy else 1.00

    def to_telemetry_dict(self) -> Dict[str, Any]:
        return {
            "track_uuid": self.track_uuid,
            "threat_class": self.threat_class,
            "subkind": self.subkind or ("decoy" if self.is_decoy else "standard"),
            "is_decoy": self.is_decoy,
            "status": self.status,
            "imm_mode": self.imm_mode,
            "count": self.count,
            "locality": self.locality or self.h3_cell,
            "lat": round(float(self.lat), 5),
            "lon": round(float(self.lon), 5),
            "ground_speed_kmh": round(float(self.ground_speed_kmh), 1),
            "course_deg": round(float(self.course_deg) % 360.0, 1),
            "turn_rate_rad_s": round(float(self.turn_rate_rad_s), 5),
            "ellipse": {
                "major_km": round(float(self.cov_major_km * self.ellipse_scale), 2),
                "minor_km": round(float(self.cov_minor_km * self.ellipse_scale), 2),
                "angle_deg": round(float(self.cov_angle_deg) % 360.0, 1),
            },
            "projected_path_h3": list(self.projected_path_h3),
            "sector_eta_min": dict(self.sector_eta_min),
            "trail": list(self.trail),
            "jamming_score": round(float(self.jamming_score), 3),
            "oosm_streak": int(self.oosm_streak),
            "gross_streak": int(self.gross_streak),
        }


@dataclass
class KillChainState:
    tu95_stage: str = "STAGE_0_GROUND"
    mig31k_active: bool = False
    mig31k_takeoff_ts: Optional[float] = None
    kalibr_sea_salvo: int = 0
    countdown_to_ua_border_min: Optional[int] = None
    detected_callsigns: List[str] = field(default_factory=list)
    active_callsigns: List[str] = field(default_factory=list)
    active_airfields: List[str] = field(default_factory=list)
    launch_windows: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tu95_stage": self.tu95_stage,
            "mig31k_active": self.mig31k_active,
            "kalibr_sea_salvo": self.kalibr_sea_salvo,
            "countdown_to_ua_border_min": self.countdown_to_ua_border_min,
            "active_callsigns": list(self.active_callsigns),
            "active_airfields": list(self.active_airfields),
            "launch_windows": list(self.launch_windows),
        }


@dataclass
class HorizonRisk:
    prob: int
    level: str
    red_prob: Optional[int] = None

    @classmethod
    def from_prob(cls, prob_pct: int, red_prob_pct: Optional[int] = None) -> "HorizonRisk":
        p = max(1, min(99, int(prob_pct)))
        rp = max(1, min(p, int(red_prob_pct))) if red_prob_pct is not None else None
        if p >= 70:
            lvl = "RED"
        elif p >= 35:
            lvl = "ORANGE"
        else:
            lvl = "GREEN"
        return cls(prob=p, level=lvl, red_prob=rp)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"prob": self.prob, "level": self.level}
        if self.red_prob is not None:
            out["red_prob"] = self.red_prob
        return out


@dataclass
class ClearanceEstimate:
    expected_clear_eta_min: Optional[int] = None
    upper_clear_eta_min: Optional[int] = None
    confidence_pct: Optional[int] = None
    mode: str = "NO_ACTIVE_ALERT"
    bifurcation_node_id: Optional[str] = None
    bifurcation_label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_clear_eta_min": self.expected_clear_eta_min,
            "upper_clear_eta_min": self.upper_clear_eta_min,
            "confidence_pct": self.confidence_pct,
            "mode": self.mode,
            "bifurcation_node_id": self.bifurcation_node_id,
            "bifurcation_label": self.bifurcation_label,
        }



@dataclass
class SectorState:
    sector_id: str
    alert_active: bool = False
    alert_started_ts: Optional[float] = None
    threat_summary_code: str = "TACTICAL_CLEAN"
    horizons: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    clearance_weibull: Dict[str, Any] = field(default_factory=dict)
    gates_risk: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_active": self.alert_active,
            "threat_summary_code": self.threat_summary_code,
            "horizons": self.horizons,
            "clearance_weibull": self.clearance_weibull,
            "gates_risk": self.gates_risk,
        }
