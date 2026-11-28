# -*- coding: utf-8 -*-
"""Synthetic pytest suite for bayes-track-core (IMM-EKF, ContentHashCache, Zero-Trust Auth)."""
import json
import math
import os
import numpy as np
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from core.content_hash_cache import compute_bundle_hash, load_with_content_hash
import core.content_hash_cache as cache_mod
from core.imm_ekf import BatchIMMEKF
from core.models import Observation, SOURCE_SIGMA_R_KM, Track, enu_to_wgs84, wgs84_to_enu
from core.yager_dempster import YagerDempsterFusion, compute_echo_novelty_factor
import api.auth as auth_mod

V_KMS = 185.0 / 3600.0
T0 = 1_800_000_000.0
START = (49.0, 30.0)


def _obs_at(ts: float, px: float, py: float, hdg=None) -> Observation:
    lat, lon = enu_to_wgs84(px, py)
    return Observation(
        obs_id=f"o{ts}",
        timestamp=ts,
        lat=lat,
        lon=lon,
        heading_deg=hdg,
        speed_kmh=None,
        uncertainty_km=2.0,
        source_trust=0.85,
    )


def _warmup(n: int = 6, hdg_deg: float = 270.0):
    ekf = BatchIMMEKF()
    px, py = wgs84_to_enu(*START)
    psi = math.radians(hdg_deg)
    trk = ekf.initialize_track_from_obs(_obs_at(T0, px, py, hdg_deg), "t1")
    ts = T0
    for _ in range(n):
        ts += 60.0
        px += V_KMS * 60.0 * math.sin(psi)
        py += V_KMS * 60.0 * math.cos(psi)
        ekf.predict_batch([trk], ts)
        ekf.update_track_with_obs(trk, _obs_at(ts, px, py, hdg_deg))
    return ekf, trk, ts, px, py


def _pos(trk) -> np.ndarray:
    return np.array(wgs84_to_enu(trk.lat, trk.lon))


def _dcourse(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def test_imm_lateral_outlier_rejected():
    ekf, trk, ts, px, py = _warmup()
    c0 = trk.course_deg
    ts += 60.0
    px += -V_KMS * 60.0
    ekf.predict_batch([trk], ts)
    ekf.update_track_with_obs(trk, _obs_at(ts, px, py + 15.0, None))
    assert _dcourse(trk.course_deg, c0) <= 5.0


def test_imm_single_backward_oosm_no_rollback():
    ekf, trk, ts, px, py = _warmup()
    ts += 60.0
    px += -V_KMS * 60.0
    ekf.predict_batch([trk], ts)
    ekf.update_track_with_obs(trk, _obs_at(ts, px + 8.0, py, None))
    along_err = _pos(trk)[0] - px
    assert along_err <= 1.0
    assert _dcourse(trk.course_deg, 270.0) <= 5.0
    assert getattr(trk, "oosm_streak", None) == 1


def test_imm_follows_real_180_reversal():
    ekf, trk, ts, px, py = _warmup()
    for _ in range(6):
        ts += 60.0
        px += V_KMS * 60.0
        ekf.predict_batch([trk], ts)
        ekf.update_track_with_obs(trk, _obs_at(ts, px, py, None))
    err = float(np.linalg.norm(_pos(trk) - np.array([px, py])))
    assert err <= 8.0
    assert _dcourse(trk.course_deg, 90.0) <= 60.0


def test_imm_follows_real_90_turn():
    ekf, trk, ts, px, py = _warmup()
    for _ in range(5):
        ts += 60.0
        py += -V_KMS * 60.0
        ekf.predict_batch([trk], ts)
        ekf.update_track_with_obs(trk, _obs_at(ts, px, py, None))
    err = float(np.linalg.norm(_pos(trk) - np.array([px, py])))
    assert err <= 3.0
    assert _dcourse(trk.course_deg, 180.0) <= 35.0


def test_imm_coasting_crosswind_drift():
    res = {}
    for label, jam in (("nominal", 0.0), ("degraded", 0.8)):
        ekf, trk, ts, _, _ = _warmup(n=4)
        trk.status = "COASTING"
        trk.jamming_score = jam
        ekf2, trk2, *_ = _warmup(n=4)
        trk2.status = "CONFIRMED"
        tgt = ts + 120.0
        ekf.predict_batch([trk], tgt, wind_speed_kmh=36.0, wind_dir_deg=180.0)
        ekf2.predict_batch([trk2], tgt, wind_speed_kmh=36.0, wind_dir_deg=180.0)
        res[label] = float(np.linalg.norm(_pos(trk) - _pos(trk2)))
    assert res["degraded"] > res["nominal"] > 0.0


def test_content_hash_cache_roundtrip(tmp_path):
    src = tmp_path / "sample.json"
    src.write_text('{"score": 42}', encoding="utf-8")
    calls = {"n": 0}

    def _loader(p):
        calls["n"] += 1
        return json.loads(p.read_text(encoding="utf-8"))

    e1 = load_with_content_hash(src, loader=_loader)
    e2 = load_with_content_hash(src, loader=_loader)
    assert e1.payload == {"score": 42}
    assert e2.payload == {"score": 42}
    assert calls["n"] == 1  # second call served from SHA-256 content cache
    bundle_h1 = compute_bundle_hash([src])

    # Mutating content with identical byte length (13 bytes) invalidates immediately without sleep
    src.write_text('{"score": 99}', encoding="utf-8")
    bundle_h2 = compute_bundle_hash([src])
    assert bundle_h2 != bundle_h1

    e3 = load_with_content_hash(src, loader=_loader)
    assert e3.payload == {"score": 99}
    assert e3.file_hash != e1.file_hash
    assert calls["n"] == 2


def test_content_hash_cache_large_file_stat_fast_path(tmp_path, monkeypatch):
    large = tmp_path / "large.json"
    payload = {"blob": "x" * 70_000}
    large.write_text(json.dumps(payload), encoding="utf-8")
    # Age mtime by 2 seconds so the file is outside the 100 ms racy window
    old_ts = large.stat().st_mtime - 2.0
    os.utime(large, (old_ts, old_ts))

    calls = {"n": 0}

    def _loader(p):
        calls["n"] += 1
        return json.loads(p.read_text(encoding="utf-8"))

    e1 = load_with_content_hash(large, loader=_loader)
    assert calls["n"] == 1

    # On second read, stat() fast-path must serve the cached entry without rehashing disk
    def _forbid_rehash(_p):
        raise AssertionError("compute_file_hash should not be called on aged unchanged >64KB file")

    monkeypatch.setattr(cache_mod, "compute_file_hash", _forbid_rehash)
    e2 = load_with_content_hash(large, loader=_loader)
    assert e2 is e1
    assert calls["n"] == 1
    assert compute_bundle_hash([large]) == compute_bundle_hash([large])


def test_zero_trust_proxy_auth_and_rate_limit(tmp_path, monkeypatch):
    keys_file = tmp_path / "api_keys.json"
    keys_file.write_text(
        json.dumps(
            {
                "keys": {
                    "test_key_valid_123": {
                        "owner": "UnitTest",
                        "role": "readonly",
                        "rate_limit_rpm": 2,
                        "enabled": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(auth_mod, "API_KEYS_PATH", str(keys_file))
    auth_mod._keys_cache = {}
    auth_mod._keys_mtime = 0
    auth_mod._rate_limits.clear()

    app = FastAPI()

    @app.get("/protected")
    async def protected(meta=Depends(auth_mod.verify_api_key)):
        return {"owner": meta["owner"]}

    client = TestClient(app)

    # Proxied request without key must NEVER bypass auth -> 401
    r_no_key = client.get("/protected", headers={"CF-Connecting-IP": "203.0.113.10"})
    assert r_no_key.status_code == 401

    # Invalid key -> 403
    r_bad = client.get(
        "/protected",
        headers={"CF-Connecting-IP": "203.0.113.10", "X-API-Key": "wrong_key"},
    )
    assert r_bad.status_code == 403

    # Valid key -> 200 twice, then 429 on 3rd request within 60s window
    hdrs = {"CF-Connecting-IP": "203.0.113.10", "X-API-Key": "test_key_valid_123"}
    assert client.get("/protected", headers=hdrs).status_code == 200
    assert client.get("/protected", headers=hdrs).status_code == 200
    assert client.get("/protected", headers=hdrs).status_code == 429


def test_yager_dempster_normal_and_high_conflict():
    # Consistent threat observations increase m_threat with zero conflict
    m_t, m_c, m_u, k_low = YagerDempsterFusion.combine_two_bpa((0.7, 0.0, 0.3), (0.6, 0.0, 0.4))
    assert k_low == pytest.approx(0.0)
    assert m_t > 0.85
    assert m_t + m_c + m_u == pytest.approx(1.0)

    # Strong contradiction routes conflict K into ignorance m(Theta) without Zadeh singularity
    m_t2, m_c2, m_u2, k_high = YagerDempsterFusion.combine_two_bpa((0.95, 0.0, 0.05), (0.0, 0.95, 0.05))
    assert k_high == pytest.approx(0.9025)
    assert m_u2 > 0.90
    assert m_t2 + m_c2 + m_u2 == pytest.approx(1.0)


def test_yager_echo_novelty_suppression():
    # Correlated rapid repost (< 90s) is discounted to floor 0.20
    alpha_echo = compute_echo_novelty_factor(30.0, "obs_region_1", "obs_region_2", same_h3_or_locality=True)
    assert alpha_echo == pytest.approx(0.20)
    # Report outside 180s window has full novelty 1.0
    alpha_fresh = compute_echo_novelty_factor(240.0, "obs_region_1", "obs_region_2", same_h3_or_locality=True)
    assert alpha_fresh == pytest.approx(1.0)


def test_auth_ipv6_forwarded_and_fail_closed_keys(tmp_path, monkeypatch):
    keys_file = tmp_path / "api_keys.json"
    keys_file.write_text(
        json.dumps(
            {
                "keys": {
                    "valid_key_abc": {
                        "owner": "UnitTest",
                        "role": "readonly",
                        "rate_limit_rpm": 10,
                        "enabled": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(auth_mod, "API_KEYS_PATH", str(keys_file))
    auth_mod._keys_cache = {}
    auth_mod._keys_mtime = 0
    auth_mod._rate_limits.clear()

    app = FastAPI()

    @app.get("/protected")
    async def protected(meta=Depends(auth_mod.verify_api_key)):
        return {"owner": meta["owner"]}

    local_client = TestClient(app, client=("::1", 50000))

    # Direct IPv6 localhost with bracketed Host header bypasses key check
    r_ipv6 = local_client.get("/protected", headers={"Host": "[::1]:8000"})
    assert r_ipv6.status_code == 200
    assert r_ipv6.json()["owner"] == "Local Service (Internal CLI)"

    # RFC 7239 Forwarded header blocks localhost bypass -> 401
    r_fwd = local_client.get(
        "/protected",
        headers={"Host": "[::1]:8000", "Forwarded": "for=203.0.113.10"},
    )
    assert r_fwd.status_code == 401

    # Valid key succeeds initially, then corrupting api_keys.json fails closed -> 403
    ext_hdrs = {"Host": "api.example.com", "Forwarded": "for=203.0.113.10", "X-API-Key": "valid_key_abc"}
    assert local_client.get("/protected", headers=ext_hdrs).status_code == 200
    keys_file.write_text("{corrupted json", encoding="utf-8")
    assert local_client.get("/protected", headers=ext_hdrs).status_code == 403


def test_yager_interleaved_echo_suppression():
    # Interleaving an unrelated H3 cell must not defeat echo suppression for H3 cell A
    interleaved = [
        {"source_id": "obs_region_1", "timestamp": 1000.0, "base_trust": 0.70, "is_clear": False, "h3_cell": "h3_cell_a"},
        {"source_id": "sensor_ext_1", "timestamp": 1010.0, "base_trust": 0.40, "is_clear": False, "h3_cell": "h3_cell_b"},
        {"source_id": "obs_region_2", "timestamp": 1020.0, "base_trust": 0.70, "is_clear": False, "h3_cell": "h3_cell_a"},
    ]
    independent = [
        {"source_id": "obs_region_1", "timestamp": 1000.0, "base_trust": 0.70, "is_clear": False, "h3_cell": "h3_cell_a"},
        {"source_id": "sensor_ext_1", "timestamp": 1010.0, "base_trust": 0.40, "is_clear": False, "h3_cell": "h3_cell_b"},
        {"source_id": "obs_region_2", "timestamp": 1020.0, "base_trust": 0.70, "is_clear": False, "h3_cell": "h3_cell_c"},
    ]
    res_interleaved = YagerDempsterFusion.fuse_evidence_sequence(interleaved)
    res_independent = YagerDempsterFusion.fuse_evidence_sequence(independent)
    assert res_interleaved["m_threat"] < res_independent["m_threat"]


def test_imm_lazy_init_and_ctrv_large_omega_covariance():
    ekf = BatchIMMEKF()
    raw_trk = Track(track_uuid="raw", lat=START[0], lon=START[1], last_updated_ts=T0)
    px, py = wgs84_to_enu(*START)
    # Direct update without prior predict_batch must lazily initialize mode tensors
    ekf.update_track_with_obs(raw_trk, _obs_at(T0 + 10.0, px + 0.5, py + 0.5, 45.0))
    assert raw_trk.mode_states is not None
    assert np.all(np.isfinite(raw_trk.mode_covs))

    # Large turn rate CTRV prediction must keep covariance finite and symmetric
    raw_trk.mode_states[1, 4] = 0.03
    ekf.predict_batch([raw_trk], raw_trk.last_updated_ts + 60.0)
    P_ctrv = raw_trk.mode_covs[1]
    assert np.all(np.isfinite(P_ctrv))
    assert np.allclose(P_ctrv, P_ctrv.T, atol=1e-9)


def test_models_sanitized_and_observation_no_broken_properties():
    assert all(k.startswith(("sensor_", "feed_", "obs_")) for k in SOURCE_SIGMA_R_KM)
    assert all(not k.startswith("tg_") for k in SOURCE_SIGMA_R_KM)
    obs = Observation(obs_id="o1", timestamp=T0, lat=START[0], lon=START[1])
    for attr in ("course_rad", "ellipse_major_km", "ellipse_minor_km"):
        assert not hasattr(obs, attr)
