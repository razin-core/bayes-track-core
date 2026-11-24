# bayes-track-core

Sanitized, domain-agnostic mathematical and systems core extracted from a private real-time multi-source OSINT geospatial tracking & risk-estimation engine.

> **OPSEC & Scope Note:** This public repository intentionally contains **only** the pure tracking math, evidence fusion, caching, zero-trust API auth, and synthetic tests. Live ingestion feeds, operational gazetteers, real coordinates, and private keys are excluded.

## Architecture Overview

```mermaid
flowchart LR
    A["Async Text / Telemetry Ingestion"] --> B["NLP Normalization & Gazetteer"]
    B --> C["Batch IMM-EKF Tracker\n(core/imm_ekf.py)"]
    B --> D["Yager-Dempster Evidence Fusion\n(core/yager_dempster.py)"]
    C --> E["Read-Only SQLite WAL + Cache\n(core/content_hash_cache.py)"]
    D --> E
    E --> F["Zero-Trust FastAPI + WebSocket\n(api/auth.py)"]
    F --> G["MapLibre GL Real-Time Client"]
```

## Included Modules

- **`core/imm_ekf.py`** — Vectorized Interacting Multiple Model Extended Kalman Filter (`BatchIMMEKF`) with:
  - Huber-style robust outlier weighting (`+15 km` lateral spike rejection without destabilizing course);
  - Out-of-Sequence Measurement (OOSM) backward-fix guard (ignores single stale/delayed behind-track fixes while still following real `180°` reversals and `90°` turns);
  - Cross-wind drift propagation during `COASTING` under degraded navigation.
- **`core/models.py`** — WGS84 ↔ local tangent-plane ENU coordinate transforms (`wgs84_to_enu`, `enu_to_wgs84`) and typed `Observation` / `Track` dataclasses.
- **`core/yager_dempster.py`** — Yager-modified Dempster-Shafer belief fusion with echo suppression and novelty discounting for correlated noisy observers.
- **`core/content_hash_cache.py`** — Path-independent SHA-256 content-hash cache for expensive parsing/inference stages.
- **`api/auth.py`** — Zero-trust FastAPI dependency (`verify_api_key`) featuring:
  - Strict reverse-proxy header detection (`CF-Connecting-IP`, `CF-Ray`, `X-Forwarded-For`, etc.) so proxied requests can never spoof `localhost`;
  - Constant-time key matching (`secrets.compare_digest`) with hot-reloading by `mtime`;
  - Bounded sliding-window per-key rate limiter (`HTTP 429`).

## Quickstart & Synthetic Test Suite

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -v
```

All tests run on synthetic coordinates (`(49.0, 30.0)`) and temporary JSON keyfiles — no external services or network access required.
