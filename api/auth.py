# -*- coding: utf-8 -*-
"""
api/auth.py - Multi-tier zero-trust API key authentication & rate limiting for bayes-track-core.
Direct localhost requests (without Cloudflare/reverse-proxy headers) bypass token checks.
External clients authenticate via X-API-Key, Bearer token, cookie, or ?api_key= query param.
"""
import os
import json
import secrets
import time
from collections import defaultdict
from typing import Annotated, Optional
from fastapi import Depends, Request, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials

API_KEYS_PATH = os.environ.get("BAYES_TRACK_API_KEYS_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "api_keys.example.json"))

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_auth = HTTPBearer(auto_error=False)

_keys_cache = {}
_keys_mtime = 0
_rate_limits = defaultdict(list)  # key -> list of request timestamps

PROXY_HEADERS = ("cf-connecting-ip", "cf-ray", "x-forwarded-for", "x-real-ip", "x-forwarded-host", "x-forwarded-proto")

def load_keys() -> dict:
    global _keys_cache, _keys_mtime
    if not os.path.exists(API_KEYS_PATH):
        return {}
    try:
        mtime = os.path.getmtime(API_KEYS_PATH)
        if mtime > _keys_mtime:
            with open(API_KEYS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                _keys_cache = data.get("keys", {})
                _keys_mtime = mtime
    except Exception:
        pass
    return _keys_cache


def match_api_key(token: Optional[str]) -> Optional[tuple[str, dict]]:
    """Constant-time API key lookup returning (matched_key, meta) iff key exists and is enabled."""
    if not token or not isinstance(token, str):
        return None
    keys = load_keys()
    matched_key: Optional[str] = None
    meta: Optional[dict] = None
    token_bytes = token.encode("utf-8", errors="ignore")
    for k, v in keys.items():
        if secrets.compare_digest(k.encode("utf-8", errors="ignore"), token_bytes):
            matched_key = k
            meta = v
    if not meta or not meta.get("enabled", True) or matched_key is None:
        return None
    return matched_key, meta

def is_direct_localhost(request_or_ws) -> bool:
    """
    Returns True ONLY if the socket originates from 127.0.0.1/::1,
    the Host header is localhost/127.0.0.1, AND no reverse-proxy headers are present.
    """
    client_host = request_or_ws.client.host if request_or_ws.client else ""
    if not (client_host in ("127.0.0.1", "::1", "localhost") or client_host.startswith("127.")):
        return False
    headers = request_or_ws.headers
    host_hdr = headers.get("host", "").split(":")[0].lower()
    if host_hdr not in ("127.0.0.1", "localhost", "::1", ""):
        return False
    for ph in PROXY_HEADERS:
        if ph in headers:
            return False
    return True

async def verify_api_key(
    request: Request,
    header_key: Optional[str] = Security(api_key_header),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(bearer_auth)
) -> dict:
    # 1. Direct Localhost auto-bypass (Telegram bot & internal CLI)
    if is_direct_localhost(request):
        return {
            "key": "internal_localhost",
            "owner": "Local Service (Telegram Bot / Local CLI)",
            "role": "internal",
            "rate_limit_rpm": 999999
        }

    # 2. Extract key from Header, Bearer, or Query param
    token = header_key
    if not token and bearer:
        token = bearer.credentials
    if not token:
        token = request.query_params.get("api_key")
    if not token:
        token = request.cookies.get("rbc_api_key")

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide via X-API-Key header, Authorization: Bearer, or ?api_key= query parameter."
        )

    matched = match_api_key(token)
    if matched is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or deactivated API key."
        )
    matched_key, meta = matched

    # 3. Sliding-window rate limiter (60s window) with bounded state pruning
    now = time.time()
    rpm_limit = meta.get("rate_limit_rpm", 120)
    if len(_rate_limits) > 256:
        stale_keys = [rk for rk, ts_list in _rate_limits.items() if not ts_list or (now - ts_list[-1] >= 60)]
        for rk in stale_keys:
            _rate_limits.pop(rk, None)
    timestamps = [t for t in _rate_limits[matched_key] if now - t < 60]
    if len(timestamps) >= rpm_limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded ({rpm_limit} requests/min). Retry later."
        )
    timestamps.append(now)
    _rate_limits[matched_key] = timestamps

    return {
        "key": matched_key,
        "owner": meta.get("owner", "External Client"),
        "role": meta.get("role", "readonly"),
        "rate_limit_rpm": rpm_limit
    }


async def require_write_access(
    client_info: Annotated[dict, Depends(verify_api_key)],
) -> dict:
    """Enforces write/mutation authorization (rejects 'readonly' guest keys with 403 Forbidden)."""
    role = str(client_info.get("role") or "readonly").lower()
    if role not in ("admin", "master", "internal", "operator", "bridge"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Write access denied: API key has read-only role.",
        )
    return client_info


AuthClientDep = Annotated[dict, Depends(verify_api_key)]
WriteClientDep = Annotated[dict, Depends(require_write_access)]
