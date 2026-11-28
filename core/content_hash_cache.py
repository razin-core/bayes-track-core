# -*- coding: utf-8 -*-
"""
core/content_hash_cache.py - Content-hash file cache (SHA-256) for static GeoJSON,
frontline cartography, and dictionary assets.

Follows the Content-Hash File Cache Pattern:
- 64 KB chunked SHA-256 hashing (path-independent, auto-invalidating on content change)
- Frozen, slotted dataclass cache entries
- Pure extraction separated from cache layer
"""
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Union

_HASH_CHUNK_SIZE = 65536  # 64 KB chunks
_SMALL_FILE_HASH_THRESHOLD_BYTES = _HASH_CHUNK_SIZE  # 64 KB: hash directly in a single chunk
_RACY_MTIME_WINDOW_NS = 100_000_000  # 100 ms coarse-clock mtime race guard (racy-git style)


@dataclass(frozen=True, slots=True)
class FileCacheEntry:
    file_hash: str
    source_path: str
    mtime_ns: int
    size_bytes: int
    payload: Any
    loaded_ts: float
    ctime_ns: int = 0
    inode: int = 0


# Keyed by canonical path -> last observed FileCacheEntry, plus hash -> payload for move/rename hits
_PATH_INDEX: Dict[str, FileCacheEntry] = {}
_HASH_STORE: Dict[str, Any] = {}
_CACHE_STATS: Dict[str, int] = {"hits": 0, "misses": 0}


def compute_file_hash(path: Union[str, Path]) -> str:
    """Computes SHA-256 hex digest of a file in 64 KB chunks."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    sha256 = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def _can_reuse_stat_fast_path(
    prev: Optional[FileCacheEntry],
    st: os.stat_result,
    now_ns: int,
) -> bool:
    """Returns True only when stat() metadata is safe to trust without rehashing."""
    if prev is None:
        return False
    if st.st_size <= _SMALL_FILE_HASH_THRESHOLD_BYTES:
        return False
    if abs(now_ns - st.st_mtime_ns) < _RACY_MTIME_WINDOW_NS:
        return False
    return (
        prev.mtime_ns == st.st_mtime_ns
        and prev.ctime_ns == st.st_ctime_ns
        and prev.size_bytes == st.st_size
        and prev.inode == st.st_ino
    )


def compute_bundle_hash(paths: Iterable[Union[str, Path]]) -> str:
    """Computes a combined SHA-256 digest across multiple files (missing files encoded deterministically)."""
    combined = hashlib.sha256()
    now_ns = time.time_ns()
    for raw_p in paths:
        p = Path(raw_p)
        if p.is_file():
            st = p.stat()
            prev = _PATH_INDEX.get(str(p))
            if _can_reuse_stat_fast_path(prev, st, now_ns):
                digest = prev.file_hash
            else:
                digest = compute_file_hash(p)
            combined.update(f"{p.name}:{digest};".encode("utf-8"))
        else:
            combined.update(f"{p.name}:missing;".encode("utf-8"))
    return combined.hexdigest()


def load_with_content_hash(
    path: Union[str, Path],
    loader: Callable[[Path], Any] = lambda p: json.loads(p.read_text(encoding="utf-8")),
    *,
    cache_enabled: bool = True,
) -> FileCacheEntry:
    """
    Service-layer wrapper: computes/reuses SHA-256 content hash and returns cached payload.
    Auto-invalidates immediately when file content changes; survives file move/rename via _HASH_STORE.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")

    if not cache_enabled:
        f_hash = compute_file_hash(p)
        st = p.stat()
        return FileCacheEntry(
            file_hash=f_hash,
            source_path=str(p),
            mtime_ns=st.st_mtime_ns,
            size_bytes=st.st_size,
            payload=loader(p),
            loaded_ts=time.time(),
            ctime_ns=st.st_ctime_ns,
            inode=st.st_ino,
        )

    st = p.stat()
    key = str(p)
    prev = _PATH_INDEX.get(key)
    if _can_reuse_stat_fast_path(prev, st, time.time_ns()):
        _CACHE_STATS["hits"] += 1
        return prev

    f_hash = compute_file_hash(p)
    if f_hash in _HASH_STORE:
        _CACHE_STATS["hits"] += 1
        entry = FileCacheEntry(
            file_hash=f_hash,
            source_path=key,
            mtime_ns=st.st_mtime_ns,
            size_bytes=st.st_size,
            payload=_HASH_STORE[f_hash],
            loaded_ts=time.time(),
            ctime_ns=st.st_ctime_ns,
            inode=st.st_ino,
        )
        _PATH_INDEX[key] = entry
        return entry

    _CACHE_STATS["misses"] += 1
    data = loader(p)
    if len(_HASH_STORE) > 64:
        _HASH_STORE.clear()
    _HASH_STORE[f_hash] = data
    entry = FileCacheEntry(
        file_hash=f_hash,
        source_path=key,
        mtime_ns=st.st_mtime_ns,
        size_bytes=st.st_size,
        payload=data,
        loaded_ts=time.time(),
        ctime_ns=st.st_ctime_ns,
        inode=st.st_ino,
    )
    _PATH_INDEX[key] = entry
    return entry


def get_content_cache_stats() -> Dict[str, Any]:
    hits = _CACHE_STATS["hits"]
    misses = _CACHE_STATS["misses"]
    total = hits + misses
    return {
        "hits": hits,
        "misses": misses,
        "entries": len(_HASH_STORE),
        "hit_rate": round(hits / total, 4) if total > 0 else 1.0,
    }
