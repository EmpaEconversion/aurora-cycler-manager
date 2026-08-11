# Copyright © 2026, Empa.
"""Server-side cache of sample data.

Entries are keyed on file identity (sample, kind, mtime, size), and are shared
between users.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import TYPE_CHECKING

from aurora_cycler_manager.config import get_config
from aurora_cycler_manager.data_parse import (
    get_cycles_summary,
    get_cycling,
    get_cycling_shrunk,
    get_eis,
    get_sample_folder,
)

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl

logger = logging.getLogger(__name__)
CONFIG = get_config()

# Total bytes of cached frames before least-recently-used entries are dropped
MAX_BYTES = int(CONFIG.get("Plot cache MB", 2000)) * 1_000_000
# A frame larger than MAX_ENTRY_BYTES this is served but never cached
# Otherwise it can kick out everything else
MAX_ENTRY_BYTES = MAX_BYTES // 2
# If the data folder is a network share .stat() is costly. Cache the stat() for
# a few seconds so it does not get re-hit many times in one function
STAT_LIFETIME_S = 2.0

Stamp = tuple[int, int]
Key = tuple[str, str, Stamp]

_LOADERS = {
    "full": get_cycling,
    "shrunk": get_cycling_shrunk,
    "eis": get_eis,
    "cycles": get_cycles_summary,
}
_FILENAMES = {
    "full": ("full.{s}.parquet", "full.{s}.h5"),
    "shrunk": ("shrunk.{s}.parquet", "shrunk.{s}.h5"),
    "eis": ("eis.{s}.parquet",),
    "cycles": ("cycles.{s}.parquet", "cycles.{s}.json"),
}


def _locate(sample_id: str, kind: str) -> tuple[Path, Stamp] | None:
    """Find the file for this sample and kind.

    A successful stat proves existence and yields the identity in one network round trip.
    """
    folder = get_sample_folder(sample_id)
    for pattern in _FILENAMES[kind]:
        path = folder / pattern.format(s=sample_id)
        try:
            st = path.stat()
        except OSError:
            continue
        return path, (st.st_mtime_ns, st.st_size)
    return None


class _FrameCache:
    """LRU cache of dataframes."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self._entries: OrderedDict[Key, pl.DataFrame] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self._load_locks: dict[tuple[str, str], threading.Lock] = {}
        self._located: dict[tuple[str, str], tuple[float, tuple[Path, Stamp] | None]] = {}
        self.hits = 0
        self.misses = 0
        self.superseded = 0
        self.evicted = 0
        self.rejected = 0

    def locate(self, sample_id: str, kind: str) -> tuple[Path, Stamp] | None:
        """Return the file and its identity."""
        now = time.monotonic()
        cached = self._located.get((sample_id, kind))
        if cached and now - cached[0] < STAT_LIFETIME_S:
            return cached[1]
        found = _locate(sample_id, kind)
        self._located[sample_id, kind] = (now, found)
        return found

    def get(self, sample_id: str, kind: str, working_set: set[str] | None = None) -> pl.DataFrame | None:
        """Return the collected frame for this sample and kind, loading and caching it on miss.

        Pass the samples currently being plotted as working_set so a selection larger than
        the budget does not evict its own earlier frames on every pass.
        """
        found = self.locate(sample_id, kind)
        if found is None:
            return None
        _, stamp = found
        key = (sample_id, kind, stamp)

        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                self.hits += 1
                return self._entries[key]
            load_lock = self._load_locks.setdefault((sample_id, kind), threading.Lock())

        # Load outside the cache lock so a slow network read cannot block other sessions.
        # The per-sample lock stops two threads collecting the same file at once.
        with load_lock:
            with self._lock:
                if key in self._entries:
                    self._entries.move_to_end(key)
                    self.hits += 1
                    return self._entries[key]
            self.misses += 1
            started = time.perf_counter()
            df = _LOADERS[kind](sample_id)
            if df is None:
                return None
            logger.info(
                "Loaded %s %s: %d rows, %.1f MB, %.2f s",
                kind,
                sample_id,
                df.height,
                df.estimated_size() / 1e6,
                time.perf_counter() - started,
            )
            self._put(key, df, working_set)
            return df

    def _put(self, key: Key, df: pl.DataFrame, working_set: set[str] | None) -> None:
        """Insert a frame, dropping superseded stamps and then trimming to the byte budget."""
        sample_id, kind, _ = key
        size = df.estimated_size()
        with self._lock:
            # Older stamps of the same file can never be requested again, so drop them outright
            for stale in [k for k in self._entries if k[0] == sample_id and k[1] == kind and k != key]:
                self._bytes -= self._entries.pop(stale).estimated_size()
                self.superseded += 1
            if size > MAX_ENTRY_BYTES:
                logger.warning("%s %s is %.0f MB, too large to cache", kind, sample_id, size / 1e6)
                return
            self._entries[key] = df
            self._bytes += size
            while self._bytes > self.max_bytes:
                victim = next(iter(self._entries))
                # Evicting a frame the caller still needs this pass would thrash, so decline to
                # cache this one instead and let the earlier frames keep serving hits
                if victim == key or (working_set and victim[0] in working_set):
                    self._bytes -= self._entries.pop(key).estimated_size()
                    self.rejected += 1
                    return
                self._bytes -= self._entries.pop(victim).estimated_size()
                self.evicted += 1

    def drop_unused(self, keep: set[str]) -> None:
        """Drop frames for samples outside the given set, freeing memory ahead of LRU pressure."""
        with self._lock:
            for key in [k for k in self._entries if k[0] not in keep]:
                self._bytes -= self._entries.pop(key).estimated_size()
                self.evicted += 1

    def stats(self) -> dict:
        """Return current size and hit counters."""
        with self._lock:
            return {
                "entries": len(self._entries),
                "MB": round(self._bytes / 1e6, 1),
                "budget_MB": round(self.max_bytes / 1e6),
                "hits": self.hits,
                "misses": self.misses,
                "superseded": self.superseded,
                "evicted": self.evicted,
                "rejected": self.rejected,
            }

    def clear(self) -> None:
        """Empty the cache."""
        with self._lock:
            self._entries.clear()
            self._located.clear()
            self._bytes = 0


_CACHE = _FrameCache(MAX_BYTES)

get_frame = _CACHE.get
drop_unused = _CACHE.drop_unused
cache_stats = _CACHE.stats
clear_cache = _CACHE.clear


def get_cycling_frame(sample_id: str, *, compressed: bool, working_set: set[str] | None = None) -> pl.DataFrame | None:
    """Time series for a sample, preferring the shrunk file when asked for."""
    if compressed and (df := get_frame(sample_id, "shrunk", working_set)) is not None:
        return df
    return get_frame(sample_id, "full", working_set)
