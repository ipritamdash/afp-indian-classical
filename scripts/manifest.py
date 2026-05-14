"""Manifest schema for the AFP benchmark.

A single source of truth for what queries / references look like on disk.
All downstream code (per-system runners, scorer) reads these CSVs.

Reference manifest (one row per indexed library track):
    ref_id, corpus, source_track_id, audio_path, duration_sec, samplerate,
    channels, raagas, taalas, artists, works, role
        # role ∈ {"library_only", "library_and_test_source"}

Query manifest (one row per cut query clip):
    query_id, ref_id, corpus, audio_path, offset_sec, length_sec, seed
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

import pandas as pd


# Reproducibility — used as the base seed for any RNG that affects sampling
GLOBAL_SEED = 20260511


REF_COLUMNS = [
    "ref_id", "corpus", "source_track_id", "audio_path",
    "duration_sec", "samplerate", "channels",
    "raagas", "taalas", "artists", "works", "work_mbids", "role",
]

QUERY_COLUMNS = [
    "query_id", "ref_id", "corpus", "audio_path",
    "offset_sec", "length_sec", "seed",
]


@dataclass
class RefEntry:
    ref_id: str
    corpus: str
    source_track_id: str
    audio_path: str
    duration_sec: float
    samplerate: int
    channels: int
    raagas: str | None = None
    taalas: str | None = None
    artists: str | None = None
    works: str | None = None
    work_mbids: str | None = None
    role: str = "library_only"  # or library_and_test_source


@dataclass
class QueryEntry:
    query_id: str
    ref_id: str
    corpus: str
    audio_path: str
    offset_sec: float
    length_sec: float
    seed: int


def write_refs(rows: Iterable[RefEntry], path: Path) -> None:
    df = pd.DataFrame([asdict(r) for r in rows], columns=REF_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_queries(rows: Iterable[QueryEntry], path: Path) -> None:
    df = pd.DataFrame([asdict(q) for q in rows], columns=QUERY_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def read_refs(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_queries(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
