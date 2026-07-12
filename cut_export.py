"""v2 cut-export serializer (CineScript S3, design §3).

Stdlib only. Assembles the ratified document from a list of CutEvents plus
source / run / analysis metadata, GENERATES the legacy ``cuts`` array as a
projection of the events (events are the single source of truth, design §2),
and enforces the §3.3 validity rules -- including a self-test that the
projection matches the events -- before returning.

`cut_times.py` wiring is a later slice (S3-E6); this module is the library.
"""
from __future__ import annotations

import json

from cut_events import (
    ExportValidationError, event_to_dict, reject_nan_and_null, validate_event_dict,
)
from cut_events import _finite as _finite

FORMAT = "film-cut-analysis/cut-events"
SCHEMA_VERSION = 2
VALID_STATUS = ("complete", "partial", "failed")


def _drop_none(d: dict) -> dict:
    """Omission, not null (design §3.2): strip keys whose value is None."""
    return {k: v for k, v in d.items() if v is not None}


def build_document(events, *, source, run, analysis, fps, video, detector) -> dict:
    """Build the §3 v2 document from a list of CutEvents, then validate it through the
    SAME path used on read (`validate_document`) -- write and read share one validator
    (amendment A3.2). ``source`` / ``run`` are provenance dicts (None values omitted);
    ``analysis`` supplies status / coverage / warnings; ``event_count`` and the legacy
    ``cuts`` projection are derived here, not trusted. Raises `ExportValidationError`."""
    cut_events = [event_to_dict(e) for e in events]
    cuts = [ce["time"] for ce in cut_events]           # projection derived from the events
    analysis_block = {
        "status": analysis.get("status"),
        "coverage": [dict(s) for s in (analysis.get("coverage") or [])],
        "event_count": len(events),
        "warnings": list(analysis.get("warnings") or []),
    }
    doc = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "source": _drop_none(dict(source)),
        "run": _drop_none(dict(run)),
        "analysis": analysis_block,
        "cut_events": cut_events,
        "cuts": cuts,
        # trailing legacy keys, retained verbatim for old readers (design §3):
        "fps": fps,
        "video": video,
        "detector": detector,
    }
    validate_document(doc)                             # one validation path (write == read)
    return doc


def serialize(events, *, source, run, analysis, fps, video, detector, indent=2) -> str:
    """`build_document` then `json.dumps` with ``allow_nan=False`` (amendment A3.2 --
    NaN / Infinity can never reach the file). The string a v2 export writes to disk."""
    doc = build_document(events, source=source, run=run, analysis=analysis,
                         fps=fps, video=video, detector=detector)
    return json.dumps(doc, indent=indent, allow_nan=False)


def validate_document(doc: dict) -> None:
    """The single authority for a v2 document, used by both write and read (A3.2).

    Import policy (§3.1): wrong ``format`` / unsupported ``schema_version`` rejected
    outright. Then: no nulls or NaN/Inf anywhere (recursive); status valid; coverage
    ordered (start < end) and finite; every event validated by the shared
    `validate_event_dict` (required keys genuinely required -- no defaulting/repair --
    plus span-only-on-gradual, flags whitelist, bool checks, ordering, coverage bound);
    ``event_count`` matches; and the ``cuts`` projection equals the events. Raises
    `ExportValidationError`."""
    if not isinstance(doc, dict):
        raise ExportValidationError("document must be an object")
    if doc.get("format") != FORMAT:
        raise ExportValidationError(f"wrong format {doc.get('format')!r} (expected {FORMAT!r})")
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ExportValidationError(
            f"unsupported schema_version {doc.get('schema_version')!r} (expected {SCHEMA_VERSION})")

    reject_nan_and_null(doc)                           # no nulls / NaN / Inf at any depth

    analysis = doc.get("analysis") or {}
    status = analysis.get("status")
    if status not in VALID_STATUS:
        raise ExportValidationError(f"analysis.status {status!r} not in {VALID_STATUS}")

    coverage = list(analysis.get("coverage") or [])
    for j, s in enumerate(coverage):
        if not (isinstance(s, dict) and "start" in s and "end" in s):
            raise ExportValidationError(f"coverage[{j}] must be an object with start and end")
        if not (_finite(s["start"]) and _finite(s["end"])):
            raise ExportValidationError(f"coverage[{j}] bounds must be finite")
        if not s["start"] < s["end"]:
            raise ExportValidationError(f"coverage[{j}] must be ordered (start < end)")

    cut_events = doc.get("cut_events")
    if not isinstance(cut_events, list):
        raise ExportValidationError("cut_events must be a list")
    prev = None
    for i, ed in enumerate(cut_events):
        prev = validate_event_dict(ed, i, coverage, prev)

    if "event_count" in analysis and analysis["event_count"] != len(cut_events):
        raise ExportValidationError(
            f"analysis.event_count {analysis['event_count']} != {len(cut_events)} cut_events")

    projected = [ed["time"] for ed in cut_events]
    if doc.get("cuts") != projected:
        raise ExportValidationError(
            "projection mismatch: document cuts != [e.time for e in cut_events]")
