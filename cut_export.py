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


def _req(d, key, ctx):
    """Fetch a genuinely-required key -- absence is rejection, never a default (A3.2 /
    S3-F2 gap 1). No silent repair on read."""
    if not isinstance(d, dict):
        raise ExportValidationError(f"{ctx} must be an object")
    if key not in d:
        raise ExportValidationError(f"missing required key {ctx}.{key}")
    return d[key]


def _req_nonempty_str(d, key, ctx):
    v = _req(d, key, ctx)
    if not (isinstance(v, str) and v):
        raise ExportValidationError(f"{ctx}.{key} must be a non-empty string (got {v!r})")
    return v


def validate_document(doc: dict) -> None:
    """The single authority for a v2 document, used by both write and read (A3.2).

    Required document sections and their core fields are genuinely required on read --
    absence is rejection, not defaulting (S3-F2 gap 1). Optional-by-design fields
    (backend, model, tool_commit, tool_dirty, generated_by/utc, source.duration_seconds,
    analysis.coverage, and event frame/span/confidence/flags) may be absent but are
    type-checked when present. See the S3-F2 report for the full required/optional table.
    Raises `ExportValidationError`."""
    if not isinstance(doc, dict):
        raise ExportValidationError("document must be an object")
    # -- discriminators (§3.1 import policy) --
    if doc.get("format") != FORMAT:
        raise ExportValidationError(f"wrong format {doc.get('format')!r} (expected {FORMAT!r})")
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ExportValidationError(
            f"unsupported schema_version {doc.get('schema_version')!r} (expected {SCHEMA_VERSION})")

    reject_nan_and_null(doc)                           # no nulls / NaN / Inf at any depth

    # -- top-level required scalars --
    fps = _req(doc, "fps", "$")
    if not (_finite(fps) and fps > 0):                 # real, not bool, finite, positive
        raise ExportValidationError(f"fps must be a positive finite number, not bool (got {fps!r})")
    _req_nonempty_str(doc, "video", "$")
    _req_nonempty_str(doc, "detector", "$")

    # -- source (media identity: path + size + mtime; duration optional per E6) --
    source = _req(doc, "source", "$")
    _req_nonempty_str(source, "path", "source")
    size = _req(source, "size_bytes", "source")
    if not (isinstance(size, int) and not isinstance(size, bool) and size >= 0):
        raise ExportValidationError(f"source.size_bytes must be a non-negative int (got {size!r})")
    _req_nonempty_str(source, "mtime_utc", "source")
    if "duration_seconds" in source and not (_finite(source["duration_seconds"]) and source["duration_seconds"] > 0):
        raise ExportValidationError(
            f"source.duration_seconds must be a positive finite number (got {source['duration_seconds']!r})")

    # -- run (detector identity + resolved settings required; provenance optional) --
    run = _req(doc, "run", "$")
    _req_nonempty_str(run, "detector_id", "run")
    settings = _req(run, "settings", "run")
    if not isinstance(settings, dict):
        raise ExportValidationError(f"run.settings must be an object (got {settings!r})")
    for opt in ("backend", "model", "tool_commit", "generated_by", "generated_utc"):
        if opt in run and not (isinstance(run[opt], str) and run[opt]):
            raise ExportValidationError(f"run.{opt} must be a non-empty string when present (got {run[opt]!r})")
    if "tool_dirty" in run and not isinstance(run["tool_dirty"], bool):
        raise ExportValidationError(f"run.tool_dirty must be a bool when present (got {run['tool_dirty']!r})")

    # -- analysis (status + event_count + warnings required; coverage optional per E6) --
    analysis = _req(doc, "analysis", "$")
    if not isinstance(analysis, dict):
        raise ExportValidationError("analysis must be an object")
    status = _req(analysis, "status", "analysis")
    if status not in VALID_STATUS:
        raise ExportValidationError(f"analysis.status {status!r} not in {VALID_STATUS}")
    warnings = _req(analysis, "warnings", "analysis")
    if not (isinstance(warnings, list) and all(isinstance(w, str) for w in warnings)):
        raise ExportValidationError(f"analysis.warnings must be a list of strings (got {warnings!r})")
    coverage = analysis.get("coverage")                # OPTIONAL (E6 duration ruling)
    if coverage is not None:
        if not isinstance(coverage, list):
            raise ExportValidationError(f"analysis.coverage must be a list (got {coverage!r})")
        for j, s in enumerate(coverage):
            if not (isinstance(s, dict) and "start" in s and "end" in s):
                raise ExportValidationError(f"coverage[{j}] must be an object with start and end")
            if not (_finite(s["start"]) and _finite(s["end"])):
                raise ExportValidationError(f"coverage[{j}] bounds must be finite")
            if not s["start"] < s["end"]:
                raise ExportValidationError(f"coverage[{j}] must be ordered (start < end)")
    else:
        coverage = []

    # -- cut_events (each via the shared event validator) --
    cut_events = _req(doc, "cut_events", "$")
    if not isinstance(cut_events, list):
        raise ExportValidationError("cut_events must be a list")
    prev = None
    for i, ed in enumerate(cut_events):
        prev = validate_event_dict(ed, i, coverage, prev)

    # -- event_count required and must match --
    ec = _req(analysis, "event_count", "analysis")
    if not (isinstance(ec, int) and not isinstance(ec, bool) and ec == len(cut_events)):
        raise ExportValidationError(
            f"analysis.event_count must be an int equal to {len(cut_events)} (got {ec!r})")

    # -- cuts projection --
    cuts = _req(doc, "cuts", "$")
    if cuts != [ed["time"] for ed in cut_events]:
        raise ExportValidationError(
            "projection mismatch: document cuts != [e.time for e in cut_events]")
