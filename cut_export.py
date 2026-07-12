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
    CutEvent, ExportValidationError, event_from_dict, event_to_dict,
    project_cuts, validate_events,
)

FORMAT = "film-cut-analysis/cut-events"
SCHEMA_VERSION = 2
VALID_STATUS = ("complete", "partial", "failed")


def _drop_none(d: dict) -> dict:
    """Omission, not null (design §3.2): strip keys whose value is None."""
    return {k: v for k, v in d.items() if v is not None}


def build_document(events, *, source, run, analysis, fps, video, detector) -> dict:
    """Build the §3 v2 document.

    ``source`` / ``run`` are dicts of provenance (None values are omitted).
    ``analysis`` supplies ``status`` (complete|partial|failed), ``coverage``
    (list of {start,end} spans actually analysed) and optional ``warnings``;
    ``event_count`` is derived here, not trusted from the caller.

    Validates the events against §3.3 and self-tests the ``cuts`` projection.
    Raises `ExportValidationError` on any violation."""
    status = analysis.get("status")
    if status not in VALID_STATUS:
        raise ExportValidationError(f"analysis.status {status!r} not in {VALID_STATUS}")
    coverage = list(analysis.get("coverage") or [])

    validate_events(events, coverage=coverage)

    cut_events = [event_to_dict(e) for e in events]
    cuts = project_cuts(events)
    # design §3.3 rule 3: cuts is exactly the event projection -- verified, not trusted.
    if cuts != [ce["time"] for ce in cut_events]:
        raise ExportValidationError("projection mismatch: cuts != [e.time for e in cut_events]")

    analysis_block = {
        "status": status,
        "coverage": [{"start": s["start"], "end": s["end"]} for s in coverage],
        "event_count": len(events),
        "warnings": list(analysis.get("warnings") or []),
    }

    return {
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


def serialize(events, *, source, run, analysis, fps, video, detector, indent=2) -> str:
    """`build_document` then `json.dumps`. The string a v2 export writes to disk."""
    return json.dumps(
        build_document(events, source=source, run=run, analysis=analysis,
                       fps=fps, video=video, detector=detector),
        indent=indent,
    )


def validate_document(doc: dict) -> None:
    """Validate an already-built v2 document (e.g. a repo fixture) against §3.

    Import policy (design §3.1): wrong ``format`` or unsupported
    ``schema_version`` are rejected outright -- never best-effort. Then the
    event-level §3.3 rules are re-run and the ``cuts`` projection is checked
    against the events. Raises `ExportValidationError`."""
    if doc.get("format") != FORMAT:
        raise ExportValidationError(f"wrong format {doc.get('format')!r} (expected {FORMAT!r})")
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ExportValidationError(
            f"unsupported schema_version {doc.get('schema_version')!r} (expected {SCHEMA_VERSION})")

    analysis = doc.get("analysis") or {}
    status = analysis.get("status")
    if status not in VALID_STATUS:
        raise ExportValidationError(f"analysis.status {status!r} not in {VALID_STATUS}")
    coverage = list(analysis.get("coverage") or [])

    cut_events = doc.get("cut_events")
    if not isinstance(cut_events, list):
        raise ExportValidationError("cut_events must be a list")
    events = [event_from_dict(ce) for ce in cut_events]
    validate_events(events, coverage=coverage)

    if "event_count" in analysis and analysis["event_count"] != len(events):
        raise ExportValidationError(
            f"analysis.event_count {analysis['event_count']} != {len(events)} cut_events")

    projected = project_cuts(events)
    if doc.get("cuts") != projected:
        raise ExportValidationError(
            "projection mismatch: document cuts != [e.time for e in cut_events]")
