"""Event-first cut representation for the v2 export (CineScript S3, design §3).

Stdlib only -- importable without torch / opencv / scenedetect. A cut is a
single timestamped event; richer detectors attach a frame index, a transition
kind, a measured span (gradual transitions only), and a native-scale confidence.

Omission, not null (design §3.2): a field a detector cannot supply is left
absent from the serialized event, never written as ``null``. In memory that
absence is a ``None`` attribute; `event_to_dict` drops it on the way out.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

VALID_KINDS = ("hard", "gradual", "unknown")


class ExportValidationError(ValueError):
    """A cut event / document violated a design §3.3 validity rule."""


@dataclass
class Confidence:
    """A detector's native confidence. ``metric`` names the scale so values are
    never compared across metrics (design §3.2); ``value`` is the raw number --
    not normalised to [0,1], not calibrated, not a probability unless it is one."""
    value: float
    metric: str
    higher_is_stronger: bool = True


@dataclass
class Span:
    """The measured extent of a gradual transition. Never invented from a point;
    ``start <= time <= end`` (enforced in `validate_events`)."""
    start: float
    end: float


@dataclass
class CutEvent:
    """One cut. ``time`` (seconds, the detector's anchor) is the only required
    field; everything else is omitted when the detector does not measure it.
    ``transition_kind`` is required in the document and defaults to ``"unknown"``
    -- a detector that cannot classify emits ``unknown``, never a guessed value."""
    time: float
    frame: int | None = None
    transition_kind: str = "unknown"
    span: Span | None = None
    confidence: Confidence | None = None
    flags: list[str] | None = None


def _finite(x) -> bool:
    """Real, finite number -- and NOT a bool (True/False must not pass as a time)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def project_cuts(events) -> list[float]:
    """The legacy `cuts` projection: one timestamp per event, in event order.

    This is the ONLY place a float cut list is derived from. Events are the
    single source of truth (design §2), so this never re-sorts or de-duplicates
    -- ordering/uniqueness are enforced at write by `validate_events`."""
    return [e.time for e in events]


def wrap_times(times) -> list[CutEvent]:
    """Wrap bare timestamps as minimal events (transition_kind ``unknown``,
    everything else omitted). The shape the six poor/rich detectors will use in
    later slices; handy now for tests and callers that only have floats."""
    return [CutEvent(time=t) for t in times]


def _within_coverage(t, coverage) -> bool:
    for s in coverage:
        if s["start"] <= t <= s["end"]:
            return True
    return False


def validate_events(events, coverage=None) -> None:
    """Enforce the event-level slice of design §3.3. Raises `ExportValidationError`
    on the first violation.

    Rules: times finite and >= 0 (and within ``coverage`` when supplied);
    strictly ascending with no duplicate times; ``transition_kind`` in the known
    set; span bounds finite with ``start <= time <= end``; ``confidence`` (when
    present) carries a finite value and a non-empty ``metric``."""
    prev = None
    for i, e in enumerate(events):
        if not _finite(e.time) or e.time < 0:
            raise ExportValidationError(f"event {i}: time must be finite and >= 0 (got {e.time!r})")
        if e.transition_kind not in VALID_KINDS:
            raise ExportValidationError(
                f"event {i}: transition_kind {e.transition_kind!r} not in {VALID_KINDS}")
        if e.frame is not None and (not isinstance(e.frame, int) or isinstance(e.frame, bool) or e.frame < 0):
            raise ExportValidationError(f"event {i}: frame must be a non-negative int (got {e.frame!r})")
        if prev is not None:
            if e.time == prev:
                raise ExportValidationError(f"event {i}: duplicate time {e.time} (dedupe before emit)")
            if e.time < prev:
                raise ExportValidationError(
                    f"event {i}: times must be strictly ascending ({e.time} after {prev})")
        prev = e.time
        if e.span is not None:
            if not (_finite(e.span.start) and _finite(e.span.end)):
                raise ExportValidationError(f"event {i}: span bounds must be finite")
            if not (e.span.start <= e.time <= e.span.end):
                raise ExportValidationError(
                    f"event {i}: span sandwich violated "
                    f"(need start {e.span.start} <= time {e.time} <= end {e.span.end})")
        if e.confidence is not None:
            if not _finite(e.confidence.value):
                raise ExportValidationError(f"event {i}: confidence.value must be finite")
            if not (isinstance(e.confidence.metric, str) and e.confidence.metric):
                raise ExportValidationError(f"event {i}: confidence requires a non-empty metric")
        if coverage and not _within_coverage(e.time, coverage):
            raise ExportValidationError(f"event {i}: time {e.time} outside declared coverage")


def event_to_dict(e: CutEvent) -> dict:
    """Serialize one event, omitting absent optional fields (never ``null``).
    Field order follows the design §3 example: time, frame, transition_kind,
    span, confidence, flags."""
    d: dict = {"time": e.time}
    if e.frame is not None:
        d["frame"] = e.frame
    d["transition_kind"] = e.transition_kind          # required -- always present
    if e.span is not None:
        d["span"] = {"start": e.span.start, "end": e.span.end}
    if e.confidence is not None:
        d["confidence"] = {
            "value": e.confidence.value,
            "metric": e.confidence.metric,
            "higher_is_stronger": e.confidence.higher_is_stronger,
        }
    if e.flags:
        d["flags"] = list(e.flags)
    return d


def event_from_dict(d: dict) -> CutEvent:
    """Rebuild a CutEvent from a serialized event (used when re-validating an
    already-written document, e.g. the repo fixtures). Tolerant of missing
    optional keys; structural validity is checked by `validate_events`."""
    span = d.get("span")
    conf = d.get("confidence")
    return CutEvent(
        time=d["time"],
        frame=d.get("frame"),
        transition_kind=d.get("transition_kind", "unknown"),
        span=Span(span["start"], span["end"]) if span else None,
        confidence=Confidence(conf.get("value"), conf.get("metric"), conf.get("higher_is_stronger", True))
        if conf else None,
        flags=d.get("flags"),
    )
