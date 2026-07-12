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
DOCUMENTED_FLAGS = ("strobe_suppressed",)   # the only sanctioned flags (design §3.2)


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


def reject_nan_and_null(obj, path="$") -> None:
    """Recursively reject ``None`` (omission-only, never null) and non-finite floats
    (NaN / Infinity) at any depth of a document (amendment A3.2). Booleans pass."""
    if obj is None:
        raise ExportValidationError(f"null value at {path} -- v2 is omission-only, no nulls")
    if isinstance(obj, bool):
        return
    if isinstance(obj, float) and not math.isfinite(obj):
        raise ExportValidationError(f"non-finite number at {path}")
    if isinstance(obj, dict):
        for k, v in obj.items():
            reject_nan_and_null(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for j, v in enumerate(obj):
            reject_nan_and_null(v, f"{path}[{j}]")


def validate_event_dict(d, index=0, coverage=None, prev_time=None):
    """Validate ONE serialized event dict against design §3.3 + amendment A3.2, and
    return its time (for ordering). This is the SINGLE event-validation path shared by
    write (`build_document`) and read (`validate_document`): required keys are genuinely
    required -- nothing is defaulted or repaired; invalid input is rejected. Raises
    `ExportValidationError`."""
    if not isinstance(d, dict):
        raise ExportValidationError(f"event {index}: must be an object")
    if "time" not in d:
        raise ExportValidationError(f"event {index}: missing required 'time'")
    t = d["time"]
    if not _finite(t) or t < 0:
        raise ExportValidationError(f"event {index}: time must be finite and >= 0 (got {t!r})")
    if "transition_kind" not in d:
        raise ExportValidationError(f"event {index}: missing required 'transition_kind'")
    kind = d["transition_kind"]
    if kind not in VALID_KINDS:
        raise ExportValidationError(f"event {index}: transition_kind {kind!r} not in {VALID_KINDS}")
    if prev_time is not None:
        if t == prev_time:
            raise ExportValidationError(f"event {index}: duplicate time {t} (dedupe before emit)")
        if t < prev_time:
            raise ExportValidationError(
                f"event {index}: times must be strictly ascending ({t} after {prev_time})")
    if "frame" in d:
        f = d["frame"]
        if not isinstance(f, int) or isinstance(f, bool) or f < 0:
            raise ExportValidationError(f"event {index}: frame must be a non-negative int (got {f!r})")
    if "span" in d:
        if kind != "gradual":
            raise ExportValidationError(
                f"event {index}: span only allowed on gradual events (kind={kind!r})")
        span = d["span"]
        if not (isinstance(span, dict) and "start" in span and "end" in span):
            raise ExportValidationError(f"event {index}: span must be an object with start and end")
        lo, hi = span["start"], span["end"]
        if not (_finite(lo) and _finite(hi)):
            raise ExportValidationError(f"event {index}: span bounds must be finite")
        if not (lo <= t <= hi):
            raise ExportValidationError(
                f"event {index}: span sandwich violated (need {lo} <= {t} <= {hi})")
    if "confidence" in d:
        c = d["confidence"]
        if not isinstance(c, dict):
            raise ExportValidationError(f"event {index}: confidence must be an object")
        for key in ("value", "metric", "higher_is_stronger"):
            if key not in c:
                raise ExportValidationError(f"event {index}: confidence missing required {key!r}")
        if not _finite(c["value"]):
            raise ExportValidationError(f"event {index}: confidence.value must be finite")
        if not (isinstance(c["metric"], str) and c["metric"]):
            raise ExportValidationError(f"event {index}: confidence.metric must be a non-empty string")
        if not isinstance(c["higher_is_stronger"], bool):
            raise ExportValidationError(f"event {index}: confidence.higher_is_stronger must be a bool")
    if "flags" in d:
        flags = d["flags"]
        if not isinstance(flags, list) or not all(isinstance(x, str) for x in flags):
            raise ExportValidationError(f"event {index}: flags must be a list of strings")
        bad = [x for x in flags if x not in DOCUMENTED_FLAGS]
        if bad:
            raise ExportValidationError(
                f"event {index}: undocumented flags {bad} (allowed: {DOCUMENTED_FLAGS})")
    if coverage and not _within_coverage(t, coverage):
        raise ExportValidationError(f"event {index}: time {t} outside declared coverage")
    return t


def validate_events(events, coverage=None) -> None:
    """Object-level entry point: validate a list of CutEvents by routing each through
    the shared `validate_event_dict` (via `event_to_dict`) -- so objects and documents
    are checked by exactly the same rules. Raises `ExportValidationError`."""
    prev = None
    for i, e in enumerate(events):
        prev = validate_event_dict(event_to_dict(e), i, coverage, prev)


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
