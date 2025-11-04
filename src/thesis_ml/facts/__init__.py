"""Facts system for training metrics and events.

The facts system provides a standardized way to emit, store, and read
training events and scalar metrics.

Components:
- builders: Construct standardized event payloads
- writers: Write facts to disk (JSONL for events, CSV for scalars)
- readers: Read facts from disk for analysis in reports
"""

from .builders import build_event_payload
from .writers import append_jsonl_event, append_scalars_csv, ensure_facts_dir

__all__ = [
    "build_event_payload",
    "append_jsonl_event",
    "append_scalars_csv",
    "ensure_facts_dir",
]
