"""Facts system for training metrics, events, and metadata.

The facts system provides a standardized way to emit, store, and read
training events, scalar metrics, and run metadata.

Components:
- builders: Construct standardized event payloads
- writers: Write facts to disk (JSONL for events, CSV for scalars)
- readers: Read facts from disk for analysis in reports
- meta: Build and write canonical run metadata (facts/meta.json)
"""

from .builders import build_event_payload
from .meta import (
    PROCESS_ID_NAMES,
    build_class_def_str,
    build_meta,
    build_process_groups_key,
    canonicalize_datatreatment,
    canonicalize_process_groups,
    compute_meta_hash,
    load_meta_override,
    merge_meta_with_override,
    read_meta,
    write_meta,
)
from .writers import append_jsonl_event, append_scalars_csv, ensure_facts_dir

__all__ = [
    # Builders
    "build_event_payload",
    # Writers
    "append_jsonl_event",
    "append_scalars_csv",
    "ensure_facts_dir",
    # Meta
    "PROCESS_ID_NAMES",
    "build_meta",
    "write_meta",
    "read_meta",
    "load_meta_override",
    "merge_meta_with_override",
    "canonicalize_process_groups",
    "canonicalize_datatreatment",
    "build_process_groups_key",
    "build_class_def_str",
    "compute_meta_hash",
]
