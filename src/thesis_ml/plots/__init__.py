"""Config-driven plotting package.

Exposes the orchestrator entrypoint used by training loops. Families are
registered in `registry.py`. Keep the surface area small and explicit.
"""

from .orchestrator import handle_event

__all__ = ["handle_event"]
