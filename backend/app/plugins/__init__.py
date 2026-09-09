"""Pluggable tool providers for the assistant's agent loop.

A ``ToolProvider`` bundles a set of LLM tools, the code that executes them, and
an optional system-prompt fragment describing when to use them. The ``Registry``
aggregates every enabled provider so the rest of the app never has to know which
capabilities exist - it just asks the registry for the merged tool list, an
executor, and a combined system note.

Adding a new capability (notes, tasks, an MCP-bridged service, …) is a matter of
writing one provider and appending it to the registry in ``main.py`` - no edits
to the LLM layer or the chat route.
"""

from __future__ import annotations

from .base import ToolProvider
from .registry import Registry

__all__ = ["ToolProvider", "Registry"]
