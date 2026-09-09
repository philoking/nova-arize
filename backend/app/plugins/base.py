"""The contract every capability plugin implements.

A provider is deliberately small: it declares its tools, runs them, and
contributes a bit of system prompt. Tool names it returns are *bare* (e.g.
``create_note``); the registry namespaces them (``notes__create_note``) before
they reach the model and strips the namespace again before dispatch, so two
providers can both expose a ``search`` tool without colliding.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ToolProvider(Protocol):
    #: Short namespace, [a-z0-9_]+, unique across providers (e.g. "notes").
    name: str

    @property
    def enabled(self) -> bool:
        """Whether this provider is configured and should be active.

        Disabled providers are dropped by the registry, so their tools never
        reach the model and their ``system_note`` is never shown.
        """
        ...

    def tool_specs(self) -> list[dict]:
        """Tool definitions in Ollama/OpenAI function-calling format.

        Names are bare - the registry adds the ``<name>__`` prefix.
        """
        ...

    async def execute(self, tool: str, args: dict) -> dict:
        """Run one tool call. ``tool`` is the bare name (prefix stripped).

        Must not raise: return ``{"error": ...}`` so the model can react.
        """
        ...

    def system_note(self) -> str:
        """A prompt fragment appended to the system message, or ``""``."""
        ...

    async def health(self) -> bool:
        """Whether the provider's backing service is reachable."""
        ...
