from __future__ import annotations

from typing import TypedDict

from bot_crypto.interfaces.agent.tool_protocol import ToolResult


class Plan(TypedDict, total=True):
    """Ordered list of tool names composing a plan."""

    steps: list[str]


class ToolExecutionTrace(TypedDict, total=True):
    """Single step execution log used for audit trails."""

    step: str
    timestamp: str
    result: ToolResult


class AgentResponse(TypedDict, total=True):
    """Public API response of :class:`ToolAgent.run`."""

    final_output: str


class SerializedAgentState(TypedDict, total=True):
    """Snapshot used for persistence and recovery."""

    trace_id: str
    execution_trace: list[ToolExecutionTrace]
    intermediate_results: dict[str, ToolResult]
    final_output: str
    sha256: str
