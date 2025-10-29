from __future__ import annotations

from enum import Enum
from typing import Mapping, Protocol, TypedDict, TypeAlias, runtime_checkable

from bot_crypto.constants.ssot_constants import STATUS_FAILURE, STATUS_SUCCESS
from bot_crypto.domain.models.alert_level import AlertLevel


class ResultStatus(str, Enum):
    """Outcome status for tool executions."""

    SUCCESS = STATUS_SUCCESS
    FAILURE = STATUS_FAILURE


# Reuse AlertLevel as severity enum for results
ResultSeverity = AlertLevel

JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | Mapping[str, "JsonValue"]


class ToolInput(TypedDict, total=True):
    """Structured input passed to each tool."""

    trace_id: str
    user_roles: list[str]
    payload: dict[str, JsonValue]


class ToolResult(TypedDict, total=True):
    """Execution result returned by a tool."""

    status: ResultStatus
    severity: ResultSeverity | None
    data: dict[str, JsonValue]


@runtime_checkable
class ToolProtocol(Protocol):
    """Contract for injectable Tools (PEP 544)."""

    name: str
    description: str

    async def run(self, *, input: ToolInput) -> ToolResult: ...
