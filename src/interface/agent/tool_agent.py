from __future__ import annotations

import hashlib
import inspect
import json
import uuid
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import Any, Final, Literal, Mapping, Self, TypedDict, TypeAlias, cast, List, Dict
from pydantic import BaseModel, ConfigDict, Field

from bot_crypto.constants.security_and_resilience_constants import (
    ASYNC_TIMEOUT_SEC,
    CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    RETRY_MAX_ATTEMPTS,
)
from config.project_settings import (
    TOOL_AGENT_BIAS_CONFIG_PATH,
    TOOL_AGENT_COMMAND_MAPPING_PATH,
    TOOL_AGENT_EVENT_MANIFEST_PATH,
    TOOL_AGENT_QUERY_REGISTRY_PATH,
)

from bot_crypto.application.context.execution_context import ExecutionContext
from bot_crypto.domain.errors.exceptions.command_exceptions import CommandNotRegisteredError
from bot_crypto.infrastructure.config.di_container import DIContainer
from bot_crypto.infrastructure.config.di_provider import di_provider
from bot_crypto.infrastructure.decorators import audit_event as _audit_event
from bot_crypto.infrastructure.decorators import retry_safe as retry_async
from bot_crypto.infrastructure.decorators import with_async_timeout as with_timeout
from bot_crypto.infrastructure.decorators.logging_decorators import log_event as log_action
from bot_crypto.infrastructure.interfaces.protocols.typing_interfaces import (
    CommandBusProtocol,
    QueryBusProtocol,
)
from bot_crypto.interfaces.agent.tool_protocol import ResultSeverity, ResultStatus, ToolResult
from bot_crypto.infrastructure.services.web_verification_service import WebVerificationService
from bot_crypto.interfaces.protocols.web_verification_protocol import (
    WebVerificationEvidence,
    WebVerificationProtocol,
)
from bot_crypto.interfaces.protocols.audit_publisher_protocol import AuditPublisherProtocol
from bot_crypto.interfaces.protocols.cqrs import CommandBase
from bot_crypto.interfaces.protocols.immutable_ledger_publisher_protocol import ImmutableLedgerPublisherProtocol
from bot_crypto.interfaces.protocols.risk_engine_protocol import RiskEngineProtocol
from bot_crypto.interfaces.protocols.state_repository_protocol import StateRepositoryProtocol
from bot_crypto.constants.metrics_constants import (
    METRIC_AGENT_LATENCY_SECONDS,
    METRIC_TOKEN_USAGE_ESTIMATE,
)
from bot_crypto.infrastructure.contracts import require
from bot_crypto.interfaces.protocols.logger_factory_protocol import LoggerFactoryProtocol
from bot_crypto.interfaces.protocols.metrics_registry_protocol import MetricsRegistryProtocol
import bot_crypto.constants.risk_constants as riskc

__all__: list[str] = [
    "ToolAgent",
    "ObservabilityLogger",
    "ObservationPayload",
    "PlanStep",
    "ExecutionPlan",
    "AgentResult",
    "ObservationModel",
    "PlanStepModel",
    "ExecutionPlanModel",
    "AgentResultModel",
    "DynamicCommand",
    "RiskViolationError",
    "ASYNC_TIMEOUT_SEC",
    "CIRCUIT_BREAKER_FAILURE_THRESHOLD",
    "RETRY_MAX_ATTEMPTS",
]

# Remediation Plan (Scaffolding Audit)
# 1) Consolidate duplicated naming/modules (constants/ssot_constants*, legacy dirs) into canonical config/registry.
# 2) Centralize documentation/audit artifacts under docs/ with index and deprecate scattered root files.
# 3) Reduce overlapping interfaces/adapters/http boundaries via clear ownership matrix.
# 4) Streamline scripts/CI tooling by merging redundant check_* utilities into a single orchestration entrypoint.
# 5) Define a migration path for legacy/temp directories to prevent future drift.



class ObservabilityLogger:
    """Injects mandatory observability context into logger outputs."""

    def __init__(self, delegate: Any, context_provider: Callable[[], dict[str, Any]]) -> None:
        self._delegate = delegate
        self._context_provider = context_provider

    def _merge_extra(self, extra: Mapping[str, Any] | None) -> dict[str, Any]:
        base = dict(self._context_provider())
        if extra:
            for key, value in extra.items():
                if key not in base:
                    base[key] = value
        return base

    def debug(self, msg: str, *args: Any, extra: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        kwargs["extra"] = self._merge_extra(extra)
        self._delegate.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, extra: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        kwargs["extra"] = self._merge_extra(extra)
        self._delegate.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, extra: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        kwargs["extra"] = self._merge_extra(extra)
        self._delegate.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, extra: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        kwargs["extra"] = self._merge_extra(extra)
        self._delegate.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, extra: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        kwargs["extra"] = self._merge_extra(extra)
        self._delegate.critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, extra: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        kwargs["extra"] = self._merge_extra(extra)
        self._delegate.exception(msg, *args, **kwargs)


class ObservationPayload(TypedDict, total=True):
    trace_id: str
    timestamp: str
    user_id: int
    balance: float
    positions: List[str]
    telemetry: Dict[str, Any]


class PlanStep(TypedDict, total=True):
    command: str
    event: str
    metric: str
    parameters: Dict[str, Any]
    origin: str


class ExecutionPlan(TypedDict, total=True):
    steps: List[PlanStep]
    bias_controls: Dict[str, Any]


class AgentResult(TypedDict, total=True):
    trace_id: str
    status: ResultStatus
    confidence: str
    final_output: str
    executed_steps: List[PlanStep]
    decision_traceability: str


CommandMetadata: TypeAlias = dict[str, str]
CommandMapping: TypeAlias = dict[str, CommandMetadata]
PlanSteps: TypeAlias = List[PlanStep]
CommonFragmentOrigin: TypeAlias = Literal["canonical", "contrarian", "reframed", "system"]
SerializablePrimitive: TypeAlias = str | int | float | bool | None
# Semplificazione: evitiamo alias ricorsivi che alcuni analyzer marcano come invalidi.
NormalizedMapping: TypeAlias = Mapping[str, Any]
NormalizedSequence: TypeAlias = Sequence[Any]
NormalizedValue: TypeAlias = SerializablePrimitive | NormalizedSequence | NormalizedMapping | tuple[Any, ...]

class ObservationModel(BaseModel):
    """Schema di validazione per l'osservazione."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_id: str
    timestamp: str
    user_id: int = Field(ge=0)
    balance: float = Field(ge=0.0)
    positions: List[str]
    telemetry: Dict[str, Any]


class PlanStepModel(BaseModel):
    """Schema per un singolo step di piano."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    command: str
    event: str
    metric: str
    parameters: dict[str, Any]
    origin: str


class ExecutionPlanModel(BaseModel):
    """Schema per il piano esecutivo."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    steps: List[PlanStepModel]
    bias_controls: Dict[str, Any]



class AgentResultModel(BaseModel):
    """Schema per l'output finale dell'agente."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_id: str
    status: ResultStatus
    confidence: str
    final_output: str
    executed_steps: List[PlanStepModel]
    decision_traceability: str


def audit_event(event_type: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Rende esplicito l'uso del decoratore audit_event nel modulo."""

    return _audit_event(event_type=event_type)


@dataclass(frozen=True, slots=True)
class DynamicCommand(CommandBase):
    """Fallback command conforme al protocollo CQRS."""

    command_type: str
    payload: dict[str, Any]


class RiskViolationError(RuntimeError):
    """Segnala una violazione critica rilevata dal RiskEngine prima dell'esecuzione."""

    def __init__(
        self,
        message: str,
        *,
        severity: ResultSeverity | None = None,
        decision: Any | None = None,
    ) -> None:
        super().__init__(message)
        self._severity = severity
        self._decision = decision

    @property
    def severity(self) -> ResultSeverity | None:
        return self._severity

    @property
    def decision(self) -> Any | None:
        return self._decision


class ToolAgent:
    """Orchestratore observe-plan-act con controlli di rischio e bias cognitivo."""

    _BIAS_CONFIG_FILE: Final[Path] = TOOL_AGENT_BIAS_CONFIG_PATH
    _COMMAND_MAPPING_FILE: Final[Path] = TOOL_AGENT_COMMAND_MAPPING_PATH
    _QUERY_REGISTRY_FILE: Final[Path] = TOOL_AGENT_QUERY_REGISTRY_PATH
    _EVENT_MANIFEST_FILE: Final[Path] = TOOL_AGENT_EVENT_MANIFEST_PATH
    _LATENCY_HISTORY_LIMIT: Final[int] = 50
    _SEVERITY_PRIORITY: Final[dict[ResultSeverity, int]] = {
        ResultSeverity.INFO: 0,
        ResultSeverity.WARN: 1,
        ResultSeverity.CRITICAL: 2,
    }


    def __init__(
        self,
        *,
        logger_factory: LoggerFactoryProtocol | None = None,
        audit_publisher: AuditPublisherProtocol | None = None,
        metrics_registry: MetricsRegistryProtocol | None = None,
        command_bus: CommandBusProtocol | None = None,
        query_bus: QueryBusProtocol | None = None,
        risk_engine: RiskEngineProtocol | None = None,
        state_repo: StateRepositoryProtocol | None = None,
        ledger_publisher: ImmutableLedgerPublisherProtocol | None = None,
        web_verifier: WebVerificationProtocol | None = None,
        container: DIContainer | None = None,
    ) -> None:
        self._container = container or DIContainer.get_instance()
        self._logger_factory = self._resolve_logger_factory(logger_factory)
        base_logger = self._logger_factory.get_logger("tool_agent")
        self._logger = ObservabilityLogger(base_logger, self._log_context_provider)
        self._audit = self._resolve_audit_publisher(audit_publisher)
        self._metrics = self._resolve_metrics_registry(metrics_registry)
        self._command_bus = self._resolve_command_bus(command_bus)
        self._query_bus = self._resolve_query_bus(query_bus)
        self._risk_engine = self._resolve_risk_engine(risk_engine)
        self._state_repo = self._require_dependency(state_repo, "state_repo")
        self._ledger = self._require_dependency(ledger_publisher, "ledger_publisher")

        self._web_verifier = web_verifier or WebVerificationService(
            audit_publisher=self._audit,
            ledger=self._ledger,
            metrics_registry=self._metrics,
            logger_factory=self._logger_factory,
        )

        self._toolagent_id: str = f"tool_agent_{uuid.uuid4().hex}"
        self._phase_latency_samples: Dict[str, List[float]] = {}
        self._current_context: ExecutionContext | None = None
        self._current_user_id: int = 0
        self._token_usage_estimate: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        self._act_phase_severity: ResultSeverity = ResultSeverity.INFO

        self._trace_id: str = self._generate_trace_id()
        self._confidence_level: str = "bassa"
        self._plan_invocations: int = 0
        self._pattern_frequency: Counter[str] = Counter()
        self._bias_policy_cache: dict[str, Any] | None = None
        self._command_mapping_cache: CommandMapping | None = None
        self._event_manifest_cache: dict[str, Any] | None = None
        self._query_registry_cache: dict[str, dict[str, str]] | None = None
        self._execution_trace: list[ToolResult] = []
        self._final_output: str = ""
        self._decision_traceability: str = "unverifiable"
        self._consecutive_failures: int = 0
        self._web_verification_evidence: WebVerificationEvidence | None = None
        self._root_path: Path = Path(__file__).resolve().parents[3]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @retry_async(max_attempts=RETRY_MAX_ATTEMPTS)
    @audit_event("tool_agent.run")
    @log_action
    @with_timeout(seconds=ASYNC_TIMEOUT_SEC)
    async def run(self, ctx: ExecutionContext) -> AgentResult:
        """Esegue la pipeline observe-plan-act per il contesto fornito."""

        require(isinstance(ctx, ExecutionContext), "ctx must be an ExecutionContext")
        self._reset_state()
        self._trace_id = self._generate_trace_id(ctx.trace_id)
        self._initialize_run_context(ctx)
        cycle_start = perf_counter()
        result: AgentResult | None = None
        try:
            self._web_verification_evidence = await self._perform_web_verification(ctx)
            observe_start = perf_counter()
            try:
                observation = await self._observe(ctx)
            finally:
                self._record_phase_latency("observe", perf_counter() - observe_start, severity=ResultSeverity.INFO)
            self._current_user_id = observation["user_id"]
            plan_start = perf_counter()
            try:
                plan = await self._plan(ctx, observation)
            finally:
                self._record_phase_latency("plan", perf_counter() - plan_start, severity=ResultSeverity.INFO)
            act_start = perf_counter()
            try:
                result = await self._act(ctx, observation, plan)
            finally:
                self._record_phase_latency("act", perf_counter() - act_start, severity=self._act_phase_severity)
            AgentResultModel.model_validate(result)
            return result
        finally:
            cycle_duration = perf_counter() - cycle_start
            cycle_severity = ResultSeverity.INFO
            if result is None or result["status"] is ResultStatus.FAILURE:
                cycle_severity = ResultSeverity.CRITICAL
            self._emit_cycle_metrics(cycle_duration, cycle_severity)

    async def _perform_web_verification(self, ctx: ExecutionContext) -> WebVerificationEvidence:
        return await self._web_verifier.verify(ctx, trace_id=self._trace_id)

    def _initialize_run_context(self, ctx: ExecutionContext) -> None:
        self._current_context = ctx
        self._current_user_id = self._extract_user_id(ctx)

    # ------------------------------------------------------------------ #
    # Observe
    # ------------------------------------------------------------------ #
    @retry_async(max_attempts=RETRY_MAX_ATTEMPTS)
    @audit_event("tool_agent.observe")
    @log_action
    @with_timeout(seconds=ASYNC_TIMEOUT_SEC)
    async def _observe(self, ctx: ExecutionContext) -> ObservationPayload:
        telemetry: Dict[str, Any] = {}
        query_registry = self._load_query_registry()
        for query_name, entry in query_registry.items():
            query_cls = self._resolve_object(entry["query"])
            query_kwargs = self._build_query_kwargs(query_cls, ctx)
            try:
                query_instance = query_cls(**query_kwargs)  # type: ignore[call-arg]
            except TypeError as exc:
                self._logger.error("query_init_failed", extra={"query": query_name, "error": str(exc)})
                continue
            telemetry_start = perf_counter()
            try:
                response = await self._query_bus.dispatch(query_instance)
                telemetry[query_name] = self._serialize_query_result(response)
            except Exception as exc:  # pragma: no cover - unexpected query failures
                self._logger.exception(
                    "query_dispatch_failed",
                    extra={"query": query_name, "error": str(exc)},
                )
            finally:
                duration = perf_counter() - telemetry_start
                self._observe_metric(
                    "agent_query_latency_seconds",
                    duration,
                    labels={"query": query_name},
                    component="tool_agent.observe",
                    severity=ResultSeverity.INFO,
                    latency_bucket="raw",
                )
        user_id = self._extract_user_id(ctx)
        balance = float(telemetry.get("GetUserBalanceQuery", {}).get("balance", 0.0))
        positions = list(telemetry.get("ListPositionsQuery", {}).get("positions", []))
        observation: ObservationPayload = {
            "trace_id": self._trace_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "balance": balance,
            "positions": positions,
            "telemetry": telemetry,
        }
        ObservationModel.model_validate(observation)
        return observation

    # ------------------------------------------------------------------ #
    # Plan
    # ------------------------------------------------------------------ #
    @retry_async(max_attempts=RETRY_MAX_ATTEMPTS)
    @audit_event("tool_agent.plan")
    @log_action
    @with_timeout(seconds=ASYNC_TIMEOUT_SEC)
    async def _plan(self, ctx: ExecutionContext, observation: ObservationPayload) -> ExecutionPlan:
        self._plan_invocations += 1
        command_mapping = self._load_command_mapping()
        bias_policy = self._load_bias_policy()
        knowledge_ratio = self._extract_knowledge_ratio(bias_policy)

        canonical_steps = self._build_canonical_steps(command_mapping, observation, ctx, knowledge_ratio[0])
        contrarian_steps = self._apply_contrarian_review(canonical_steps, observation, command_mapping)
        reframed_steps = self._apply_semantic_reframing(
            canonical_steps,
            observation,
            command_mapping,
            bias_policy,
        )
        merged_steps = self._merge_plan_variants(canonical_steps, reframed_steps, contrarian_steps)
        confidence = self._apply_confidence_calibration(merged_steps, knowledge_ratio)
        self._confidence_level = confidence

        bias_controls = {
            "contrarian_review": bool(bias_policy["cognitive_bias_controls"].get("contrarian_review", False)),
            "semantic_reframing": bias_policy["cognitive_bias_controls"].get("semantic_reframing"),
            "confidence_level": confidence,
            "knowledge_balance_ratio": knowledge_ratio,
        }
        plan: ExecutionPlan = {
            "steps": merged_steps,
            "bias_controls": bias_controls,
        }
        ExecutionPlanModel.model_validate(plan)
        self._pattern_frequency.update(step["command"] for step in merged_steps)
        self._audit_pattern_frequency(merged_steps, bias_policy)
        self._token_usage_estimate = self._estimate_token_usage(plan)
        return plan

    # ------------------------------------------------------------------ #
    # Act
    # ------------------------------------------------------------------ #
    @retry_async(max_attempts=RETRY_MAX_ATTEMPTS)
    @audit_event("tool_agent.act")
    @log_action
    @with_timeout(seconds=ASYNC_TIMEOUT_SEC)
    async def _act(
        self,
        ctx: ExecutionContext,
        observation: ObservationPayload,
        plan: ExecutionPlan,
    ) -> AgentResult:
        failures = 0
        executed_steps: List[PlanStep] = []
        self._act_phase_severity = ResultSeverity.INFO
        for step in plan["steps"]:
            try:
                pre_check_severity = await self._invoke_risk_pre_check(ctx, step, observation)
            except RiskViolationError as risk_error:
                failures += 1
                violation_severity = risk_error.severity
                if violation_severity is None:
                    violation_severity = ResultSeverity.CRITICAL
                self._logger.error(
                    "risk_pre_check_violation",
                    extra={
                        "command": step["command"],
                        "severity": self._severity_label(violation_severity),
                        "decision": str(risk_error.decision),
                    },
                )
                self._update_act_severity(violation_severity)
                self._increment_metric(
                    "tool_agent_risk_failure_total",
                    labels={"policy": "risk_pre_check_violation"},
                    component="tool_agent.act",
                    severity=violation_severity,
                    error_category="agent_error",
                )
                break
            if pre_check_severity == ResultSeverity.WARN:
                self._logger.warning(
                    "risk_pre_check_warn_step",
                    extra={
                        "command": step["command"],
                        "severity": self._severity_label(pre_check_severity),
                    },
                )
                self._update_act_severity(pre_check_severity)
            execution_start = perf_counter()
            command_instance = self._instantiate_command(step, ctx, observation)
            try:
                command_result = await self._command_bus.dispatch(command_instance)
                severity = getattr(command_result, "severity", ResultSeverity.INFO)
                status = ResultStatus.SUCCESS
            except CommandNotRegisteredError as exc:
                self._logger.warning(
                    "command_not_registered",
                    extra={"command": step["command"], "error": str(exc)},
                )
                command_result = None
                severity = ResultSeverity.CRITICAL
                status = ResultStatus.FAILURE
            except Exception as exc:  # pragma: no cover - safety net
                self._logger.exception(
                    "command_dispatch_failed",
                    extra={"command": step["command"], "error": str(exc)},
                )
                command_result = None
                severity = ResultSeverity.CRITICAL
                status = ResultStatus.FAILURE
            self._update_act_severity(severity)
            duration = perf_counter() - execution_start
            self._observe_metric(
                "tool_agent_command_latency_seconds",
                duration,
                labels={"command": step["command"]},
                component="tool_agent.act",
                severity=severity,
                latency_bucket="raw",
                error_category="agent_error" if severity is ResultSeverity.CRITICAL else "unknown_error",
            )
            tool_result = self._compose_tool_result(step, status, severity, command_instance, command_result, duration)
            self._execution_trace.append(tool_result)
            await self._record_audit_events(step, tool_result)
            self._update_failure_counters(status, step["command"])
            executed_steps.append(step)
            if severity == ResultSeverity.CRITICAL or self._consecutive_failures >= CIRCUIT_BREAKER_FAILURE_THRESHOLD:
                await self._trigger_kill_switch(ctx, observation, step["command"], reason="critical_or_threshold")
                break
            if status == ResultStatus.FAILURE:
                failures += 1
        self._final_output = self._summarize_execution(executed_steps, failures)
        self._decision_traceability = (
            "verified" if failures == 0 and self._consecutive_failures == 0 else "unverifiable"
        )
        result: AgentResult = {
            "trace_id": self._trace_id,
            "status": ResultStatus.SUCCESS if failures == 0 else ResultStatus.FAILURE,
            "confidence": self._confidence_level,
            "final_output": self._final_output,
            "executed_steps": executed_steps,
            "decision_traceability": self._decision_traceability,
        }
        AgentResultModel.model_validate(result)
        await self._persist_state()
        return result

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #
    def _resolve_logger_factory(
        self,
        injected: LoggerFactoryProtocol | None,
    ) -> LoggerFactoryProtocol:
        if injected is not None:
            return cast(LoggerFactoryProtocol, injected)
        candidate = getattr(self._container, "logger_factory", None)
        if candidate is None:
            raise RuntimeError("logger_factory non disponibile nel DIContainer")
        return cast(LoggerFactoryProtocol, candidate)

    def _resolve_audit_publisher(self, injected: AuditPublisherProtocol | None) -> AuditPublisherProtocol:
        if injected is not None:
            return injected
        candidate = getattr(self._container, "audit_publisher", None)
        if candidate is None:
            raise RuntimeError("audit_publisher non disponibile nel DIContainer")
        return cast(AuditPublisherProtocol, candidate)

    def _resolve_metrics_registry(self, injected: MetricsRegistryProtocol | None) -> MetricsRegistryProtocol:
        if injected is not None:
            return injected
        candidate = getattr(self._container, "metrics_registry", None)
        if candidate is None:
            raise RuntimeError("metrics_registry non disponibile nel DIContainer")
        return cast(MetricsRegistryProtocol, candidate)

    def _resolve_command_bus(self, injected: CommandBusProtocol | None) -> CommandBusProtocol:
        if injected is not None:
            return injected
        provider = di_provider()
        if provider.is_registered("command_bus"):
            return cast(CommandBusProtocol, provider.resolve("command_bus"))
        candidate = getattr(self._container, "command_bus", None)
        if candidate is None:
            raise RuntimeError("command_bus non disponibile via DI")
        return cast(CommandBusProtocol, candidate)

    def _resolve_query_bus(self, injected: QueryBusProtocol | None) -> QueryBusProtocol:
        if injected is not None:
            return injected
        provider = di_provider()
        if provider.is_registered("query_bus"):
            return cast(QueryBusProtocol, provider.resolve("query_bus"))
        candidate = getattr(self._container, "query_bus", None)
        if candidate is None:
            raise RuntimeError("query_bus non disponibile via DI")
        return cast(QueryBusProtocol, candidate)

    def _resolve_risk_engine(self, injected: RiskEngineProtocol | None) -> RiskEngineProtocol:
        if injected is not None:
            return injected
        candidate = getattr(self._container, "risk_center", None)
        if candidate is None:
            raise RuntimeError("risk_engine non disponibile via DI")
        return cast(RiskEngineProtocol, candidate)

    def _require_dependency(
        self,
        injected: Any,
        attribute_name: str,
    ) -> Any:
        if injected is not None:
            return injected
        candidate = getattr(self._container, attribute_name, None)
        if candidate is None:
            raise RuntimeError(f"{attribute_name} non disponibile via DIContainer")
        return candidate

    @staticmethod
    def _extract_user_id(ctx: ExecutionContext) -> int:
        extras = cast(Mapping[str, Any], ctx.extras)
        raw_user = extras.get("user_id")
        if raw_user is None:
            return 0
        if isinstance(raw_user, int):
            return raw_user
        if isinstance(raw_user, str) and raw_user.isdigit():
            return int(raw_user)
        return 0

    def _load_bias_policy(self) -> dict[str, Any]:
        if self._bias_policy_cache is None:
            path = self._root_path / self._BIAS_CONFIG_FILE
            yaml_adapter = self._require_yaml_adapter()
            self._bias_policy_cache = cast(dict[str, Any], yaml_adapter.load(path.read_text()))
        return self._bias_policy_cache

    def _load_command_mapping(self) -> CommandMapping:
        if self._command_mapping_cache is None:
            path = self._root_path / self._COMMAND_MAPPING_FILE
            yaml_adapter = self._require_yaml_adapter()
            self._command_mapping_cache = cast(CommandMapping, yaml_adapter.load(path.read_text()))
        return self._command_mapping_cache

    def _load_event_manifest(self) -> dict[str, Any]:
        if self._event_manifest_cache is None:
            path = self._root_path / self._EVENT_MANIFEST_FILE
            yaml_adapter = self._require_yaml_adapter()
            self._event_manifest_cache = cast(dict[str, Any], yaml_adapter.load(path.read_text()))
        return self._event_manifest_cache

    def _load_query_registry(self) -> dict[str, dict[str, str]]:
        if self._query_registry_cache is None:
            path = self._root_path / self._QUERY_REGISTRY_FILE
            yaml_adapter = self._require_yaml_adapter()
            self._query_registry_cache = cast(dict[str, dict[str, str]], yaml_adapter.load(path.read_text()))
        return self._query_registry_cache

    def _require_yaml_adapter(self) -> Any:
        yaml_adapter = getattr(self._container, "yaml_adapter", None)
        if yaml_adapter is None:
            raise RuntimeError("yaml_adapter non configurato nel DIContainer")
        return yaml_adapter

    def _build_query_kwargs(self, query_cls: type, ctx: ExecutionContext) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        try:
            signature_params = inspect.signature(query_cls).parameters
        except (TypeError, ValueError):
            return kwargs
        for name, param in signature_params.items():
            if name == "user_id":
                kwargs[name] = self._extract_user_id(ctx)
            elif name == "trace_id":
                kwargs[name] = self._trace_id
            elif name == "query_id":
                kwargs[name] = uuid.uuid4().hex
            elif param.default is inspect.Parameter.empty:
                self._logger.debug(
                    "query_param_unmapped",
                    extra={"param": name, "query": query_cls.__name__},
                )
        return kwargs

    @staticmethod
    def _serialize_query_result(result: Any) -> dict[str, Any]:
        if hasattr(result, "model_dump"):
            return cast(dict[str, Any], result.model_dump())
        if hasattr(result, "__dict__"):
            return {key: value for key, value in result.__dict__.items() if not key.startswith("_")}
        return {"raw": str(result)}

    @staticmethod
    def _process_common_fragment(
        *,
        command_name: str,
        metadata: Mapping[str, str],
        parameters: dict[str, Any],
        origin: CommonFragmentOrigin,
    ) -> PlanStep:
        """Generate a PlanStep with consistent metadata to avoid duplicated fragments."""
        # TypedDicts are not callable  return a plain dict matching the PlanStep shape
        return {
            "command": command_name,
            "event": metadata["event"],
            "metric": metadata["metric"],
            "parameters": parameters,
            "origin": origin,
        }

    def _build_canonical_steps(
        self,
        command_mapping: CommandMapping,
        observation: ObservationPayload,
        ctx: ExecutionContext,
        canonical_ratio: float,
    ) -> list[PlanStep]:
        commands = list(command_mapping.items())
        canonical_count = max(1, int(len(commands) * canonical_ratio)) if commands else 0
        canonical_items = commands[:canonical_count] if canonical_count else commands
        steps: PlanSteps = []
        for command_name, meta in canonical_items:
            parameters = self._build_command_parameters(command_name, observation, ctx)
            steps.append(
                self._process_common_fragment(
                    command_name=command_name,
                    metadata=meta,
                    parameters=parameters,
                    origin="canonical",
                )
            )
        return steps

    def _apply_contrarian_review(
        self,
        canonical_steps: PlanSteps,
        observation: ObservationPayload,
        command_mapping: CommandMapping,
    ) -> PlanSteps:
        policy = self._load_bias_policy()
        if not policy["cognitive_bias_controls"].get("contrarian_review", False):
            return canonical_steps
        remaining_commands = [
            (command, meta)
            for command, meta in command_mapping.items()
            if all(step["command"] != command for step in canonical_steps)
        ]
        alt_steps: PlanSteps = []
        for command_name, meta in remaining_commands:
            parameters = self._build_command_parameters(command_name, observation, None)
            alt_steps.append(
                self._process_common_fragment(
                    command_name=command_name,
                    metadata=meta,
                    parameters=parameters,
                    origin="contrarian",
                )
            )
        if not alt_steps and canonical_steps:
            for step in reversed(canonical_steps):
                metadata = {"event": step["event"], "metric": step["metric"]}
                alt_steps.append(
                    self._process_common_fragment(
                        command_name=step["command"],
                        metadata=metadata,
                        parameters=dict(step["parameters"]),
                        origin="contrarian",
                    )
                )
        return alt_steps

    def _apply_semantic_reframing(
        self,
        canonical_steps: PlanSteps,
        observation: ObservationPayload,
        command_mapping: CommandMapping,
        policy: dict[str, Any],
    ) -> PlanSteps:
        reframing_mode = policy["cognitive_bias_controls"].get("semantic_reframing")
        if reframing_mode != "every_3_iterations":
            return canonical_steps
        if self._plan_invocations % 3 != 0:
            return canonical_steps
        diversified_steps: PlanSteps = []
        reversed_positions = list(reversed(observation["positions"]))
        for step in canonical_steps:
            parameters = dict(step["parameters"])
            if reversed_positions:
                parameters["positions"] = reversed_positions
            metadata = command_mapping.get(
                step["command"],
                {"event": step["event"], "metric": step["metric"]},
            )
            diversified_steps.append(
                self._process_common_fragment(
                    command_name=step["command"],
                    metadata=metadata,
                    parameters=parameters,
                    origin="reframed",
                )
            )
        return diversified_steps or canonical_steps

    @staticmethod
    def _merge_plan_variants(
        canonical: PlanSteps,
        reframed: PlanSteps,
        contrarian: PlanSteps,
    ) -> PlanSteps:
        merged: PlanSteps = []
        for sequence in (canonical, reframed, contrarian):
            for step in sequence:
                if not any(existing["command"] == step["command"] for existing in merged):
                    merged.append(step)
        return merged

    def _apply_confidence_calibration(
        self,
        steps: PlanSteps,
        knowledge_ratio: tuple[float, float],
    ) -> str:
        policy = self._load_bias_policy()["cognitive_bias_controls"]
        if policy.get("confidence_calibration") != "enforced":
            return "media"
        canonical_count = sum(1 for step in steps if step["origin"] == "canonical")
        exploratory_count = sum(1 for step in steps if step["origin"] != "canonical")
        total = max(1, canonical_count + exploratory_count)
        canonical_ratio = canonical_count / total
        exploratory_ratio = exploratory_count / total
        if canonical_ratio >= knowledge_ratio[0] and exploratory_ratio >= knowledge_ratio[1]:
            return "alta"
        if canonical_ratio >= knowledge_ratio[0]:
            return "media"
        return "bassa"

    def _audit_pattern_frequency(self, steps: PlanSteps, policy: dict[str, Any]) -> None:
        if not policy["cognitive_bias_controls"].get("pattern_frequency_audit", False):
            return
        for step in steps:
            frequency = self._pattern_frequency[step["command"]]
            if frequency >= 3:
                self._logger.warning(
                    "pattern_frequency_threshold",
                    extra={
                        "command": step["command"],
                        "frequency": frequency,
                    },
                )

    async def _invoke_risk_pre_check(
        self,
        ctx: ExecutionContext,
        step: PlanStep,
        observation: ObservationPayload,
    ) -> ResultSeverity:
        payload = {
            "command": step["command"],
            "trace_id": self._trace_id,
            "user_id": observation["user_id"],
            "balance": observation["balance"],
            "positions": observation["positions"],
        }
        try:
            decision = await self._risk_engine.pre_check(ctx, payload)  # type: ignore[arg-type]
        except TypeError:
            tool_input = {
                "trace_id": self._trace_id,
                "user_roles": [ctx.role],
                "payload": payload,
            }
            decision = await self._risk_engine.pre_check(tool_input)  # type: ignore[arg-type]
        except Exception as exc:
            self._logger.exception(
                "risk_pre_check_failed",
                extra={"command": step["command"], "error": str(exc)},
            )
            self._increment_metric(
                "tool_agent_risk_failure_total",
                labels={"policy": "risk_pre_check_exception"},
                component="tool_agent.act",
                severity=ResultSeverity.CRITICAL,
                error_category="external_dependency_error",
            )
            await self._trigger_kill_switch(ctx, observation, step["command"], reason="risk_pre_check_exception")
            raise RiskViolationError(
                "Risk engine pre-check raised an exception before execution.",
                severity=ResultSeverity.CRITICAL,
                decision=None,
            ) from exc

        severity = self._normalize_severity(decision)
        self._increment_metric(
            "tool_agent_risk_precheck_total",
            labels={"risk_level": self._severity_label(severity)},
            component="tool_agent.act",
            severity=severity,
            error_category="agent_error" if severity is ResultSeverity.CRITICAL else "unknown_error",
        )
        if severity == ResultSeverity.CRITICAL:
            self._logger.critical(
                "risk_pre_check_blocked",
                extra={
                    "command": step["command"],
                    "severity": self._severity_label(severity),
                    "decision": str(decision),
                },
            )
            self._consecutive_failures = max(self._consecutive_failures, CIRCUIT_BREAKER_FAILURE_THRESHOLD)
            await self._trigger_kill_switch(ctx, observation, step["command"], reason="risk_pre_check_critical")
            raise RiskViolationError(
                "Risk engine pre-check returned CRITICAL severity.",
                severity=severity,
                decision=decision,
            )
        if severity == ResultSeverity.WARN:
            self._logger.warning(
                "risk_pre_check_warning",
                extra={
                    "command": step["command"],
                    "severity": self._severity_label(severity),
                    "decision": str(decision),
                },
            )
        return severity

    def _instantiate_command(
        self,
        step: PlanStep,
        ctx: ExecutionContext,
        observation: ObservationPayload,
    ) -> CommandBase:
        try:
            command_cls = self._resolve_command_class(step["command"])
            return command_cls(**step["parameters"])  # type: ignore[call-arg]
        except Exception as exc:
            self._logger.debug(
                "dynamic_command_fallback",
                extra={"command": step["command"], "error": str(exc)},
            )
            payload = dict(step["parameters"])
            payload.setdefault("trace_id", self._trace_id)
            payload.setdefault("user_id", observation["user_id"])
            payload.setdefault("requested_by", ctx.role)
            return DynamicCommand(command_type=step["command"], payload=payload)

    def _resolve_command_class(self, command_name: str) -> type:
        snake = self._camel_to_snake(command_name)
        module_candidates = [
            f"bot_crypto.application.commands.{snake}",
            f"bot_crypto.application.commands.{snake}_command",
            f"bot_crypto.application.commands.{snake}_cmd",
        ]
        for module_path in module_candidates:
            try:
                module = import_module(module_path)
            except ModuleNotFoundError:
                continue
            if hasattr(module, command_name):
                return getattr(module, command_name)
        raise ImportError(f"Command {command_name} non trovato nei moduli: {module_candidates}")

    @staticmethod
    def _compose_tool_result(
        step: PlanStep,
        status: ResultStatus,
        severity: ResultSeverity,
        command_instance: CommandBase,
        command_result: Any,
        duration: float,
    ) -> ToolResult:
        # ToolResult is a TypedDict; construct and return a plain dict with the expected keys
        result_data: dict[str, Any] = {
            "command": step["command"],
            "duration": duration,
            "payload": getattr(command_instance, "__dict__", {}),
            "command_result": getattr(command_result, "__dict__", {}) if command_result else {},
        }
        return {
            "status": status,
            "severity": severity,
            "data": result_data,
        }

    async def _record_audit_events(self, step: PlanStep, tool_result: ToolResult) -> None:
        manifest = self._load_event_manifest()
        manifest_events = {entry["event"] for entry in manifest.get("events", [])}
        if step["event"] not in manifest_events:
            self._logger.warning(
                "event_not_in_manifest",
                extra={"event": step["event"]},
            )
        event_payload = {
            "event": step["event"],
            "trace_id": self._trace_id,
            "event_id": uuid.uuid4().hex,
            "step": step["command"],
            "result": tool_result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self._audit.publish(event_payload)

    def _update_failure_counters(self, status: ResultStatus, command_name: str) -> None:
        if status == ResultStatus.FAILURE:
            self._consecutive_failures += 1
            self._increment_metric(
                "tool_call_failure_total",
                labels={"tool": command_name},
                component="tool_agent.act",
                severity=ResultSeverity.CRITICAL,
                error_category="agent_error",
            )
            self._increment_metric(
                "tool_agent_risk_failure_total",
                labels={"policy": "command_failure"},
                component="tool_agent.act",
                severity=ResultSeverity.CRITICAL,
                error_category="agent_error",
            )
        else:
            self._consecutive_failures = 0
            self._increment_metric(
                "tool_call_success_total",
                labels={"tool": command_name},
                component="tool_agent.act",
                severity=ResultSeverity.INFO,
                error_category="unknown_error",
            )

    async def _trigger_kill_switch(
        self,
        ctx: ExecutionContext,
        observation: ObservationPayload,
        command_name: str,
        reason: str,
    ) -> None:
        self._increment_metric(
            "tool_agent_risk_failure_total",
            labels={"policy": reason},
            component="tool_agent.act",
            severity=ResultSeverity.CRITICAL,
            error_category="agent_error",
        )
        kill_parameters = {"reason": reason, "source_command": command_name, "trace_id": self._trace_id}
        kill_step = self._process_common_fragment(
            command_name="KillExecutionProcessCommand",
            metadata={
                "event": "StrategyKillEvent",
                "metric": "command.kill_execution_process.count",
            },
            parameters=kill_parameters,
            origin="system",
        )
        kill_command = self._instantiate_command(kill_step, ctx=ctx, observation=observation)
        try:
            await self._command_bus.dispatch(kill_command)
        except Exception as exc:  # pragma: no cover - fallback
            self._logger.exception(
                "kill_switch_dispatch_failed",
                extra={"error": str(exc)},
            )
        await self._audit.publish(
            {
                "event": "StrategyKillEvent",
                "trace_id": self._trace_id,
                "reason": reason,
                "source": command_name,
            }
        )

    def _build_command_parameters(
        self,
        command_name: str,
        observation: ObservationPayload,
        ctx: ExecutionContext | None,
    ) -> dict[str, Any]:
        parameters: dict[str, Any] = {
            "trace_id": self._trace_id,
            "user_id": observation["user_id"],
            "timestamp": observation["timestamp"],
            "positions": observation["positions"],
            "balance": observation["balance"],
        }
        if ctx is not None:
            parameters["requested_by"] = ctx.role
            parameters["tenant_id"] = ctx.tenant_id
        if command_name.lower().startswith("registerrisk"):
            parameters["severity"] = ResultSeverity.CRITICAL.value
        return parameters

    @staticmethod
    def _summarize_execution(executed_steps: PlanSteps, failures: int) -> str:
        executed_names = ",".join(step["command"] for step in executed_steps) or "none"
        status = "ok" if failures == 0 else f"failures={failures}"
        return f"steps={executed_names}; status={status}"

    async def _persist_state(self) -> None:
        await self._state_repo.write_snapshot(self.serialize())
        await self._ledger.publish_hash(self.serialize()["sha256"])

    def _reset_state(self) -> None:
        self._execution_trace.clear()
        self._final_output = ""
        self._consecutive_failures = 0
        self._decision_traceability = "unverifiable"
        self._web_verification_evidence = None
        self._current_user_id = 0
        self._current_context = None
        self._token_usage_estimate = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        self._act_phase_severity = ResultSeverity.INFO

    def serialize(self) -> dict[str, Any]:
        payload = {
            "trace_id": self._trace_id,
            "execution_trace": self._execution_trace.copy(),
            "final_output": self._final_output,
            "confidence": self._confidence_level,
            "decision_traceability": self._decision_traceability,
            "web_verification": self._web_verification_evidence,
        }
        normalized = self._normalize_for_digest(payload)
        digest = hashlib.sha256(json.dumps(normalized, sort_keys=True).encode("utf-8")).hexdigest()
        payload["sha256"] = digest
        return payload

    def load_state(self, state: dict[str, Any]) -> Self:
        self._trace_id = cast(str, state.get("trace_id", self._generate_trace_id()))
        self._execution_trace = cast(list[ToolResult], state.get("execution_trace", []))
        self._final_output = cast(str, state.get("final_output", ""))
        self._confidence_level = cast(str, state.get("confidence", "bassa"))
        self._decision_traceability = cast(str, state.get("decision_traceability", "unverifiable"))
        self._web_verification_evidence = cast(WebVerificationEvidence | None, state.get("web_verification"))
        return self

    def _aggregate_context(self) -> dict[str, str]:
        context: dict[str, str] = {}
        for result in self._execution_trace:
            command = cast(str, result["data"].get("command", "unknown"))
            context[command] = str(result["data"])
        return context

    @staticmethod
    def _generate_trace_id(seed: str | None = None) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        unique = uuid.uuid4().hex
        if seed:
            return f"{timestamp}-{seed[:8]}-{unique[:8]}"
        return f"{timestamp}-{unique}"

    def _observe_metric(
        self,
        name: str,
        value: float,
        *,
        labels: dict[str, str],
        component: str = "tool_agent",
        severity: ResultSeverity | str = ResultSeverity.INFO,
        latency_bucket: str = "raw",
        error_category: str = "unknown_error",
    ) -> None:
        metric_labels = self._build_metric_labels(
            component=component,
            severity=severity,
            latency_bucket=latency_bucket,
            error_category=error_category,
        )
        metric_labels.update(labels)
        self._metrics.observe(name, value, tags=metric_labels)

    def _set_gauge_metric(
        self,
        name: str,
        value: float,
        *,
        labels: dict[str, str],
        component: str = "tool_agent",
        severity: ResultSeverity | str = ResultSeverity.INFO,
        latency_bucket: str = "n/a",
        error_category: str = "unknown_error",
    ) -> None:
        metric_labels = self._build_metric_labels(
            component=component,
            severity=severity,
            latency_bucket=latency_bucket,
            error_category=error_category,
        )
        metric_labels.update(labels)
        self._metrics.gauge(name, value, tags=metric_labels)

    def _increment_metric(
        self,
        name: str,
        *,
        labels: dict[str, str],
        component: str = "tool_agent",
        severity: ResultSeverity | str = ResultSeverity.INFO,
        error_category: str = "unknown_error",
        latency_bucket: str = "n/a",
    ) -> None:
        metric_labels = self._build_metric_labels(
            component=component,
            severity=severity,
            latency_bucket=latency_bucket,
            error_category=error_category,
        )
        metric_labels.update(labels)
        self._metrics.increment(name, tags=metric_labels)

    def _build_metric_labels(
        self,
        *,
        component: str,
        severity: ResultSeverity | str,
        latency_bucket: str,
        error_category: str,
        event_id: str | None = None,
    ) -> Dict[str, str]:
        # Cast esplicito per soddisfare analyzer in ambienti che propagano Unions troppo ampie
        if isinstance(severity, ResultSeverity):
            severity_value = cast(ResultSeverity, severity).value
        else:
            severity_value = str(severity)

        return {
            "trace_id": self._trace_id,
            "event_id": event_id or uuid.uuid4().hex,
            "component": component,
            "severity": severity_value,
            "latency_bucket": latency_bucket,
            "error_category": error_category,
        }

    def _record_phase_latency(
        self,
        phase: str,
        duration: float,
        *,
        severity: ResultSeverity | str,
    ) -> None:
        samples = self._phase_latency_samples.setdefault(phase, [])
        samples.append(duration)
        if len(samples) > self._LATENCY_HISTORY_LIMIT:
            del samples[: len(samples) - self._LATENCY_HISTORY_LIMIT]
        for bucket, value in self._percentile_buckets(samples).items():
            self._observe_metric(
                METRIC_AGENT_LATENCY_SECONDS,
                value,
                labels={"phase": phase},
                component=f"tool_agent.{phase}",
                severity=severity,
                latency_bucket=bucket,
                error_category="unknown_error",
            )

    def _percentile_buckets(self, samples: List[float]) -> Dict[str, float]:
        if not samples:
            return {}
        ordered = sorted(samples)
        return {
            "p50": self._percentile(ordered, 50),
            "p95": self._percentile(ordered, 95),
            "p99": self._percentile(ordered, 99),
        }

    @staticmethod
    def _percentile(samples: List[float], percentile: int) -> float:
        if not samples:
            return 0.0
        if len(samples) == 1:
            return samples[0]
        rank = (percentile / 100) * (len(samples) - 1)
        lower = int(rank)
        upper = min(lower + 1, len(samples) - 1)
        weight = rank - lower
        return samples[lower] + (samples[upper] - samples[lower]) * weight

    def _emit_cycle_metrics(self, cycle_duration: float, severity: ResultSeverity) -> None:
        self._record_phase_latency("cycle", cycle_duration, severity=severity)
        self._observe_metric(
            "inference_latency_ms",
            cycle_duration * 1000.0,
            labels={"phase": "cycle"},
            component="tool_agent.cycle",
            severity=severity,
            latency_bucket="raw",
            error_category="unknown_error",
        )
        for metric_name in ("total_tokens", "input_tokens", "output_tokens"):
            value = float(self._token_usage_estimate.get(metric_name, 0))
            self._observe_metric(
                metric_name,
                value,
                labels={"phase": "cycle"},
                component="tool_agent.cycle",
                severity=severity,
                latency_bucket="aggregate",
                error_category="unknown_error",
            )
            self._set_gauge_metric(
                METRIC_TOKEN_USAGE_ESTIMATE,
                value,
                labels={"phase": "cycle", "metric": metric_name},
                component="tool_agent.cycle",
                severity=severity,
                error_category="unknown_error",
            )
        self._increment_metric(
            "prompt_type",
            labels={"phase": "cycle", "prompt_type": "react"},
            component="tool_agent.cycle",
            severity=severity,
            error_category="unknown_error",
        )

    @staticmethod
    def _estimate_token_usage(plan: ExecutionPlan) -> dict[str, int]:
        parameter_lengths = [
            len(str(value))
            for step in plan["steps"]
            for value in step["parameters"].values()
        ]
        input_tokens = sum(parameter_lengths)
        output_tokens = len(plan["steps"]) * 4
        total_tokens = input_tokens + output_tokens
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

    def _normalize_for_digest(self, data: Any) -> NormalizedValue:
        if isinstance(data, dict):
            return {key: self._normalize_for_digest(value) for key, value in sorted(data.items())}
        if isinstance(data, list):
            return [self._normalize_for_digest(item) for item in data]
        if isinstance(data, tuple):
            return [self._normalize_for_digest(item) for item in data]
        if isinstance(data, (str, int, float, bool)) or data is None:
            return data
        if isinstance(data, datetime):
            return data.astimezone(timezone.utc).isoformat()
        if isinstance(data, ResultStatus):
            return data.value
        if isinstance(data, ResultSeverity):
            return data.value
        if hasattr(data, "model_dump"):
            return self._normalize_for_digest(data.model_dump())
        if hasattr(data, "__dict__"):
            serialisable = {
                key: value
                for key, value in data.__dict__.items()
                if not key.startswith("_")
            }
            return self._normalize_for_digest(serialisable)
        return str(data)

    def _log_context_provider(self) -> dict[str, Any]:
        context = self._current_context
        span_id = "" if context is None or context.span_id is None else context.span_id
        session_id = "" if context is None else context.request_id
        return {
            "trace_id": self._trace_id,
            "span_id": span_id,
            "event_id": uuid.uuid4().hex,
            "user_id": str(self._current_user_id),
            "session_id": session_id,
            "toolagent_id": self._toolagent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _update_act_severity(self, severity: ResultSeverity) -> None:
        if self._SEVERITY_PRIORITY[severity] > self._SEVERITY_PRIORITY[self._act_phase_severity]:
            self._act_phase_severity = severity

    @staticmethod
    def _severity_label(value: ResultSeverity | None) -> str:
        if isinstance(value, ResultSeverity):
            return value.value
        return "unknown"

    @staticmethod
    def _normalize_severity(raw_decision: Any) -> ResultSeverity:
        def _coerce(value: Any) -> ResultSeverity | None:
            if value is None:
                return None
            if isinstance(value, ResultSeverity):
                return value
            if isinstance(value, ResultStatus):
                return ResultSeverity.CRITICAL if value is ResultStatus.FAILURE else ResultSeverity.INFO
            if isinstance(value, str):
                normalized = value.strip().replace("-", "_").upper()
                alias_map = {"WARNING": "WARN", "ERROR": "CRITICAL", "ALERT": "WARN", "HALT": "CRITICAL"}
                normalized = alias_map.get(normalized, normalized)
                try:
                    return ResultSeverity[normalized]
                except KeyError:
                    try:
                        return ResultSeverity(normalized)
                    except ValueError:
                        return None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if value >= 2:
                    return ResultSeverity.CRITICAL
                if value >= 1:
                    return ResultSeverity.WARN
                return ResultSeverity.INFO
            if isinstance(value, Mapping):
                keys = ("severity", "level", "status", "decision", "risk")
                for key in keys:
                    if key in value:
                        coerced = _coerce(value[key])
                        if coerced is not None:
                            return coerced
                for candidate in value.values():
                    coerced = _coerce(candidate)
                    if coerced is not None:
                        return coerced
                return None
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                for candidate in value:
                    coerced = _coerce(candidate)
                    if coerced is not None:
                        return coerced
                return None
            name_attr = getattr(value, "name", None)
            if isinstance(name_attr, str):
                return _coerce(name_attr)
            if hasattr(value, "value"):
                attr_value = getattr(value, "value")
                if isinstance(attr_value, str):
                    return _coerce(attr_value)
            return None

        severity = _coerce(raw_decision)
        return severity if severity is not None else ResultSeverity.INFO

    @staticmethod
    def _extract_knowledge_ratio(policy: dict[str, Any]) -> tuple[float, float]:
        ratio = policy["cognitive_bias_controls"].get(
            "knowledge_balance_ratio",
            riskc.KNOWLEDGE_BALANCE_RATIO_FALLBACK,
        )
        primary = (
            float(ratio[0]) if isinstance(ratio, Sequence) and len(ratio) >= 1 else riskc.KNOWLEDGE_BALANCE_PRIMARY_DEFAULT
        )
        exploratory = (
            float(ratio[1]) if isinstance(ratio, Sequence) and len(ratio) >= 2 else riskc.KNOWLEDGE_BALANCE_EXPLORATORY_DEFAULT
        )
        total = primary + exploratory
        if total == 0:
            default_total = riskc.KNOWLEDGE_BALANCE_PRIMARY_DEFAULT + riskc.KNOWLEDGE_BALANCE_EXPLORATORY_DEFAULT
            if default_total == 0:
                return (0.5, 0.5)
            return (
                riskc.KNOWLEDGE_BALANCE_PRIMARY_DEFAULT / default_total,
                riskc.KNOWLEDGE_BALANCE_EXPLORATORY_DEFAULT / default_total,
            )
        return (primary / total, exploratory / total)

    @staticmethod
    def _resolve_object(dotted_path: str) -> Any:
        module_name, _, attr = dotted_path.rpartition(".")
        if not module_name:
            raise ImportError(f"Percorso non valido: {dotted_path}")
        # Accept both fully-qualified paths starting with `bot_crypto.` or relative
        module_path = module_name if module_name.startswith("bot_crypto.") else f"bot_crypto.{module_name}"
        try:
            module = import_module(module_path)
        except ModuleNotFoundError as exc:
            raise ImportError(f"Modulo non trovato: {module_path}") from exc
        if not hasattr(module, attr):
            raise ImportError(f"Attributo {attr} non trovato in {module_path}")
        return getattr(module, attr)

    @staticmethod
    def _camel_to_snake(name: str) -> str:
        characters: list[str] = []
        for idx, char in enumerate(name):
            if char.isupper() and idx > 0:
                characters.append("_")
            characters.append(char.lower())
        return "".join(characters)
