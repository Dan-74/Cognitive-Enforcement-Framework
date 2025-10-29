<!-- ToolAgent Specification Block. No placeholder permitted. -->
<!-- hash: sha256-a02f35f946a7cf33605fc55b998dde74d47191f975fa6bd8f2bb56025bbfd4e6 -->
## 3. ToolAgent Specification (NASA/ONU/NIST-Grade, SSoT-Aligned)
> Root Path Reference: `./`
> Config Reference: `./ToolAgent - 06_Cognitive_Bias_Prevention.yaml`


### 3.1 Context and Purpose

`ToolAgent` collega l'LLM all'ambiente crypto-bot con flusso **observe -> plan -> act**.

### 3.2 Component Overview

ReAct loop su architettura Hexagonal + Clean con uso di `asyncio`, DI e Prometheus.
**Reference:**  
Tutti i componenti del sistema devono rispettare le regole cognitive definite nel modulo `./ToolAgent - 06_Cognitive_Bias_Prevention.yaml`, inclusa la calibrazione della confidenza e la revisione antitetica (contrarian review).


### 3.3 Canonical Types (SSoT)

```python
from typing import TypedDict

class ToolInput(TypedDict, total=True):
    trace_id: str
    user_roles: list[str]
    payload: dict[str, str | int | float | bool | None]
```

(vedi file completo per ulteriori tipi).

### 3.4 Centralised Constants

Tutti i literal risiedono in `bot_crypto.infrastructure.definitions.constants`.

### 3.5 Result Enums

`ResultStatus` e `ResultSeverity` normalizzano gli esiti senza raw string.

### 3.6 Public API

`run(prompt, *, tools, trace_id=None, user_roles=None)` con RBAC e metodi ausiliari `_observe`, `_plan`, `_act`.

### 3.7 Risk & Kill-Switch Integration

Check pre-trade via `RiskEngine.pre_check`. Escalation con `KillProcessCommand` su failure critici.

### 3.8 Snapshot & Immutable Ledger

Decorator `@snapshot_every(n_steps=5)` scrive lo stato e pubblica l'hash su ledger immutabile.

### 3.9 Metrics & Logging

Metriche:
- `agent_execution_latency_seconds`
- `tool_call_success_total`
Log includono `trace_id`, `event_id`, `severity`, `user_id`.

### 3.10 Command & Event Registry

Definito in `config/data/infrastructure/command_registry.yaml` e `config/data/global/event_manifest.yaml`. xref:3.10

### 3.11 Example Usage

```python
response = await agent.run("Translate and upload the document", tools=[TranslateTool(), UploadTool()], user_roles=["trader"])
print(response["final_output"])
```

---

### Cognitive Bias Prevention Integration

Tutti i componenti descritti in questo documento operano in conformita con
`./ToolAgent - 06_Cognitive_Bias_Prevention.yaml`, applicando controlli di:

- **Confidence Calibration** sui risultati intermedi,
- **Contrarian Review Step** sulle decisioni automatizzate,
- **Pattern Frequency Audit** per ridurre l'overfitting semantico.

L'obiettivo e garantire che ogni componente agisca in modo epistemicamente
trasparente e verificabile, secondo le linee guida ONU/NIST/UNESCO 2025.

#### OrderNormalizerAdapter

**Scopo**  
Normalizzare quantita, precisione (decimali), `minNotional` e `lotSize` per tutte le venue, prevenendo rifiuti d'ordine e garantendo conformita ai vincoli di exchange.

**Responsabilita chiave**  
- Mappatura per-venue di: tickSize, stepSize, minQty, minNotional, maxLeverage (read-only).  
- Arrotondamento deterministico (bankers-rounding vietato) con regole esplicite per quantita/prezzo.  
- Validazioni pre-trade: quantita non nulla, prezzo non nullo, rispetto `minNotional`.  
- Integrazione con router e risk-engine per calcolo quantita finale post-risk.

**Input -> Output**  
- Input: `{symbol, side, price, qty_raw, venue_capabilities}`  
- Output: `{price_norm, qty_norm, notional_norm, compliance_flags[]}`

**Metriche (Prometheus)**  
- `order_normalizer_adjustments_total{venue,symbol}`  
- `order_rejects_prevented_total{venue,reason}`

**Vincoli**  
- No placeholder/magic number (vincoli globali).  
- Idempotenza: stessa richiesta => stesso output a parita di capabilities cache.  
- Niente stato mutabile locale; capabilities da registry/DI cache.

#### DynamicVenueRouter

**Scopo**  
Selezionare runtime la venue ottimale sulla base di costo atteso, latenza, affidabilita e slippage stimato.

**Punteggio composito (0-1)**  
`score = w_cost.S_cost + w_latency.S_latency + w_slippage.S_slippage + w_reliability.S_reliability`  
- `S_cost`: fee maker/taker + funding atteso.  
- `S_latency`: RTT ordine (p50/p95) e jitter.  
- `S_slippage`: impatto stimato vs orderbook (depth, spread, volatility).  
- `S_reliability`: tasso errori API, desync WS, incidenti recenti.

**Responsabilita**  
- Aggiornamento pesi `w_*` via configurazione versionata (governance parametri).  
- Fallback deterministico in caso di parita.  
- Esposizione spiegazioni (`why_this_venue`) per auditability.

**Input -> Output**  
- Input: `{symbol, side, qty, price_hint, constraints}`  
- Output: `{venue_selected, score_breakdown, route_constraints}`

**Metriche**  
- `venue_router_selection_total{venue}`  
- `venue_router_override_total{reason}`

**Vincoli**  
- Nessuna istanza locale di client: passa da DI/registry.  
- Hard-kill su venue con `reliability < <GREEK SMALL LETTER THETA>_fail` (policy enforcement).  
- Niente side-effects sugli import.
