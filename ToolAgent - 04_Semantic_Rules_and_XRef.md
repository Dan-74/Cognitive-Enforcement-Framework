<!-- ToolAgent Specification Block. No placeholder permitted. -->
<!-- hash: sha256-6ea8f266544ead9761ef6328b3774aa839f1783a95df0f2e067788739a706fe1 -->

## 4. Regole Semantiche e Convenzioni di Naming

Riferimento ai file `docs/semantic_check/ci_quality_gate.md` e `docs/semantic_check/naming_glossary.md`.

## 5. Prompt di Verifica Incrociata di Coerenza Strutturale e Semantica

> Root Path Reference: `./`
> Config Reference: `./ToolAgent - 06_Cognitive_Bias_Prevention.yaml`


Prompt standard per validare struttura, naming e auditabilitaa. Per i dettagli sulle verifiche automatiche consultare
xref:6.3.

---

### Cognitive Trace Link

Le regole semantiche e i prompt di verifica incrociata devono includere
i controlli cognitivi definiti in  
`./ToolAgent - 06_Cognitive_Bias_Prevention.yaml`, in particolare:

- **semantic_reframing** per mitigare il framing bias,
- **contrarian_review** per validare ipotesi alternative,
- **decision_traceability** per audit e revisione etica.

L'integrazione assicura che ogni verifica semantica sia anche una verifica cognitiva.

## 7. Moduli Principali e Responsabilita

| Percorso                         | Descrizione                       |
|----------------------------------|-----------------------------------|
| `domain/analysis`                | Algoritmi di mercato              |
| `domain/events`                  | Definizioni eventi per EventBus   |
| `application/usecases`           | Orchestratori CQRS                |
| `infrastructure/adapters`        | Implementazioni porte             |
| `infrastructure/di_container.py` | Wiring provider                   |
| `entrypoints/`                   | Bootstrap e healthcheck           |
| `tests/`                         | Struttura test con coverage >=95% |

## 11. Appendice B - Directory Overview

Dettagli sui moduli principali sono riportati in xref:7.

## 12. Appendice C - Cross-Section Mapping

Vedi file `xref_map.yaml` per la mappatura completa. Alcuni riferimenti principali:

- xref:3.10 -> config/data/global/event_manifest.yaml
- xref:6.1 -> enforcement::static
- xref:6.3 -> enforcement::ci_hooks
