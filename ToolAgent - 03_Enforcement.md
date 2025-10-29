<!-- ToolAgent Specification Block. No placeholder permitted. -->
<!-- hash: sha256-97536eedca98cd95d88dfd2cbc86fdc950f80bd05ad014f1cc7f1017440f7a97 -->
## 6. Enforcement Modules
> Root Path Reference: `./`
> Config Reference: `./ToolAgent - 06_Cognitive_Bias_Prevention.yaml`


### 6.1 enforcement::static

Analisi statica `mypy --strict`, `pylint`, `bandit` e AST hooks. Nessun `Any` non tipizzato e import senza side effects. xref:2.7

### 6.2 enforcement::decorator_validation

Validazione centralizzata di decoratori e protocolli runtime. Gli import devono provenire da `infrastructure/decorators`. xref:2.7

### 6.3 enforcement::ci_hooks

Hook CI per test, lint e coverage. Blocchi su placeholder o coverage <95%. xref:2.6

### 6.4 enforcement::dependency_registry

Gestione delle dipendenze tramite Hatch e registry YAML. Controlli `pip-audit`, `license-checker`

---

### enforcement::cognitive_integrity

Integrazione dei controlli cognitivi provenienti da  
`./ToolAgent - 06_Cognitive_Bias_Prevention.yaml`.  
Ogni modulo di enforcement (static, decorator, CI e dependency registry)
deve implementare i controlli di:

- **falsification_check**
- **semantic_reframing**
- **decision_traceability**

Tali controlli vengono applicati durante i cicli CI/CD per validare la coerenza
cognitiva e ridurre i bias sistematici nel codice generato o rifattorizzato.

#### ReconciliationValidator

**Scopo**  
Garantire riconciliazione idempotente tra stato locale e stato reale (positions, balances, open orders) su tutte le venue.

**Politiche**  
- Frequenza: schedulata (es. ogni N secondi) e on-demand su evento anomalo.  
- Idempotenza: stesso delta => stessa sequenza di fix; no doppie azioni.  
- Coerenza forte su chiusure/annullamenti; soft su bilanci frammentati, con quarantena differenze minime.

**Flusso**  
1. Snapshot locale + fetch stato venue.  
2. Diff tipizzata: `{created, missing, mismatched, stale}`.  
3. Piano d'azione deterministico: `cancel`, `close`, `resync`, `quarantine`.  
4. Pubblicazione `ReconciliationEvent` su EventBus con `trace_id`.

**Metriche & Audit**  
- `reconciliation_diffs_total{type}`  
- `reconciliation_fixes_total{action}`  
- Log/audit obbligatori: `trace_id, user_id, event_id, severity`.

**Failure Modes**  
- `CRITICAL`: divergenza su notional o posizioni aperte incoerenti.  
- `WARN`: ordini orfani senza impatto patrimoniale.  
- `INFO`: differenze di arrotondamento < <GREEK SMALL LETTER EPSILON> documentate.

**Vincoli**  
- Stop immediato del trading su `CRITICAL` fino a esito riconciliazione.  
- Nessuna modifica fuori dai casi codificati; tutte le azioni sono tracciate.
