<!-- ToolAgent Specification Block. No placeholder permitted. -->
<!-- hash: sha256-e4bea6fd051336ac461975a753aef23133b82d7c8ff907b44d745dc87307fb41 -->
## 8. Versioning e Gestione Dipendenze Python
> Root Path Reference: `./`
> Config Reference: `./ToolAgent - 06_Cognitive_Bias_Prevention.yaml`


### 8.1 Compatibilitaa Pydantic v2+ e importazioni

Import obbligatorio `from pydantic_settings import BaseSettings` con versioni dichiarate in `pyproject.toml`.

### 8.2 Error Prevention Rules

`mypy --strict` e hook AST per DIContainer e protocolli tipizzati. Regola assorbita da xref:6.1.

### 8.3 Dependency Management - Hatch, Audit, Isolation

Uso di Hatch come sistema unificato. Controlli di sicurezza `pip-audit`, `bandit` e verifica licenze. Vedi xref:6.4.

## 9. AGENT BEHAVIORAL CONSTRAINTS

Codice ASCII, niente placeholder, nessun hardcoded magic number e librerie sempre aggiornate.

---

### Appendice Integrativa - Cognitive Bias Prevention

**Modulo aggiuntivo riconosciuto:**  
`./ToolAgent - 06_Cognitive_Bias_Prevention.yaml`  
**Descrizione:** Controlli cognitivi e meta-razionali applicati al ciclo di vita del codice.  
**Versione:** 2025.10.19  
**Conformita:** ONU/NIST/UNESCO - AI Cognitive Integrity Framework.  

Questa appendice estende la ToolAgent Specification ai controlli metacognitivi,
assicurando che le pipeline CI/CD e i moduli di enforcement rispettino
principi di auditabilita cognitiva e trasparenza epistemica.

#### AuditExportModule

**Scopo**  
Produrre esportazioni firmate e certificabili per audit/compliance e adempimenti fiscali.

**Formati**  
- `JSONL` firmato (SHA-256 + firma del pacchetto); opzionale `CSV` derivato.  
- Timezone UTC, RFC3339; import safe in sistemi terzi.

**Campo dati minimi**  
`{timestamp, trace_id, event_id, user_id, venue, symbol, side, qty, price, notional, fees, pnl, leverage, risk_flags[], decision_outcome, version_manifest}`

**Provenance & SBOM**  
- Inclusione digest componenti (container image, commit, registry).  
- Allineamento a NIST SSDF/ISO 27001; allegare Model Card se AI-assisted.

**Frequenza & Governance**  
- Schedulazioni: giornaliera/settimanale/mensile + on-demand.  
- Versionamento schema con migrazioni backward-compatible; rollback documentato.

**Metriche**  
- `audit_exports_generated_total{schedule}`  
- `audit_exports_verify_fail_total{reason}`

**Vincoli**  
- Nessuna chiave sensibile; PII mascherate.  
- Firma e verifica richieste in CI/CD prima della distribuzione.
