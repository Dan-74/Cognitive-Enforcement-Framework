<!-- ToolAgent Specification Block. No placeholder permitted. -->
<!-- hash: sha256-7439d27142c3b3dc6e20fcd07abc908a5279c343d0c35d1048a715d6c1e99514 -->
## 1. Policy Computabile
> Root Path Reference: `./`
> Config Reference: `./ToolAgent - 06_Cognitive_Bias_Prevention.yaml`


### 1."1" Qualifiche
- Normative: NIST SP "800"-218, NASA SWE-139, OWASP SAMM, ISO/IEC 27001.
- Compiti: auditing CI/CD (xref:6."3") e enforcement statico (xref:6."1").

### 1."2" Valori
- Auditabilita: ledger immutabile e SHA256 (xref:3."8").
- Naming e struttura: vedi xref:4 e `docs/semantic_check/ci_quality_gate.md`.
- CI blocca placeholder o duplicazioni (xref:6."3").

### 1."3" Finalita
- Funzione: orchestrare comandi su `domain/analysis` per coerenza misurata via `tool_call_success_total` (xref:3."9").
- Funzione: registrare snapshot e hash su ledger per integrita verificabile (xref:3."8").
- Funzione: garantire coverage test >=95% con blocco CI su fallimento (xref:2."6").




## 2. Mandato di Produzione e Infrastruttura

Architettura CQRS, Hexagonal e Plugin-based. Osservabilita completa (logging, tracing, metrics). Qualita garantita da CI/CD con coverage >=95% e SAST/DAST obbligatori. Tutte le risorse condivise provengono dal `DIContainer`.

### 2."1" Framework e Standard Normativi Internazionali

Aderenza a NIST SSDF, ISO/IEC 27001/27034, OWASP ASVS, PCI-DSS e GDPR.
**Cross-Reference:**  
Questo documento eredita e implementa le regole definite in `./ToolAgent - 06_Cognitive_Bias_Prevention.yaml`, garantendo l'applicazione delle policy di decision hygiene e debiasing cognitivo a livello di governance e validazione.


### 2."2" Logging e Tracing

Logging strutturato e tracing OpenTelemetry. Log segregati e PII mascherate. Per l'enforcement dei log vedi xref:6.3.

### 2."3" Metriche e Monitoraggio

Golden Signals obbligatori con dashboard Prometheus/Grafana. Alerting automatico su eventi P1/P2.

### 2."4" Validazione Input e Schema Enforcement

Validazione su tutti i confini con pydantic e TypedDict. Fallback fail-fast con log.

### 2."5" Fail-Fast & Graceful Degradation

```yaml
rollback: on_failure
retry: exponential_backoff
logging: full_event
```

### 2."6" CI/CD & Quality Gate

Test >=95%, `mypy --strict`, `bandit`, `pylint` e audit log CI. Nessun placeholder ammesso.

### 2."7" Analisi Statica e Modulare

Codice senza side effects su import e complessita <10. Tutte le regole di enforcement sono definite in xref:6.
