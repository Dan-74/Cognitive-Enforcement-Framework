# Cognitive Enforcement Framework  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17476141.svg)](https://doi.org/10.5281/zenodo.17476141)


**Public release version** of the *Cognitive Enforcement Framework* project, developed for cognitive auditing and control of automated systems based on agent-driven toolchains.  
This repository includes only documentation, policies, semantic rules, and reference API specifications.  
Executable code remains proprietary.

---

## Purpose

*Cognitive Enforcement Framework * is a conceptual and architectural framework designed to:

- Ensure **cognitive and decision integrity** in distributed agent systems;
- Prevent **semantic and structural bias** in automated decision-making;
- Provide **auditability and traceability** compliant with **UN**, **NIST**, and **ISO 27001** standards;
- Integrate **CQRS**, **Clean Architecture**, **Event-Driven Design**, and **Design by Contract** principles into automated governance frameworks.

---

## Contents of this release

| File | Description |
|------|--------------|
| `AGENTS.md` | Conceptual model and taxonomy of agents |
| `ToolAgent - 01_Policy_and_Meta.md` | Institutional policies and cognitive meta-rules |
| `ToolAgent - 02_Components.md` | Modular structure and functional relationships |
| `ToolAgent - 03_Enforcement.md` | Enforcement and cognitive governance procedures |
| `ToolAgent - 04_Semantic_Rules_and_XRef.md` | Semantic rules, cross-references, and logical constraints |
| `ToolAgent - 05_Appendices_and_Versioning.md` | Appendices, versioning, and document consistency |
| `ToolAgent - 06_Cognitive_Bias_Prevention.yaml` | Mechanisms for cognitive bias prevention |
| `tool_protocol.py` | PEP 544 contract for asynchronous Tools and execution schema |
| `tool_types.py` | Structured types for audit trails and persistent agent state |
| `config.toml` | Standardized configuration parameters |
| `README.md` | Main documentation file (this document) |
| `LICENSE` | Document license (CC BY-NC-ND 4.0) |
| `CITATION.cff` | Metadata and DOI for academic citation |

---

## Conceptual Architecture

**Core principles:**
- *Observe → Plan → Act* as the primary cognitive cycle;  
- Integrated *Bias-Control Layer* (semantic reframing and contrarian review);  
- *Immutable Audit Trail* and observability metrics (Prometheus/Datadog compliant);  
- *ToolContract Protocol* compliant with [PEP 544](https://peps.python.org/pep-0544/);  
- Strong typing via `TypedDict`, `Enum`, and serializable JSON aliases;  
- Built-in persistence and recovery through `SerializedAgentState`.

**Example – public interface:**
```python
class ToolProtocol(Protocol):
    name: str
    description: str

    async def run(self, *, input: ToolInput) -> ToolResult: ...

## Citation

If you use or reference this work, please cite:

Luzzo, D. (2025). Cognitive Enforcement Framework (v1.0.0). Zenodo.  
DOI: 10.5281/zenodo.17476141

License

The contents of this repository are distributed under the
Creative Commons BY-NC-ND 4.0 International License.
Commercial use, modification, or redistribution without explicit permission is prohibited.

Operational Notes

This is a documentation-only release intended for research, review, and citation.

The operational code (tool_agent.py and dependent modules) remains proprietary.

All specifications conform to UN AI Ethics, NIST AI RMF 1.0, and ISO/IEC 42001 guidelines.

For institutional collaborations or licensing requests:
daniele.luzzo@gmail.com

DOI and Version


Version: 1.0.0 — Date: 2025-10-29


