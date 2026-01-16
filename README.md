# Reconsidered RAG

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io/)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/rkttu)](https://github.com/sponsors/rkttu)

English | **[한국어](README.ko.md)**

[![Watch the demo](https://img.youtube.com/vi/Uj6Vz5CZ4c4/maxresdefault.jpg)](https://youtu.be/Uj6Vz5CZ4c4)

Reconsidered RAG is a reference project that rethinks how Retrieval-Augmented Generation
should be applied in environments where responsibility, data sovereignty,
and failure boundaries matter.

This project was formerly known as **AIPACK**.

---

## Why this project exists

Reconsidered RAG exists to question an assumption that has quietly become common:
that retrieval-augmented generation should always aim to answer more, faster, and better.

In regulated, sensitive, or responsibility-heavy environments,
those goals often conflict with reality.

This project focuses on:

- where retrieval boundaries should be drawn,
- when a system should refuse to answer,
- and how responsibility remains with the operator, not the model or the service.

The code in this repository serves as a **reference implementation**
to support these discussions, not as a blueprint to be copied verbatim.

---

## What this project is NOT

Reconsidered RAG is intentionally **not** positioned as the following:

- **Not a product or service**  
  This project is not a SaaS, platform, or deployable solution for general users.
  It does not aim to provide turnkey deployments or production guarantees.

- **Not a framework or toolkit**  
  Reconsidered RAG does not attempt to standardize how RAG *should* be built.
  It documents how RAG *can fail*, and where boundaries must be drawn.

- **Not a performance benchmark**  
  Model quality, retrieval speed, or accuracy scores are not the primary focus.
  Faster answers are meaningless if responsibility and control are unclear.

- **Not a generic AI abstraction layer**  
  Complexity is not hidden behind convenience.
  Explicit constraints and visible trade-offs are considered part of the design.

- **Not a promise of safety**  
  This project does not claim that AI systems can be made fully safe.
  It exists to make risks explicit, discussable, and owned by the operator.

---

## Design Principles

Reconsidered RAG is guided by the following principles:

- **Explicit boundaries over implicit behavior**  
  Every retrieval scope, data flow, and integration point should be visible
  and explainable. Hidden defaults are treated as a design failure.

- **Responsibility is owned, not abstracted away**  
  Decisions made by the system must have a clear owner.
  External services do not absolve operators from accountability.

- **Refusal is a valid and expected outcome**  
  The system is allowed to say "no" when inputs exceed defined boundaries.
  Answering every question is not a goal.

- **Constraints are part of the feature set**  
  Limitations, trade-offs, and non-goals are documented explicitly.
  Convenience is secondary to clarity.

- **Failure modes matter more than success cases**  
  This project prioritizes understanding how and where RAG systems fail,
  especially under ambiguous or adversarial conditions.

---

## Failure modes we explicitly care about

Reconsidered RAG pays particular attention to failure modes such as:

- silent expansion of retrieval scope,
- accidental inclusion of sensitive or unintended documents,
- overconfident answers derived from weakly grounded context,
- lack of an explicit refusal path,
- and unclear ownership when things go wrong.

These are not edge cases.
They are expected outcomes when RAG systems are deployed without explicit boundaries.

---

## Reference implementation (non-goals first)

The implementation in this repository intentionally favors:

- transparent intermediate artifacts over opaque pipelines,
- database portability over aggressive optimization,
- inspection and reasoning over automation.

These choices are not meant to define "the right way" to build RAG systems.
They exist to make trade-offs and failure modes visible.

Concretely, this means:

- intermediate data stored in inspectable formats (such as parquet),
- an MCP server used for controlled inspection, not production serving,
- and deliberate avoidance of early commitment to specific databases or vendors.

---

## Implementation details

For detailed implementation documentation including:

- Architecture and data flow diagrams
- Installation and setup instructions
- Module descriptions and usage
- Configuration options
- Containerization with Docker
- IDE integration (VS Code, Cursor)
- Output schema and directory structure

See **[IMPLEMENTATION.md](IMPLEMENTATION.md)**.

---

## Scope and intended audience

Reconsidered RAG is written for:

- engineers working in regulated or sensitive environments,
- architects responsible for AI adoption decisions,
- and practitioners who have experienced RAG failures firsthand.

It is not optimized for beginners,
nor for teams seeking quick production wins.

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Sponsorship

If you find this project helpful and would like to support its continued development,
please consider sponsoring on GitHub Sponsors.
Your support helps maintain and improve this open-source project.

[![GitHub Sponsors](https://img.shields.io/github/sponsors/rkttu)](https://github.com/sponsors/rkttu)

## Contributing

1. Fork this repository.
2. Create a new branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Create a Pull Request.

---

## Final note

Reconsidered RAG exists to slow decisions down,
not to accelerate deployments.

If this repository makes you pause before adding "just one more document"
to your retrieval pipeline, it has done its job.
