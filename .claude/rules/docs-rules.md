---
paths:
  - docs/**/*
---

# Documentation Rules

## Building docs

```sh
julia --project=docs docs/make.jl
```

- Keep docs in sync with exported/public API changes.
- Update relevant pages when a public function, type, or workflow changes.
- Prefer the existing `docs/src/*.md` structure over creating new document pages unless there is a true missing topic.

## Writing docs

- Explain the package and the numerical model with concise, actual usage examples.
- Use the existing Documenter patterns and cross-reference symbols carefully.
- Do not document what is already obvious from a function signature or the package naming convention.
- For user-visible behavior changes, update the relevant API docs and examples in the same patch.

## Relevance

- If a new exported symbol is added to `src/FEMTools.jl`, make sure it is discoverable from the public docs.
- If a change alters solver behavior, convergence expectations, or boundary-condition semantics, update the affected docs and examples together.
- Do not leave stale examples or references behind when the API evolves.
