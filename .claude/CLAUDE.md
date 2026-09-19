# FEMTools.jl Claude Rules

This repository uses lightweight project-local rules and skills to keep agentic work consistent with the package's numerical, backend, and test conventions.

## Auto-loaded rules

These files are picked up automatically when working in matching paths:

- `.claude/rules/style-rules.md` — formatting and naming conventions across source, tests, and examples
- `.claude/rules/testing-rules.md` — test selection and regression-test expectations
- `.claude/rules/docs-rules.md` — documentation build and API accuracy rules
- `.claude/rules/docstring-rules.md` — docstring conventions for exported APIs and public state
- `.claude/rules/kernel-rules.md` — backend-neutral kernel and data-layout constraints

## Core operating principles

- Read the relevant source, tests, and docs before editing.
- Prefer a shared fix over caller-by-caller special cases.
- Preserve mesh and connectivity invariants: 1-based connectivity, static local layouts, and consistent element order.
- Keep compute code backend-neutral and use the package’s existing KernelAbstractions patterns.
- Keep public API changes deliberate, documented, and covered by tests.
- Run the tightest relevant verification first, then widen scope if needed.

## Repository-specific workflow

- Start from `AGENTS.md` and the relevant subsystem guide before modifying code.
- Keep numerical behavior, backend compatibility, and minimality in mind for every change.
- When touching docs or public APIs, update user-facing documentation in the same patch.
- Favor small, targeted regression checks over broad suite runs during iteration.

## Skills available

- `/docstrings` — update docstrings and comments for package source or docs
- `/minimise` — shrink a fix or test to the minimal correct diff
- `/scrutinise` — review a branch or patch against the repo's invariants
- `/verify-example` — run a headless example to validate convergence and solver behavior
- `/gpu-audit` — diagnose kernel memory, register pressure, or backend-specific issues

## Standard verification commands

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=docs docs/make.jl
```

For targeted validation, run the most relevant file or solver-level check rather than the entire suite blindly.
