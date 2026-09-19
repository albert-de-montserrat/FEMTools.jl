# Skills

This repository keeps automation lightweight and narrowly scoped to the work that most often needs agent assistance.

- `docstrings` — update docstrings, inline comments, and public docs for source or docs files
- `minimise` — shrink a patch to the smallest correct diff and verify it still passes
- `scrutinise` — review a change against the repo’s architecture and invariants
- `verify-example` — run a headless example to check convergence and numerical correctness
- `gpu-audit` — investigate kernel memory footprint or backend compatibility issues

Each skill is intentionally precise and repository-specific. Use the slash command matching the task at hand, and prefer the narrowest relevant check before widening to broader validation.
