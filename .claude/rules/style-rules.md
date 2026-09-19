---
paths:
  - src/**/*.jl
  - test/**/*.jl
  - examples/**/*.jl
---

# Style Rules

These rules are always in effect for the codebase.

## Variable naming

- Prefer descriptive names that read naturally in English: `element_connectivity`, `reference_element`, `boundary_values`, `residual_norm`.
- Use qualifier-first naming for compound concepts: `nodal_velocity`, `temperature_field`, `stiffness_matrix`.
- Keep Julia idioms: functions and variables are lowercase; `!` marks mutation; types are CamelCase.
- Do not use cryptic truncations or ambiguous abbreviations when a clear, readable name is available.
- Reserve leading underscores for internal kernel helpers only when the naming convention is already established.

## Comments

- Prefer self-explanatory code over dense comments.
- Add a comment only when it explains a non-obvious invariant, index convention, numerical choice, or a required workaround.
- Keep comments surgical: one line, placed exactly at the confusing step.
- Do not restate what the next line obviously does.
- Avoid TODO-style commentary unless explicitly requested.

## Code structure

- Keep a single responsibility per function.
- Push shared mechanics into a common helper and keep physics-specific behavior in the relevant module.
- Avoid caller-by-caller guards when the root cause is shared.
- Preserve numerical layout invariants: connectivity remains 1-based, static arrays remain static, and backend-aware allocations use the existing package conventions.

## Julia-specific conventions

- Prefer stable, explicit types when the algorithm depends on them; avoid silent promotion in hot paths.
- Use `@views` only when it improves clarity without hiding indexing hazards.
- Keep kernel code backend-neutral and use `KernelAbstractions.get_backend` / existing backend-aware helpers instead of hard-coded GPU branches.
- Preserve element ordering conventions and do not modify node ordering in a way that would silently change the discretization.

## Review standard

If a change can be explained in a sentence without a comment, it probably does not need one. If a change depends on a subtle convention, document the convention and not the obvious code.
