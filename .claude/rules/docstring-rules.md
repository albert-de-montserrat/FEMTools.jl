---
paths:
  - src/**/*.jl
---

# Docstring Rules

## Style

Follow the package convention exactly:

```julia
"""
    signature(arg1, arg2) -> ReturnType

One-sentence summary.

Longer description if needed: invariants, preconditions, non-obvious behaviour.
Do NOT describe what is already obvious from the name or type signature.
"""
```

Rules:

- The first line is the exact method signature with four spaces indentation.
- Leave a blank line after the signature, then a one-sentence summary.
- Add a second blank line before any extended description only when it is genuinely useful.
- Use `# Arguments` and `# Returns` sections only when the signature is not self-documenting.
- Keep docstrings about invariants, not restatements of code.

## Scope

- Every exported type and exported function should have a docstring.
- Constructors with non-trivial semantics should describe the intended setup and assumptions.
- Public methods with non-obvious numerical behavior need a short invariant explanation.
- Avoid adding docstrings to obvious accessors or trivial wrappers.

## Examples

- Do not add `@param` or `@return` tags.
- Preserve existing numerical layout or node-ordering comments verbatim when they are part of the source explanation.
- Keep module-level documentation at the top of `src/FEMTools.jl` brief and high-value.
