---
name: docstrings
description: Generate and update Julia docstrings, inline comments, and module-level documentation for FEMTools.jl source files. Invoke with /docstrings.
tools: Bash, Glob, Grep, Read, Edit
---

# Docstrings

Add or update docstrings, inline comments, and module documentation across
`src/`. Apply only to the files named in the invocation arguments, or — when
called with no arguments — to every file under `src/`.

## 1. Identify targets

If arguments were given (e.g. `/docstrings src/mesh/mesh.jl`), operate only on
those files. Otherwise:

```bash
find src/ -name "*.jl" | sort
```

Read each target file in full before writing anything.

## 2. Docstring style

All public docstrings must follow the project convention exactly:

```julia
"""
    signature(arg1, arg2) -> ReturnType

One-sentence summary.

Longer description if needed: invariants, preconditions, non-obvious behaviour.
Do NOT describe what is already obvious from the name or type signature.
"""
```

Rules:
- First line: indented signature matching the method exactly (no leading
  spaces beyond the four-space indent).
- Blank line after the signature, then the summary sentence.
- Second blank line before any extended description, argument table, or
  examples — only when genuinely needed.
- Use a `# Arguments` / `# Returns` section only when the signature alone is
  not self-documenting (multiple positional args of the same type, optional
  keyword args with non-obvious defaults).
- Type parameters are explained in terms of their role, not their name
  (`nDim` → "reference-space dimension", `N` → "number of local nodes", etc.).
- Ascii art node diagrams (as seen in `elements.jl`) are preserved verbatim.
- Never add `@param`, `@return`, or any Javadoc-style tags.

## 3. What to document

### Must have a docstring
- Every `abstract type` and `struct` exported from `src/FEMTools.jl`.
- Every exported function or method.
- Constructors with non-trivial semantics (e.g. `Mesh(backend, Ω, element, nels)`).

### Should have a docstring
- Non-exported types that appear in public method signatures.
- Non-trivial internal helpers (>10 lines or non-obvious algorithm).

### Must NOT get a docstring
- One-liner internal accessors that are already obvious from the name.
- `include(...)` calls.
- `const` aliases that are self-explanatory.

## 4. Inline comments

- Delete comments that merely restate the code (`# return x` above `return x`).
- Preserve or add comments only for non-obvious invariants, workarounds, or
  algorithm steps that would surprise a reader.
- Follow the existing style: comments on the same line for struct fields
  (e.g. `Ω::D   # model domain`).
- Use sentence case, no trailing period for short inline comments.

## 5. Module-level docstring (`src/FEMTools.jl`)

The `@doc """ ... """ FEMTools` block at the top of `src/FEMTools.jl` must:
- Describe the overall purpose of the package in ≤ 4 sentences.
- List the main public types and the mental model for using them
  (reference-element tag → `ReferenceElement` → `Mesh`).
- Mention GPU backend dispatch via `TA(backend)`.
- Not duplicate the per-function docstrings.

Update it only when the package structure changed since the last edit; otherwise
leave it unchanged.

## 6. Applying changes

- Edit files in place with the `Edit` tool. Do not rewrite entire files unless
  more than half of the content changes.
- One `Edit` call per docstring or comment block — do not batch unrelated edits
  into a single call.
- Do not alter any logic, signatures, or formatting outside the docstring /
  comment regions.
- After all edits, run:

```bash
julia --project=. -e 'using FEMTools'
```

to confirm the module still loads without error. Report any load error and stop.

## 7. Output

For each file edited, list:
- File path
- Names added / updated (one line each, prefixed `+` for new, `~` for updated)

If a file already has complete, accurate docstrings, write `✓ <path> — no
changes needed` and move on.
