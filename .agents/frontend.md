# Frontend

## Current reality

FEMTools.jl has no package-level graphical, web, notebook, or command-line
frontend. The supported library interface is Julia code.

Some scripts in `examples/` use GLMakie for plotting and expose keyword-driven
`main` functions such as `show_plot`, `render`, and output paths. Plotting is
host-side and belongs to the examples environment. Headless modes are already
important for CI, benchmarks, and GPU runs. The core `Project.toml` does not
depend on Makie, while `examples/Project.toml` does.

Treat examples as the current thin frontend: they assemble a problem, call the
library, optionally write output, optionally visualize it, and return useful
arrays/statistics for automation.

## Boundaries

- `src/` owns numerical models, data validation, solving, and reusable
  post-processing. It must not import GLMakie or require a display server.
- `examples/` may own visualization, interactive defaults, and problem-specific
  orchestration.
- Frontends call public package APIs. They should not force numerical behavior
  into display callbacks or duplicate residual/mesh logic.
- Every interactive path must have a headless equivalent that can be tested and
  benchmarked.
- Long solves must expose progress and cancellation at a natural solver
  boundary before a UI claims to support them; do not simulate cancellation by
  abandoning mutable state mid-kernel.
- User input is a trust boundary. Validate dimensions, element choices,
  material tuple lengths, file paths, and solver controls before launching a
  costly solve.

## Direction

Do not choose a frontend stack until a concrete user workflow is selected. The
minimum useful progression is:

1. Keep example `main` functions callable, headless, and returning structured
   results.
2. Factor only genuinely repeated problem setup or visualization code.
3. Define a small serializable run specification only when two real frontends
   or batch workflows need the same input contract.
4. Add a CLI, notebook helper, desktop UI, or web UI only for a named workflow
   and audience.

Potential workflows to decide explicitly before implementation:

- teaching/demo exploration;
- reproducible batch configuration;
- mesh and boundary-condition inspection;
- monitoring long CPU/GPU/MPI runs;
- post-processing existing result files.

Keep heavy UI dependencies in an example, separate environment, or optional
extension. A frontend must not make `using FEMTools` initialize graphics.

## Acceptance checks

For frontend/example orchestration changes:

- the main numerical path runs with plotting disabled;
- invalid inputs fail before allocation or solver launch with a useful message;
- returned results contain enough data for tests without scraping logs;
- output paths are explicit and generated files stay out of source control;
- plotting does not scalar-index live GPU arrays—transfer intended result data
  to the host once;
- defaults produce a tractable example, while scientific parameters remain
  explicit in docs;
- any actual UI preserves keyboard access, labels controls, and reports errors
  without losing the user's configuration.

Examples are currently parse-checked, not comprehensively executed in CI. Add a
small headless execution check when an example becomes a maintained interface.

## Open decisions

- Which frontend workflow is the first supported product rather than an
  example?
- Is run configuration intended to be Julia code, a data file, or both?
- Which outputs must be inspectable live versus after the solve?
- Where should long-running jobs execute: local process, MPI allocation, or an
  external service?

Do not resolve these by creating a generic framework in advance.

## Update this guide when

- an example becomes a supported user entry point;
- a CLI, notebook, GUI, or web workflow is selected or implemented;
- core/frontend dependency boundaries change;
- configuration, progress, cancellation, or result-display contracts become
  concrete;
- headless or accelerator behavior changes.
