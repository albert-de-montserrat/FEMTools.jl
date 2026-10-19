# MPI implementation plan

Status: phase 0 transport experiment and phase 1 ownership metadata are
implemented; phases 2-4 exist only as an example-level prototype. Scope:
distributed-memory domain decomposition with CPU and CUDA support from the
first complete milestone, preserving the existing matrix-free finite-element
mathematics and serial API. External package count is unrestricted; CUDA
compatibility is a mandatory dependency and design gate.

The `benchmarks/stokes/solcx2D_MPI/` scripts still call the serial package API;
their directory name does not provide a distributed implementation.

## Current state

Package code (no MPI dependency):

- `src/distributed/partition.jl`: `MeshPartition`, `partition_mesh`,
  `localize_dirichlet`, `reconstruct_owned`, and the `partition_elements` stub.
  All four names are internal; the public surface is decided in phase 4.
  Parts are labelled `1:nparts`; the MPI layer maps rank `r` to part `r + 1`.
- `ext/FEMToolsMetisExt.jl`: `partition_elements` via the face-adjacency graph.
  Metis is a weak dependency and a test dependency.
- `src/dynamic_relaxation/solver.jl`: the `execution` keyword and `_dr_*`
  hooks; `nothing` keeps the serial path. `assemble_dr_matrices_atomix!`
  handles zero local elements.
- `test/test_partition.jl`: phase-1 gate. `test/test_scalar_initial_bc.jl`:
  initial-BC regression.

Example-level prototype of phases 2-4 (`examples/` environment):

- `examples/miniapps/thermal/2D_heat_diffusion_MPI/mpi_thermal.jl`
  (`MPIThermal`): root partitions with the package helpers, serializes and
  scatters parts, builds local T3 meshes with inherited physical boundaries, and
  runs halo copy/add through PartitionedArrays `consistent!`/`assemble!` with
  KA pack/unpack and host staging. It also supplies the private communicator,
  collective error agreement, owner-range DR hooks, and owned gather.
- `.../2D_heat_diffusion_MPI.jl`: headless CPU/CUDA driver.
- `test/mpi/runtests.jl` + `thermal_worker.jl`: launcher with timeout and the
  worker fixtures (independent affine T3 oracle for `R`/`PC`/row sums, ghost-only
  rank, empty ranks, reversed split communicator, repeated assembly, two
  physical steps, budget exhaustion, injected NaN failure).

The prototype is not yet the plan's extension layout (`ext/FEMToolsMPIExt.jl`,
`FEMTools.distribute_mesh`, `scatter_field!`, `gather_owned`); phase 2 moves it.

### Gate records

All runs on 2026-10-19, working tree on `adm/mpi` after `9957536`, Linux,
24 CPU cores, one NVIDIA GeForce RTX 3080. Examples environment: Julia 1.13.1,
MPI.jl 0.20.29 with MPICH 5.0.2 (`MPICH_jll`), PartitionedArrays 0.5.14,
Metis 1.5.0, CUDA.jl 6.4.1. Package tests: Julia 1.12.7.

| Gate | Command | Result |
|---|---|---|
| Phase 1 metadata, Float32/Float64, T3 and Q9, shuffled numbering, zero-owned-node and empty parts, Metis balance | `TestEnv` include of `test/test_partition.jl` | pass (493 tests) |
| CPU worker, 1/2/4 ranks, Float32/Float64 | `julia --project=examples test/mpi/runtests.jl` | pass (119/81/77 tests per rank) |
| CUDA worker, 1 rank, `allowscalar(false)` | `mpiexec -n 1 julia --project=examples test/mpi/thermal_worker.jl --cuda` | pass (119 tests) |
| CPU driver, 2 ranks, 16×16, 2 steps | `FEMTOOLS_MPI_N=16 FEMTOOLS_MPI_STEPS=2 mpiexec -n 2 julia --project=examples .../2D_heat_diffusion_MPI.jl` | pass, range (273, 873) |
| CUDA, 2 ranks on distinct GPUs | — | unrun: one GPU available |
| Transfer volume, allocations, cached-buffer reuse evidence | — | unrun |
| Setup-failure fixtures (control mismatch, root Metis/input failure, injected allocation failure) | — | unrun |

The phase 0 transport choice (PartitionedArrays with host staging) is
provisional until the two-GPU gate passes.

## First deliverable

Implement one backward-Euler thermal step on a partitioned 2-D T3 mesh using
one, two, and four CPU ranks, plus one CUDA rank and two ranks on distinct GPUs.
Reconstruct the temperature by global node ID and compare with the serial CPU
and CUDA solves and a tiny independently assembled reference. Extend CUDA runs
to four distinct GPUs when the target hardware is available.
Residuals, preconditioners, spectral estimates, convergence decisions, and
boundary values must agree within justified floating-point tolerances.

The first release supports Metis-generated or caller-supplied element assignments,
root-side setup, fixed topology, and explicit MPI initialization in the application.
It includes portable host-staged CUDA halo transport, empty-rank handling,
collective-safe numerical failures, a headless CPU/CUDA example, CPU multi-rank
CI, and a required GPU hardware gate. CUDA-aware MPI is an optional transport
optimization. Scalability claims require subsequent measurements.

The first supported combination is deliberately specific: fixed conforming T3
topology, scalar thermal diffusion, Float32/Float64, and either CPU on every
rank or CUDA on every rank. Reject mixed CPU/CUDA jobs, adaptive/repartitioned
meshes, unsupported element families, and concurrent use of one context.
Metis and explicit assignments both belong to this milestone. A general
partition-metadata test on a high-order mesh does not enable that solver family.
Root initially supplies a host mesh; upload only the resulting local meshes.

Keep three kinds of decision separate throughout implementation:

| Fixed design | Requires an experiment | Deferred capability |
|---|---|---|
| Unique element/node ownership, ordinary local KA arrays, owner-only updates/reductions, global solver decisions, host-staged CUDA, optional MPI loading | PartitionedArrays transport fit, compatible dependency versions, communicator cleanup, actual GPU buffer/task behavior | CUDA-aware MPI, overlap, moving/repartitioned meshes, distributed input, Stokes, adjoints, parallel persistence |

## Package reuse strategy

External package count is unrestricted. Prefer maintained packages for generic
partitioning, communication, and file formats; FEMTools still owns its mesh
mapping, physical boundary semantics, element kernels, and numerical iteration.
Optional loading below is an installation/design choice, not a dependency budget.

| Package | Planned role | Adoption point |
|---|---|---|
| [MPI.jl](https://juliaparallel.org/MPI.jl/stable/) | Communicators, global reductions, setup transport, launcher, and explicit abort policy. | Required by the MPI feature. |
| [CUDA.jl](https://cuda.juliagpu.org/stable/) / existing KernelAbstractions | Device storage, kernels, local reductions, bulk transfers, and synchronization. | Required for the CUDA feature; preserve optional loading for CPU users. |
| [PartitionedArrays.jl](https://partitionedarrays.github.io/PartitionedArrays.jl/stable/reference/pvector/) | Preferred owner/ghost vector exchange and distributed index infrastructure. | Evaluate before writing transport; adopt if the integration gate passes. |
| [Metis.jl](https://github.com/JuliaSparse/Metis.jl) | Automatic root-side element graph partitioning. | Include in the first milestone; keep explicit assignments for fixtures and external partitions. |
| [WriteVTK.jl](https://juliavtk.github.io/WriteVTK.jl/stable/metadata/parallel/) | Rank-local XML VTK pieces and a parallel header. | Parallel visualization increment; already present in the examples environment. |
| [MPIPreferences.jl](https://juliaparallel.org/MPI.jl/stable/configuration/) | Select an application/cluster MPI implementation. | Deployment/test environment configuration. |

### Resolve the communication choice first

Run a small PartitionedArrays integration experiment in phase 0. Its `PVector`
supports local/owned storage access; `consistent!` refreshes ghosts and
`assemble!` adds ghost contributions to owners, clears those ghosts, and returns
a task. Await completion before kernels consume results. Construct cached
adapters once, keeping ordinary local solver arrays and avoiding full-field
copies on every iteration. The
[PVector reference](https://partitionedarrays.github.io/PartitionedArrays.jl/stable/reference/pvector/)
documents these operations; compatibility with our buffer layout needs execution.

Prove arbitrary unstructured ownership, external-ID-to-dense-index mapping,
owned-first local ordering, Float32/Float64 preservation,
empty ranks, communicator isolation/lifetime, cache reuse, and allocation cost.
Run the same copy/add fixtures with CuArray-backed local state and scalar
indexing disabled. Generic AbstractArray signatures do not establish CUDA
support: verify device placement, safe asynchronous task/device
handling, and transfer volumes. If the library exchanges host vectors, prove a
public adapter can stage only packed halos while the solver fields stay on GPU.
Two adapter shapes are acceptable: borrow solver buffers without copying, or
pack into cached communication vectors containing only shared-node entries.
The latter must map every shared global ID to its unique owner and all sharers;
it must not create a host mirror of every local field. Requiring a CuArray-backed
PVector itself would unnecessarily exclude a valid compact host-staging adapter.
Translate library part IDs to/from zero-based MPI ranks explicitly. Build local
mesh connectivity from the actual library index map; never assume it retained
our ordering. Compare copy/add results against exact hand-calculated fixtures.
Use its DebugArray backend for additional serial debugging after adoption.

PartitionedArrays setup can initialize MPI and can duplicate or take ownership
of a communicator. Require prior application initialization. Its documented
`duplicate_comm=false` transfers ownership: never pass the application's
communicator that way. Establish ownership and public cleanup of all library
copies, including repeated construction/destruction, in the experiment. Share
one verified communication backend across cached field adapters rather than
duplicating communicators on every assembly. Use `with_mpi` only at the application's
fatal-error boundary: its abort behavior is different from a solver's ordinary
non-convergence return. See the
[backend contract](https://partitionedarrays.github.io/PartitionedArrays.jl/stable/reference/backends/).

If a required behavior cannot be supported through public APIs, record the
specific blocker and use the direct MPI.jl transport below. Select one transport
implementation supporting CPU and CUDA host staging for the first milestone;
do not maintain competing distributed-vector paths without a demonstrated need.
The device adapter is part of phase 0's acceptance gate. CPU-only success is
insufficient to select PartitionedArrays.

## Existing code to reuse

| Area | Implementation and implication |
|---|---|
| Meshes | `src/mesh/mesh.jl` and `mixed_mesh.jl`: process-local arrays and counts; no ownership or global IDs. Preserve these local containers inside a distributed wrapper. |
| Scalar assembly | `src/dynamic_relaxation/assembly.jl`: shared atomic/colored scatter of residual, absolute element Jacobian row sums, and diagonals. Reuse these kernels over owned elements. |
| Scalar iteration | `src/dynamic_relaxation/solver.jl`: shared thermal/lithostatic DR loop. Add ownership and communication boundaries here instead of duplicating the iteration in an extension. |
| Physics adapters | `src/heat_diffusion/solvers/DR.jl` and `src/lithostatic_pressure/solvers/DR.jl`: prepare element assembly closures. Retain their element residuals and Jacobians. |
| Scalar reductions | `norm(R)`, `_checked_λmax` in `src/utils.jl`, and Rayleigh numerator/denominator in the DR loop are local today. Each needs an owned-entry reduction followed by a communicator reduction. |
| Stokes | `src/stokes/solvers/DR.jl` has mixed Powell-Hestenes/DYREL and a separate Hex27 cell-pressure loop. Preserve their distinct iteration schemes. |
| Pressure preparation | `src/stokes/assemblers/pressure_scaling.jl` accumulates mass and a scaling numerator, then divides. Shared pressure requires communication before division. |
| Output | `src/postprocess/postprocess.jl` writes serial legacy VTK. Gather by global ID for small validation runs only. |
| Optional dependencies | Existing CUDA and Triangulate extensions demonstrate the package pattern. Put MPI transport in a weak-dependency extension. |
| Tests | `test/runtests.jl` discovers top-level `test_*.jl`; `.github/workflows/CI.yml` currently has no MPI job. Keep MPI workers in a separate subdirectory. |

Three initialization details need explicit treatment. Unstructured `Mesh`
construction infers boundaries from locally exposed edges/faces, and `MixedMesh`
construction computes normals similarly. Those routines must not infer physical
boundaries from a partition. Also, scalar DR currently repins solution BCs after
an update, rather than before its first residual; reconcile initialization in
the shared path with a focused regression before claiming one-rank equivalence.
Finally, thermal `integrate_residual` currently uses consistent quadrature mass
and source terms. The independent oracle must follow that discrete formulation;
a lumped transient term would test a different problem.

## Proposed architecture

### Partition data and ownership

Add MPI-independent host metadata in `src/distributed/partition.jl`. Initially
test it without a launcher or MPI installed.

- Assign stable global node and element IDs before reordering. Initial defaults
  are original serial indices; preserve supplied IDs when available. Use
  `Int64` global IDs and retain an appropriate checked local connectivity type.
- Accept one zero-based communicator rank per global element. Validate length,
  rank bounds, uniqueness of IDs, connectivity, and incident-node coverage.
  Generate this assignment with Metis when not supplied. Reuse
  `generate_element_adjacency(...; shared=:face, dimension=...)` and convert it
  to a symmetric sparse graph for `Metis.partition`. That graph is setup data,
  not an assembled finite-element operator. The current wrapper returns
  one-based part labels; normalize and validate them before use. Special-case
  one rank and fewer elements than ranks, and accept empty parts. The
  [Metis implementation](https://raw.githubusercontent.com/JuliaSparse/Metis.jl/master/src/Metis.jl)
  accepts sparse graphs, so Graphs.jl is optional rather than a prerequisite.
- Every element belongs to exactly one rank and is assembled there exactly once.
  Store only owned elements and the nodes needed by those elements; no ghost
  elements are needed for this first element-local assembly path.
- A shared continuous node is owned by the lowest rank among its incident
  elements. Store owned nodes first, then ghosts, each sorted by global ID.
  This gives a contiguous owned range for existing update kernels.
- A rank can own elements while owning no nodes: all of its nodes can be ghosts.
  It still assembles those elements and sends their contributions. Only its
  owned-node updates/reductions are empty. Never use `nowned == 0` to skip assembly.
- Store local/global maps, owner ranks, global node/element counts, neighbor
  lists, and exchange metadata. Reuse library maps when adopted; keep stable
  external IDs separate from any dense library index space. A peer may share
  nodes through a vertex
  alone: face adjacency is insufficient for nodal communication.
- Preserve original physical boundary IDs and BC values. Translate them to
  local indices, with owner BCs defining authoritative values; ghost values
  arrive from the owner. Reject inconsistent duplicate constraints.
- Treat orphan global nodes as invalid input rather than dropping them during
  partitioning. Allow a rank with zero nodes/elements; reject an empty global
  problem explicitly.
- Local `mesh.nnodes` includes owned and ghost nodes; `mesh.nels` counts owned
  elements. Global counts belong in partition metadata and never substitute for
  local allocation or launch sizes.

Treat serial array positions, stable external IDs, library indices, and MPI
ranks as separate spaces. Connectivity still indexes local arrays from one;
arbitrary Int64 external IDs do not become array subscripts. Store an explicit
element-column permutation and per-field node maps. Preserve local element node
ordering, quadrature ordering, and any geometry-precision override. Validate
peer ID sequences/counts during setup, including vertex-only sharing and more
than two sharers; a face-neighbor list cannot supply these guarantees.

Introduce an internal partition-to-local-mesh constructor that accepts inherited
physical boundary nodes explicitly. Reuse geometry precomputation and coloring;
skip zero-length launches and allocate correctly shaped empty geometry. For
mixed meshes later, preserve original pressure mappings and physical normals
rather than regenerating them from local topology.

Topology partitioning and ID maps may remain on the host. Select the rank's GPU
before allocation; transfer local coordinates, connectivity, physical BC arrays,
and kernel index lists in bulk during setup. Geometry, state, assembly scratch,
and all kernel-consumed inputs must then use the same selected backend. Do not
run a host topology constructor over CuArrays through scalar indexing.

### Field localization and ownership

Distribution must make the existing problem data usable, not only its mesh.
Use cached maps in a small collective field-scatter/localization helper and
owner-only gather. Root arrays are in original serial order; explicit external
IDs have a separate map. Non-root ranks supply `nothing`. Root need not be
communicator rank zero. Reject incompatible lengths/layouts before data transfer.

| Data | Localization and communication |
|---|---|
| Coordinates, connectivity, physical boundary IDs | Partition once on the host; upload local arrays once. Geometry/colors are recomputed locally with inherited boundaries. |
| Thermal `T`, `T0`, `P`, nodal `source`, nodal `phases` | Localize by node map. At solve entry refresh owner values for every coefficient/history array read by elements. Refresh current `T` before each assembly and before an ordinary return. |
| Material tuples, `dt`, `Tref`, solver controls | Agree communicator-wide; preserve scalar type and phase numbering. Workgroup size and selected device may be rank-local. |
| Residual, `PC`, absolute element row sums | Zero local partial buffers, assemble owned elements, add ghost contributions to owners. Apply algebraic BC constraints afterward. |
| Rate, `R0`, spectral/convergence scratch | Authoritative on owned nodes. Do not reverse-scatter these completed owner quantities; ghosts need no meaningful values unless explicitly read. |
| Later cell-phase rows and `nq × nels` inputs/history | Permute element columns by owned global element IDs, preserving quadrature/local-basis rows; never use the nodal halo operation. |
| Later assembled nodal body-force vector | Localize authoritative global values; apply once on owners after residual accumulation. It is distinct from nodal source density integrated over every owned element. |

BC localization returns owner-indexed ordinary `DirichletBoundaryCondition`
objects. Resolve duplicate constraints by global ID before discarding ghost
constraints; conflicting values are errors. Preserve physical boundary metadata
on every local copy even where that rank does not apply the algebraic constraint.
At every solve entry revalidate/refresh mutable coefficient/history inputs;
do not introduce a dirty-cache protocol in the first milestone. Previous-time
fields remain caller-owned and are never advanced by distribution or `solve!`.

### Optional MPI boundary

Add `MPI` to `[weakdeps]`, `[extras]`, and `[compat]`, and register
`FEMToolsMPIExt = "MPI"` for the direct transport, or
`FEMToolsMPIExt = ["MPI", "PartitionedArrays"]` if phase 0 selects the library
adapter, declaring PartitionedArrays alongside MPI. Add Metis through
`FEMToolsMetisExt = "Metis"` with a core partitioning stub. Add WriteVTK through
an output extension when parallel visualization lands. Choose compatible package
ranges by resolving the dedicated test environment against supported Julia
versions in phase 0; record the selected releases and integration results.
Core source must contain no `using MPI`, MPI runtime initialization, or MPI types.
Keep CUDA dependencies in extensions too. A combined
`FEMToolsMPICUDAExt = ["MPI", "CUDA"]` can supply CUDA transport methods; include
the selected distributed-vector package as an extension prerequisite if needed.
Retain the existing CUDA backend mapping and keep KA exchange kernels generic.
MPI and MPI+CUDA extensions must communicate through small core-owned hooks,
not types/private names defined in a sibling extension whose load order is
unspecified. Document which extension owns each method and test both import
orders; prefer dispatch on the backend/buffers over runtime extension probing.

Use a concrete distributed wrapper, provisionally
`DistributedMesh{M,P,C}(local_mesh, partition, context)`, with a concretely typed
MPI context defined by the extension. Keep transport operations internal; avoid
a general communicator/backend framework. Serial methods use a small no-op
execution context where the shared iteration needs dispatch.

The application initializes MPI and supplies a communicator; FEMTools never
finalizes MPI or frees that caller-owned communicator. Each distributed context
uses a private duplicated communicator for message isolation. Direct MPI owns
and frees its copy; the selected library adapter must prove an equivalent
public lifecycle without double-freeing library-owned copies. All rank/root/
neighbor IDs are relative to this communicator, never implicitly `COMM_WORLD`.

Provide an explicit collective close operation while MPI is still initialized.
It completes outstanding work and releases owned resources; a Julia finalizer
must not initiate MPI collectives. Reject use after close and test repeated
setup/solve/gather/close, two independent contexts, and split/reordered
communicators. One context permits only one operation at a time initially.

Keep MPI calls outside KA kernels. Query `MPI.Query_thread()` and honor the
provided thread level, including calls made by library-created tasks. A serial
orchestration loop does not prove those tasks remain on MPI's main thread.
The baseline driver requests `MPI.Init(threadlevel=:funneled)` and confines MPI
calls to that thread; if the selected adapter needs another level, request and
validate it explicitly. See [MPI environment/thread support](https://juliaparallel.org/MPI.jl/stable/reference/environment/).
Reserve message tags per operation/field on the private communicator and check
count/index-width limits before posting communication.

### Communication contract

Implement two operations with different meanings:

| Operation | Direction | Result |
|---|---|---|
| State refresh | Owner to ghost | Overwrite ghost values with authoritative state. |
| Assembly accumulation | Ghost to owner | Add partial element contributions to the owner's local contribution. |

With PartitionedArrays selected, implement these operations through awaited
`consistent!` and `assemble!` calls on cached vector adapters. Keep explicit
owned reductions for solver formulas; confirm any library reduction before using
it so ghosts cannot enter convergence statistics.

For direct MPI fallback, precompute all index lists and allocate typed contiguous
buffers once per field layout and scalar type. Post receives for all peers,
then sends, complete all
requests, then unpack. This avoids neighbor-order deadlocks without introducing
compute/communication overlap. MPI.jl supplies `Irecv!`, `Isend`, and `Waitall`;
buffer reuse occurs only after completion. See its
[point-to-point reference](https://juliaparallel.org/MPI.jl/stable/reference/pointtopoint/).

Clear partial assembly buffers each assembly. Never reverse-scatter a buffer
whose ghost entries already hold a completed global result. Refresh assembled
ghost residuals/diagonals only when a subsequent computation actually reads them;
owner-only scalar updates do not require that extra exchange.
Use one contract for both transports: after reverse scatter, owned entries hold
the sum and consumed ghost partials are zero. A later owner-to-ghost refresh
changes them into copies, which must be cleared before another accumulation.
Test two successive assemblies on the same buffers so stale ghost values cannot
silently double-count the previous pass.

Static nodal inputs (phases, source density, frozen pressure/temperature, previous
time fields) are refreshed at solve setup or after explicit changes. Current
unknowns are refreshed before local element assembly and before returning a
successful or non-converged state. Local quadrature/cell arrays follow owned
element IDs. Caller writes to ghost entries are overwritten by owner refresh.

### Mandatory CUDA execution and transport

Use one GPU per MPI rank. The application selects it before distributed setup,
using scheduler visibility or node-local rank, and records the mapping. Reject
unintended device sharing in hardware tests. Device selection is task-local:
any asynchronous transport task touching CuArrays must establish the correct
device. Buffers and MPI requests must outlive both GPU work and communication.
See [CUDA multi-GPU guidance](https://cuda.juliagpu.org/stable/usage/multigpu/).

The baseline requires ordinary MPI and a functional CUDA device, without
requiring MPI to accept device pointers. Preallocate GPU pack/receive buffers,
matching host buffers, and device index lists for each exchange layout. Keep
array-valued material tables, unknowns, residuals, diagonals, geometry, and rates
on GPU; compile-time tuples/constants may remain kernel arguments.
Pinned host buffers are an optimization after the ordinary staged path passes.

For both owner-to-ghost copy and ghost-to-owner addition:

1. Pack selected entries with KA kernels into contiguous device send buffers.
2. Complete producing GPU work and bulk-copy only those buffers to the host.
3. Post host receives and sends for all peers, then wait for all MPI requests.
4. Bulk-copy received halos to device buffers; unpack with KA kernels.
5. Complete unpacking before dependent assembly, updates, or reductions.

Reverse scatter must handle multiple peers contributing to the same owner.
Initially use conflict-free per-peer index lists and serialize additive unpack
kernels on the same stream, or prove an atomic scatter implementation. Copy
unpack has one authoritative source per ghost. Empty lists skip launches while
idle ranks still participate in collective decisions. Never reuse staging
buffers before all relevant transfers and MPI requests finish.

Compute owned norms, signed Rayleigh sums, maxima, and validity flags on the
device; communicate their scalar results through host MPI reductions. Transfer
only halo payloads and a bounded number of reduced scalars per scheduled check.
Full-field downloads belong to explicit setup, diagnostics, or output boundaries.
Small status reductions are also allowed at the collective failure boundaries
below. Record their frequency and cost; correctness must not depend on checking
errors only at the less frequent convergence checks. The baseline is blocking
with explicit completion; overlap/pinned buffers/request reuse are later work.
Run all CUDA gates with `CUDA.allowscalar(false)`, following the
[CUDA workflow](https://cuda.juliagpu.org/stable/usage/workflow/).

An explicit `transport=:cuda_aware` option may send contiguous CuArray halo
buffers directly once its separate deployment gate passes. Synchronize the
producer before MPI reads device memory, wait for MPI before GPU consumers, and
keep buffers alive throughout. Verify actual multi-GPU device-buffer exchanges
with the configured MPI build; `MPI.has_cuda()` is an OpenMPI-specific check,
not portable proof. A requested unsupported transport fails collectively.
See [MPI.jl GPU support](https://juliaparallel.org/MPI.jl/stable/usage/#GPU-aware-MPI-support).

### Shared DR integration and reductions

Thread the execution context through scalar solve preparation and the shared
DR loop. Reuse the existing residual/Jacobian closures on the local mesh.
The context controls the owned update count, setup/state refresh, reverse
assembly accumulation, scalar reductions, collective failure reporting, and
which rank logs. Do not copy the solver loop into the MPI extension.

Land the initial-BC correction as a separate serial change before MPI wiring:
pin `u` before the first residual and preserve the previous-time field. Compare
an initially inconsistent state with an already constrained initial state using
nonzero BCs and an independent reference. Capture this corrected serial baseline
before comparing distributed iteration. State/BC constructors and
`_mesh_solver_arguments` can fail before entering the DR loop. The distributed
wrappers must cover validation, allocation, and localization at those earlier
boundaries as well as failures inside iteration.

Within an iteration:

1. Snapshot completed owner residuals when the existing schedule requires it.
2. Refresh the current field; assemble all local owned elements.
3. Accumulate `R` to owners; on Jacobian refreshes also accumulate `PC` and
   absolute row-sum diagnostics. Sum the same element diagnostics as serial,
   rather than taking absolute values after cross-rank summation.
4. Constrain owner residuals/rates; validate required owner preconditioners.
5. Obtain the global spectral bound when due; update rates/unknowns over the
   owned range only and repin owner BCs.
6. At the existing checks, reduce owned statistics globally, derive common
   Chebyshev parameters, and make one common convergence/failure decision.

Preserve the current ordering of residual sampling and updates, tolerance meaning,
check cadence, return fields, and history records except for the separately
tested initial-BC correction. Do not bundle a residual-history redesign into MPI.

| Quantity | Required global operation |
|---|---|
| L2 norm | `sqrt(SUM_r(sum_owned(abs2, R)))`; ghosts contribute zero. |
| History RMS | Global norm divided by `sqrt(global_nnodes)`, including constrained nodes in the count as serial does. |
| Relative scalar residual | Global current norm divided by the guarded global initial norm. |
| Upper spectral bound | `MAX_r(max_owned(rowsums ./ PC))`, after reverse assembly and collective validity checks. |
| Lower spectral estimate | Reduce signed Rayleigh numerators with SUM, then take absolute value; separately SUM denominators. Never sum local absolute numerators or local ratios. |
| Nonfinite/invalid flags | Reduce a failure flag so every rank takes the same exit path. |
| Later pressure gauge | Global weighted numerator divided by global weight; count each pressure DoF/element once. |

For idle ranks, sums/counts are zero, maxima have an appropriate neutral value,
and local validity is true. An idle rank still participates in all global
reductions. Local zero Rayleigh denominator must not skip a collective: branch
only after reducing the denominator globally. Global zero denominator preserves
the existing zero-estimate fallback. Keep field arithmetic and buffers in the
original Float32/Float64 type. MPI.jl provides the required all-rank operations
in its [collective reference](https://juliaparallel.org/MPI.jl/stable/reference/collective/).

### Collective setup, failure, and return protocol

Validate rank agreement before any schedule-dependent branch: root, precision,
element family, transport/backend family, material tables, `dt`, tolerances,
positive check interval, iteration budget, and history/failure controls. Local
lengths and physical values legitimately differ; validate them against the
local maps instead. No rank can reject an ordinary argument then leave peers
entering its first exchange.

| Boundary | Required order before the next communication |
|---|---|
| Root partitioning | All ranks enter setup; root catches input/Metis/preparation errors and broadcasts success or a bounded diagnostic **before** scatter. Peers never wait for payloads after a root-only failure. |
| Local setup | Distribute sizes/type descriptors; validate/allocate receive buffers and agree readiness before payload scatter. After scatter, construct local geometry/caches and agree readiness again before any halo or solve. |
| State refresh | Complete local packing/device producers; agree success before posting sends/receives. All requests complete before unpack. Validate unpack completion before dependent assembly. |
| Element assembly | Capture local execution/numerical failure and complete device work. Reduce a status before reverse scatter; reject invalid contributions without silently replacing them. |
| Owner update/check | Validate finite residuals/rates/unknowns and required positive finite owner diagonals. Agree status before the next halo/check; check the global reduced outputs as well. |
| Return | Agree convergence or budget exhaustion, refresh final unknown ghosts, then return matching global statistics/history on all ranks; only the designated rank logs. |

Reduce status with an integer/Boolean operation, not NaN-sensitive arithmetic.
Select the lowest failing communicator rank to distribute one bounded message
with phase/iteration/global ID where available; all ranks raise the same error
class after all requests are complete. `throw_on_failure=false` continues to
permit budget exhaustion only, not malformed input or invalid arithmetic.
Test every ordinary failure boundary by injecting failure on one rank.

An exception or lost rank during active MPI communication, or an unusable CUDA
context, is fatal. The application/worker must abort its configured failure
domain; ordinary exceptions or `MPI.ERRORS_RETURN` do not promise recovery.
Do not attempt a status collective that peers cannot reach. Every job has an
external timeout that terminates the process tree. On numerical failure the
mutable iterate may be partial; accepted physical histories remain unchanged.
Automatic rollback/retry is outside this milestone.

## Implementation sequence and gates

Each phase should be a reviewable change with a recorded gate result. Phases 0
and 1 supply independent prerequisites for phase 2; pure metadata work can
proceed while GPU access is pending. Transport selection remains provisional
until its GPU gate passes. Then follow 2 -> 3 -> 4; the later physics/capability
work begins only after the thermal milestone is complete.

| Phase | Work | Acceptance gate |
|---|---|---|
| 0. Baseline and transport experiment | Record current serial/reference results; isolate the initial-BC correction; resolve MPI/PartitionedArrays/Metis/CUDA versions; evaluate the two allowed adapter shapes with small explicit maps. | Corrected serial thermal CPU/CUDA baseline in both precisions; two-CPU/two-GPU copy/add, actual transfer volume, thread support, public communicator cleanup, and cached-buffer evidence. Record a reproducible blocker before choosing direct MPI. |
| 1. Ownership prototype | MPI-independent metadata, Metis assignment adapter, local remapping, inherited boundaries, typed empty partitions, and reconstruction helpers. | Reconstruct tiny T3 and high-order nodal topology exactly; verify one owner/entity, element coverage, peer-map reciprocity, artificial-edge exclusion, shuffled IDs, and empty rank. Test Metis assignment validity and balance on a representative mesh without pinning incidental labels. Ordinary package tests require no MPI. |
| 2. Transport and setup | Optional MPI/CUDA adapters, private context/lifecycle, collective setup protocol, field/BC localization, KA copy/add pack/unpack, cached staged buffers, and isolated launcher. | CPU one/two/four-rank and CUDA one/two-GPU copy/add in both precisions; repeated assemblies, empty/uneven and ghost-only ownership, vertex/multi-peer sharing, nonzero root, split communicator, repeated close, and root/local setup failures. Loading core does not initialize MPI or CUDA. |
| 3. Shared scalar loop | Add serial/MPI context dispatch, owner-only updates, device/global reductions, coherent snapshots, and the agreed failure protocol. | Corrected serial CPU/CUDA behavior and existing inference/allocation/verbosity checks remain valid. Multi-rank fixed-step state and reductions match independent calculations, including signed numerator cancellation and zero-owned-data ranks. |
| 4. Thermal vertical slice | Mesh-aware thermal constructor/solve forwarding, collective validation, localized coefficients/history, global assembly, coherent returned ghosts, and headless CPU/CUDA driver. | Consistent-mass oracle, fixed-state and converged serial parity, source and nonzero shared-node BC, heterogeneous coefficients, Float32/Float64, repeated physical steps, matching return/history semantics, and the failure matrix below. Demonstrate halo-only GPU staging and publish the support matrix. |
| 5. Scalar reuse | Lithostatic constructor/adapter and initial temperature/phase refresh through the same mechanism. | Hydrostatic reference and pressure-dependent EOS agree with serial on CPU and CUDA; invalid preconditioner on one rank fails collectively; test a 3-D scalar mesh on both backends before claiming support. |
| 6. Forward Stokes | First mixed 2-D viscous T7/P1-disc; then Q9/P1-disc and Hex27 cell-pressure as separate increments. Add field-specific maps and all solver reductions. | Owned velocity residual/diagonal and pressure residual/mass/scaling agree with serial at fixed states on CPU and CUDA; converged fields agree under the same gauge; heterogeneous contrasts and nonzero BCs. Each discretization earns support independently. |
| 7. Further capabilities | Coupled solves, viscoelastic/plastic accepted history, adjoints, CUDA-aware optimization, scalable setup, and parallel output in separate changes. | Capability-specific gates below, with CPU/CUDA agreement for each solver feature; none is implied by scalar or viscous Stokes milestones. |

Add a minimal Ubuntu Julia 1.12 MPICH MPI job during phase 2, and expand it as
phases land. Keep the current serial jobs. Resolve/test Julia 1.11 separately:
the living testing guide records a JET constraint issue there, so do not claim
minimum-version coverage from a 1.12 run. Validate Windows Microsoft MPI locally
before documenting Windows MPI support. These implementation choices follow
MPI.jl's documented platform defaults; cluster-specific configuration uses
MPIPreferences in the application environment. See
[MPI configuration](https://juliaparallel.org/MPI.jl/stable/configuration/).

Establish a separate GPU runner or recorded hardware gate during phase 0 and
keep it required through phase 4. It needs at least two distinct GPUs, one per
rank, and runs staged transport with a compatible ordinary MPI implementation.
Log GPU UUID/device mapping, CUDA runtime/driver, Julia, and MPI versions. Test
four GPUs when available and test inter-node transport on the target cluster
before claiming that deployment. A skipped CUDA test is missing coverage and
does not satisfy the first milestone. CUDA-aware transport has its own runner
configuration and actual device-buffer exchange gate.

For every gate record the working-tree revision, exact command/environment,
precision/backend/rank count, fixtures, tolerances, pass/fail/skip, and evidence
path. A dependency experiment records the resolved package versions and public
APIs used; a timing claim also records warmup and transfer/memory measurements.
Keep large logs/results outside the source tree. Do not mark an unchecked phase
complete because a later plan section describes its intended behavior.

## Stokes and later capability gates

- **Mixed fields:** continuous velocity uses the nodal owner/ghost map.
  Discontinuous pressure DoFs belong to their element owner, with global IDs
  derived from `(global_element_id, local_pressure_dof)`; no pressure halo is
  needed for element-local pressure reads. Continuous pressure is a separate
  supported layout only after its own ownership and reverse-scatter tests.
- **Pressure preparation:** accumulate mass and scaling numerators before
  normalization for a continuous pressure layout. Preserve positive Jacobi
  weights for signed Q9 pressure functions. Do not change the storage modulus
  chosen by `finite_K` or the trial/corrected-pressure distinction.
- **Mixed iteration:** refresh velocity after updates before pressure residual
  assembly; finish shared velocity residual/diagonal sums before updates.
  Convert every inner/outer norm, spectral estimate, stopping branch, and optional
  power-iteration dot product to global owned reductions. Start with the existing
  Gershgorin option and explicitly reject measured spectra until implemented.
  Apply assembled `body_force` only once on owners after reverse residual scatter;
  pointwise source data and partial element loads have different semantics.
- **Pressure gauge:** preserve the current mixed solver behavior (its mean-removal
  call is commented out). Compare incompressible solutions with a common gauge.
  For Hex27, globally reduce the existing mass-weighted constant pressure mode;
  leave the three other cell-local modes unchanged. Do not impose a mean-zero
  condition on a finite-storage pressure problem.
- **Coupling/history:** co-partition thermal and velocity nodes and keep explicit
  temperature-to-pressure maps. Globalize thermal convergence and any `Q2D`
  source normalization. Keep quadrature stress, corrected pressure, and plastic
  histories element-owned; commit history once per globally accepted physical
  step. Check multi-step agreement, rather than only one converged state.
- **Adjoints:** transpose the distributed gather/scatter schedule as well as
  the element operator. Require a global owned inner-product transpose identity
  and objective-gradient finite differences. Global objectives/material gradients
  must count each contribution once. Preserve the distinct 2-D and 3-D adjoint
  signs. Preserve storage and augmentation blocks already checked by
  `test/test_adjoint_full_system.jl`; separately gate any finite-`K`/plastic-history
  regime, operator mode, and layout not covered by that oracle. Do not
  differentiate MPI calls through Enzyme as a shortcut.
- **CUDA-aware optimization:** baseline staged CUDA support is mandatory in
  phases 0-4. Add direct device-buffer communication only after verifying the
  selected MPI implementation on real GPUs, then repeat halo, numerical, failure,
  and buffer-lifetime gates. Benchmark it against host staging on the same case.
- **Setup and partitioning:** root mesh/metadata construction is an explicit
  memory ceiling even with Metis. Measure it before introducing rank-local input
  or selecting a distributed partitioner such as a suitable ParMETIS binding.
  The scalar milestone has distributed iteration,
  but root setup is not a scalable mesh-loading solution. Dynamic repartitioning
  and mesh motion require map/cache invalidation rules and separate validation.
- **Output:** use owner-only gather by ID for tiny tests and root visualization.
  Use WriteVTK's `pvtk_grid` for rank `.vtu` pieces and one `.pvtu` header, with
  part IDs equal to rank + 1. Write owned elements and all referenced local
  points after refreshing their values; duplicated interface points carry
  stable global IDs and owner metadata. Validate header/piece paths, field
  schemas, empty pieces, and relocated directories. Preserve the existing
  high-order linearization initially. CUDA output explicitly downloads the local
  required arrays once per output call; writers must not scalar-index CuArrays.
  Agree schemas/output step collectively before writing. Record each piece's
  success, and publish the header only after all required pieces are complete;
  a failed rank-local write must not leave a new header advertising an
  incomplete step. Verify whether `pvtk_grid` header timing permits this or
  needs a temporary-name/rename adapter in its integration gate. See
  [parallel-file documentation](https://juliavtk.github.io/WriteVTK.jl/stable/metadata/parallel/).
  Checkpoints must preserve mesh/global IDs, physical time, materials,
  accepted histories, and versioning, and pass a continuation test. Serial VTK
  gathering is neither parallel output nor a checkpoint.

## Proposed user surface and files

Keep ordinary `Mesh`, `MixedMesh`, and existing `solve!` calls intact. Provisional
qualified names are `FEMTools.distribute_mesh`, `FEMTools.DistributedMesh`,
`FEMTools.scatter_field!`, `FEMTools.localize_bc`, and `FEMTools.gather_owned`;
finalize their docstrings, collective close operation, and API tests in phase 4.
These names are proposals, not new exports or promises that existing methods
already accept distributed arguments.

`distribute_mesh(mesh_or_nothing, assignment_or_nothing, comm;
root=0, backend=CPU(), transport=:host_staged)` is
collective: root supplies the serial mesh and an element assignment, or `nothing`
for automatic Metis partitioning; other ranks pass `nothing` for both. Automatic
partitioning requires loading Metis, with a collective missing-capability error
if unavailable. Accept optional `node_ids`/`element_ids` in the root setup
contract; defaults are original serial positions.
It returns a wrapper carrying the rank-local mesh, metadata,
and execution context. Each rank allocates its usual physics state from that
wrapper. `backend` is rank-local and selects CPU or the application's active
CUDA device; `transport` is agreed across ranks. The staged option uses ordinary
host buffers on CPU and packed host halos on CUDA. Validate backend/transport
compatibility collectively before launching kernels or creating requests.
`ThermalDiffusionDR(distributed_mesh, material)` allocates the ordinary state
over the local mesh including ghosts. Its distributed wrapper is collective
so validation/allocation failures reach peers before a later solve. Plain local
state constructors and BC constructors retain their existing serial semantics.

`scatter_field!(local_array, root_array_or_nothing, distributed_mesh; root, association)`
and `localize_bc(root_bc_or_nothing, distributed_mesh; root)` are collective;
the former fills a specified nodal or element layout using cached maps, and
the latter returns only owner constraints. The phase-4 driver uses them to
distribute an existing serial problem with nonuniform `T0`, source, and phases.
Coordinate-based initialization remains an alternative, not a replacement for
localizing caller-owned data. Set current `T` and previous-time `T0` explicitly.

`gather_owned(local_array, distributed_mesh; root, association)` reconstructs
serial order by stable ID, returns the global result on root and `nothing`
elsewhere, and never concatenates ghosts or implicitly changes ownership.
Document association/layout, root, mutation, backend transfer, and collective
participation for every helper. Public `solve!(dr, distributed_mesh, local_bc;
dt, ...)` cannot infer whether an in-range integer array contains global indices;
require localized BCs and validate their owner-range membership. Unsupported
combinations fail collectively rather than falling back to serial.

| Proposed path | Responsibility |
|---|---|
| `src/distributed/partition.jl` | Global IDs, field/element maps, pure partition preparation, wrapper, inherited boundaries, typed empty local meshes. |
| `src/distributed/execution.jl` | Small internal serial context and communication/reduction hooks consumed by shared loops. |
| `src/distributed/exchange_kernels.jl` | Backend-neutral KA packing, owner copy/add unpacking, and owned reduction helpers. |
| `ext/FEMToolsMPIExt.jl` | Selected transport, private communicator lifecycle, setup/status protocol, field/BC localization, reductions, owned gather. |
| `ext/FEMToolsMPICUDAExt.jl` | CUDA staging buffers, bulk transfers, synchronization, and optional validated device-aware transport. |
| `ext/FEMToolsMetisExt.jl` | Element-graph partition assignment, label normalization, and small/empty-part handling. |
| `ext/FEMToolsWriteVTKExt.jl` | Later parallel visualization using rank pieces and one header. |
| `src/dynamic_relaxation/solver.jl` | Shared owner/communication boundaries and collective solver diagnostics. |
| Scalar constructor/solver files | Mesh wrapper forwarding and field refresh requirements. |
| `test/test_partition.jl` | Pure serial partition invariants. |
| `test/mpi/Project.toml`, `runtests.jl`, worker scripts | Explicit MPI environment, launcher/timeouts, CPU and required CUDA communication/solver workers; separate GPU environment if dependency resolution requires it. |
| `examples/mpi/thermal_diffusion.jl` | Headless CPU/CUDA driver with unconditional entry point and explicit device mapping; add MPI, Metis, and PartitionedArrays if selected to examples (CUDA is already present). |
| `docs/src/distributed.md`, `docs/make.jl` | Supported combinations, setup, ownership, launcher, failure semantics, limitations. |

Include the distributed core files after mesh/BC definitions and before the DR
loop. Test extension loading in relevant MPI/CUDA/library import orders and
without MPI or CUDA. Keep MPI
workers out of the automatically discovered serial test list. Update exports/
`public` declarations, API tests, source docstrings, and living guides together;
raw exchange helpers remain internal.

## Validation and performance

Use an isolated MPI test project with FEMTools sourced from the working tree.
The parent launcher selects the matching executable using `MPI.mpiexec()` and
`Base.julia_cmd()`, as recommended by
[MPI.jl testing guidance](https://juliaparallel.org/MPI.jl/stable/usage/#Writing-MPI-tests).
Log Julia/MPI versions and rank-qualified failures; enforce job and worker
timeouts and terminate the complete process tree on timeout. Workers must
aggregate assertion failures at agreed boundaries rather than stop one rank
while others enter another exchange.

The numerical ladder is fixed-state operator agreement first, a fixed number
of DR steps next, then converged-solution/reference agreement. The serial path
alone can share MPI integration bugs. For a constant-material, affine T3 fixture
with `α=0`, `K=Inf`, and no additional IP source, independently construct

```math
M_e=\frac{A_e}{12}\begin{bmatrix}2&1&1\\1&2&1\\1&1&2\end{bmatrix},
\quad L_e=k A_e B_e B_e^T,\quad c=\frac{dt}{ρ_0 C_p},
```

```math
R_e=M_e(T_e^0-T_e)+c(M_e s_e-L_e T_e),
\qquad J_e=-M_e-cL_e.
```

Here `B_e` has one physical shape-gradient row per local node. The current
three-point degree-two T3 rule integrates these terms exactly. Derive `B_e`
and area directly from vertex coordinates; do not call production residual,
geometry, or Jacobian helpers inside the oracle. Assemble the sparse reference
only in tests and apply Dirichlet elimination independently. Check `PC` and
row sums against **sums of element absolute diagonals/row sums**, not absolute
values of the assembled matrix. Choose nonconstant `T0`, source, and nonzero BCs
so the case exercises communication and has unconstrained interior nodes.
Use separate heterogeneous/nonlinear-density serial comparisons to cover
coefficient/phase localization outside this constant-material oracle.

Exact checks apply to IDs, topology, copy exchange, integer-valued accumulation,
and BCs.
Numerical tolerances depend on precision, number of summed terms, problem scale,
and solver stopping tolerance; record them per case. Do not require bitwise
identical multi-rank histories or exact convergence iteration counts near a
threshold, and do not loosen package-wide tolerances to accommodate MPI.

The required fixture matrix is finite and belongs in the worker suite:

| Fixture | Required evidence |
|---|---|
| One rank, uneven two/four ranks, Metis and explicit assignments | Same reconstructed topology, fixed-state residual/diagnostics, fixed-step updates, and converged thermal solution in both precisions. |
| Vertex-only sharing, three-or-more sharers | Every ghost reaches the unique owner; each element contribution is counted once. CPU four-rank coverage is mandatory; repeat on four GPUs when available. |
| Empty rank and rank with elements but zero owned nodes | Correct assembly/exchange, neutral reductions, valid empty geometry, no zero-length launches, and unchanged collective sequence. |
| Reused buffers and a second physical step | No stale ghost accumulation; changed `T0`/source/phases and updated BC values refresh correctly without advancing history implicitly. |
| Nonzero root and split/reordered communicator | No implicit world-rank/part-label assumption; field/BC localization, gather, and failure-rank reports remain correct. |
| Control mismatch, root Metis/input failure, local allocation/geometry failure | Coherent failure before scatter/iteration; timeout proves no stranded rank. Use an injected allocation failure rather than deliberately exhausting machine memory. |
| One-rank invalid diagonal, nonfinite field, pack/assembly failure | All reachable peers receive the agreed error; active-communication fatal injection is a separate expected-abort worker. |
| Budget ending between checks, history on/off | Preserve last-check residual/history semantics, coherent returned ghosts and global stats, and matching throw/return behavior. |
| Repeated setup/close and two contexts | No stale messages, outstanding requests, communicator ownership errors, or growing per-solve buffers after warmup. |

Run the applicable numerical/communication fixtures on CPU at 1/2/4 ranks and
CUDA at 1/2 ranks on distinct GPUs, always with scalar indexing disabled.
A two-GPU run must not silently substitute two ranks on the same GPU. Tests
requiring more ranks than available GPUs retain mandatory CPU coverage and an
explicit outstanding/optional four-GPU result rather than pretending to run.

At phases 3/4, run the full serial `Pkg.test()` and docs build because the shared
solver/API changes. Run focused MPI gates after every transport/solver change.
At handoff review all nine subsystem guides, `git diff --check`, and final status.

After correctness, benchmark warmed CPU and CUDA thermal and heterogeneous
viscous Stokes cases with identical meshes and tolerances at 1/2/4 ranks where
hardware permits, then larger rank counts on the intended cluster. Report maximum
rank wall time, iterations, element sweeps, setup/assembly/packing/exchange/
reduction time, network and host/device transfer bytes, owned/ghost counts,
host/device memory, hardware/MPI implementation, GPU mapping, and thread/rank
binding. Compare staged and CUDA-aware transport when both are supported.
Separate setup and compilation from iteration time. Include strong and weak
scaling; record declining efficiency rather than promising a speedup. Only then
consider packed multi-component exchanges, request reuse, interior/boundary
overlap, or fewer collective calls that preserve numerical semantics.

## Stop condition and remaining deployment choices

Phases 0-4 are the first complete milestone. It is done when the CPU thermal
driver/MPI CI and CUDA thermal driver/required two-GPU staged hardware gates
pass in both precisions, with correct collective failure/empty-rank behavior,
GPU-resident iteration, and no host scalar indexing. Serial tests/docs must
pass and ownership must be documented. Missing GPU hardware leaves the CUDA
gate outstanding. Phase 5 demonstrates shared scalar reuse; Stokes and phase 7
features have independent CPU/CUDA completion gates. The release evidence must
also name the selected transport/dependency versions, demonstrate localized
caller data and repeated physical steps, show communicator teardown and failure
fixtures, and record the exact supported combination. Gate results are
recorded under "Current state".

Metis provides the initial automatic partitioner; explicit assignments remain
available for deterministic fixtures and externally partitioned meshes. Before
cluster tuning, establish the target mesh sizes/topologies, cluster MPI and
scheduler, CPU/GPU placement, practical root-setup ceiling, and required output
format. Those deployment facts determine later optimization and dependency
choices; they do not change the ownership and assembly contract above. Generic
distributed storage and visualization should be delegated to the selected
packages wherever the integration gates establish compatibility.
