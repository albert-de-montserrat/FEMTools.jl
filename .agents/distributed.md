# Distributed Parallelization

## Current reality

The package has MPI-free partition metadata in the core and an MPI execution
layer in `FEMToolsMPIExt` (loads with `MPI` + `PartitionedArrays`). No public
solver accepts a distributed mesh yet; all names below are internal.

- `src/distributed/partition.jl`: `partition_mesh(mesh, assignment, nparts)`
  returns one `MeshPartition` per part with serial node and element IDs,
  owned-first local nodes, local connectivity, inherited physical boundary
  nodes, and halo entries carrying a dense interface-node ID and owning part.
  `local_mesh` builds the element-aware local `Mesh` on a backend with those
  boundary nodes. `localize_dirichlet` keeps owner constraints only;
  `reconstruct_owned` rebuilds serial order and rejects missing or duplicate
  owners. `partition_elements` (Metis extension) partitions the face-adjacency
  graph.
- Parts are labelled `1:nparts`; rank `r` holds part `r + 1`. A node is owned
  by its lowest incident part.
- `src/distributed/execution.jl` holds stubs for `distribute_mesh`,
  `scatter_field!`, `localize_bc`, `gather_owned`, `collective`, and `agree`.
  `FEMToolsMPIExt` implements them on its `DistributedMesh`: a private
  `Comm_dup` freed by `close`, root-side partitioning and serialized scatter,
  local mesh on the backend, and cached host/device halo buffers for the mesh
  float type and `Int32` only (other element types are rejected).
- Halo transport is host-staged: a KA kernel packs shared entries, the host
  buffer goes through PartitionedArrays `consistent!` (owner to ghosts) or
  `assemble!` (ghosts added to owners, ghosts zeroed), then a kernel unpacks.
  Only halo entries cross the host/device boundary.
- The shared DR loop accepts `execution = distributed_mesh`; `nothing` is the
  serial path. The extension supplies owner ranges, halo refresh/accumulate,
  global reductions, and collective guards through the `_dr_*` hooks in
  `src/dynamic_relaxation/solver.jl`.
- Thermal-specific glue (state allocation, coefficient refresh, `solve!`
  forwarding) is still in
  `examples/miniapps/thermal/2D_heat_diffusion_MPI/mpi_thermal.jl`.
- `test/mpi/` has its own environment, launcher, and two workers; CI job `mpi`
  runs them on 1/2/4 CPU ranks. One CUDA rank passes locally; the two-GPU gate
  is unrun. Gate records live in `MPI_IMPLEMENTATION_PLAN.md`.

Existing parallel foundations are local to one process/device:

- KernelAbstractions kernels target CPU and optional GPU backends.
- Atomix supports concurrent scatter within a backend.
- Mesh coloring provides conflict-free element groups.
- `Mesh` itself has process-local numbering with no owned/ghost distinction;
  ownership lives in `MeshPartition`.
- Without an execution context, solver norms, sums, maxima, pressure means,
  and spectral estimates are local reductions. Only the shared scalar DR loop
  has hooks; Stokes reductions remain local.
- VTK output is serial and unaware of partitions.

Distributed work must therefore start with data ownership, not by wrapping the
current solve in `mpiexec`.

## Required model

A useful partitioned mesh must eventually define:

- stable global node and element IDs;
- locally owned elements;
- owned and ghost nodes for each field;
- owner rank and neighbor ranks;
- global-to-local and local-to-global maps;
- send/receive indices for halo exchange;
- physical boundary tags distinct from artificial partition boundaries;
- mixed-field ownership for velocity and continuous or discontinuous pressure;
- device placement of rank-local arrays.

Keep the serial `Mesh`/`MixedMesh` API usable. Add distributed data only when a
serial metadata prototype proves the fields needed by a real assembly path.

## Communication semantics

Define the operation before optimizing it:

- State exchange copies owner values to ghosts before an element computation
  that reads neighboring state.
- Residual scatter accumulates ghost contributions back to owners after local
  element assembly, then refreshes ghosts if the next stage reads the assembled
  result.
- Dirichlet ownership and application are unique and deterministic across
  ranks.
- Global convergence norms use the correct sum/max reduction and global degree
  count.
- Pressure mean removal, spectral bounds, NaN detection, and iteration exit are
  collective decisions.
- All ranks take collectives in the same order, including error paths.
- CPU staging versus device-aware MPI is an explicit backend capability, not an
  assumption.

## Incremental plan

The detailed proposed roadmap is
[`MPI_IMPLEMENTATION_PLAN.md`](../MPI_IMPLEMENTATION_PLAN.md). Its first complete
milestone is a CPU and CUDA T3 thermal step with optional MPI loading,
Metis-generated or explicit element assignments, root-side setup, owner/ghost
maps, additive assembly accumulation, global DR reductions, CPU one/two/four-rank
gates, and required one/two-GPU gates.
CUDA iteration stays device-resident; the portable baseline stages packed halos
through reusable host buffers and globally reduces device-computed scalars.
Lithostatic reuse
precedes forward Stokes; adjoints, CUDA-aware transport optimization, scalable
mesh loading, and parallel persistence have separate acceptance gates.

External package count is unrestricted for this work. The revised plan prefers
PartitionedArrays for owner/ghost exchanges after an integration experiment,
Metis for automatic root-side partitioning, and WriteVTK for later parallel
visualization. Direct MPI transport is a fallback for a demonstrated library
integration blocker. Preserve ordinary local KA arrays and verify communicator
ownership, local ordering, empty ranks, ghost-clearing semantics, CUDA buffer
aliasing, and halo-only staging before adopting the distributed array adapter.
CPU-only integration success is insufficient. The prototype uses
PartitionedArrays with dense interface-node IDs as its index space; the choice
stays provisional until the two-GPU gate passes.

The first proposed solver combination is fixed conforming T3 thermal diffusion
in Float32/Float64, with all ranks on CPU or all on CUDA. A high-order metadata
test does not enable that solver family. The roadmap now requires field/BC
localization as well as mesh distribution, with separate nodal and element/IP
maps and owner-only reconstruction in original serial order. A rank with owned
elements but zero owned nodes still assembles and communicates.

Each proposed execution context isolates messages with a private communicator
and has explicit collective cleanup before MPI finalization. Library adapters
must establish public communicator ownership/cleanup and thread-level support;
a compact shared-node host adapter is acceptable if GPU fields remain resident.
MPI/CUDA sibling extensions share core-owned hooks rather than depending on
one another's private types or load order. These are design requirements, not
existing types or APIs.

The plan identifies two existing integration boundaries: local boundary/normal
inference must not classify partition interfaces as physical boundaries, and
shared assembly must accumulate preconditioner/row-sum data as well as residuals
before global spectral estimates. Initial scalar Dirichlet enforcement also
needs a shared regression before asserting one-rank solver equivalence.

Setup/failure handling covers work before the DR loop: root reports preparation
errors before scatter, ranks agree receive-buffer readiness before payloads and
geometry readiness before halos, and device packing/assembly failures are
reported before the next exchange.
Ordinary numerical failure is collective; failures inside active communication
use the application abort/timeout boundary. Accepted physical histories are not
advanced by failed solves. Return refreshes unknown ghosts and preserves existing
last-check residual/history semantics.

Record each stage's checks. Pure metadata work can proceed while the transport
experiment awaits hardware, but transport selection and the thermal milestone
remain incomplete until the required CUDA gates pass:

1. Define stable global IDs and a serial partition metadata object for a tiny
   mesh. Check ownership, ghosts, and reconstruction without MPI.
2. Add an optional MPI boundary—prefer an extension/isolated dependency path—
   and run the same metadata on one rank.
3. Implement typed CPU/CUDA halo copy and additive exchanges with device
   pack/unpack and reusable host staging; validate on two ranks/two distinct
   GPUs, including an empty neighbor/partition edge case.
4. Distribute one simple assembly/solve path, likely the smallest scalar
   problem, with correct residual accumulation and global convergence on CPU
   and CUDA, with scalar indexing disabled on CUDA.
5. Extend to mixed Stokes fields only after scalar ownership and reductions are
   stable.
6. Add partition-aware output/checkpointing and optional CUDA-aware transport
   based on measured needs. Baseline CUDA compatibility is already required.

Reuse external implementations of partitioning, transport, and file formats when
their integration gates pass. Keep the changes staged: adopting packages does
not remove the ownership, numerical, or validation obligations. Start with
root-side Metis partitioning; choose a distributed partitioner and element
weights after mesh sizes and deployment constraints are known.

## Invariants

- One global entity has one owner; ghosts never become competing sources of
  truth.
- Every owned element is assembled exactly once globally.
- Partition boundaries are not physical boundaries.
- Rank-local ordering may change results only within documented floating-point
  reduction tolerance, not mathematical semantics.
- A one-rank distributed run agrees with the serial path.
- A multi-rank result is compared after reconstruction by global ID, never by
  concatenating rank-local arrays blindly.
- Communication buffers have explicit element type, backend, and lifetime.
- Reverse assembly leaves completed owner sums and zeroed consumed ghost
  partials; two successive assemblies must not add a previous global sum again.
- GPU mesh/state/kernel inputs share a device; setup and output transfers are
  explicit, and hot-loop host transfers are bounded halos and reduced scalars.
- Device producers finish before MPI consumes buffers, and MPI/unpack work
  finishes before dependent GPU kernels or buffer reuse.
- No rank returns early while peers are entering a collective.

## Acceptance checks

Minimum distributed test ladder:

- serial partition metadata invariants;
- `mpiexec -n 1` equivalence with the serial result;
- `mpiexec -n 2` ownership and halo exchange on a tiny deterministic mesh;
- global residual/norm/convergence agreement;
- partition with zero owned entities or no neighbor on one side;
- physical boundary conditions crossing rank layouts;
- mixed-field ownership before distributed Stokes is declared supported;
- required CPU and CUDA-staged solver agreement in Float32/Float64 with scalar
  indexing disabled and one rank per distinct GPU;
- separate real device-buffer exchange tests before exposing CUDA-aware MPI.

All tests need timeouts and collective-safe failure reporting to avoid hanging
CI. Record the exact launcher and MPI implementation once chosen.
A CUDA skip is missing hardware coverage, not completion of the first milestone.

## Open decisions

- PartitionedArrays CPU/CUDA integration result and selected transport adapter.
- Scalability ceiling of root-side Metis setup and later distributed partitioning.
- Package compatibility ranges and supported MPI implementations.
- Later pressure-layout and element/IP field maps.
- Whether the public distributed API wraps `MeshPartition` or replaces it.
- CUDA-aware optimization availability; host staging is the required baseline.
- WriteVTK integration details and checkpoint format/continuation contract.
- Multi-GPU hardware runner/deployment environment (CPU MPI CI exists).
- Float32 driver tolerance: the example driver stalls at 1.32e-5 against 1e-5.

## Update this guide when

- any MPI dependency or distributed type is added;
- global IDs, ownership, ghost layers, communication, or reductions are
  specified or changed;
- a solver becomes distributed;
- a launcher, MPI implementation, GPU transport, or CI job is supported;
- measurements resolve an open decision.
