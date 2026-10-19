# Distributed Parallelization

## Current reality

FEMTools.jl does not currently implement distributed-memory parallelism.
`MPI.jl` is not a core, weak, test, or examples dependency. MPI runtime
artifacts visible in the examples manifest arrive transitively and do not
constitute MPI support.

Existing parallel foundations are local to one process/device:

- KernelAbstractions kernels target CPU and optional GPU backends.
- Atomix supports concurrent scatter within a backend.
- Mesh coloring provides conflict-free element groups.
- Meshes currently use process-local node and element numbering with no
  owned/ghost distinction.
- Solver norms, sums, maxima, pressure means, and spectral estimates are local
  reductions.
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
These are planned choices, not implemented capabilities. Lithostatic reuse
precedes forward Stokes; adjoints, CUDA-aware transport optimization, scalable
mesh loading, and parallel persistence have separate acceptance gates.

External package count is unrestricted for this work. The revised plan prefers
PartitionedArrays for owner/ghost exchanges after an integration experiment,
Metis for automatic root-side partitioning, and WriteVTK for later parallel
visualization. Direct MPI transport is a fallback for a demonstrated library
integration blocker. Preserve ordinary local KA arrays and verify communicator
ownership, local ordering, empty ranks, ghost-clearing semantics, CUDA buffer
aliasing, and halo-only staging before adopting the distributed array adapter.
CPU-only integration success is insufficient. These packages are proposed,
not current dependencies or validated capabilities.

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
- Implementation of the specified lowest-incident-rank nodal ownership and
  later pressure-layout maps; split-communicator and ghost-only-rank validation.
- CUDA-aware optimization availability; host staging is the required baseline.
- WriteVTK integration details and checkpoint format/continuation contract.
- CPU MPI CI and required multi-GPU hardware runner/deployment environments.

## Update this guide when

- any MPI dependency or distributed type is added;
- global IDs, ownership, ghost layers, communication, or reductions are
  specified or changed;
- a solver becomes distributed;
- a launcher, MPI implementation, GPU transport, or CI job is supported;
- measurements resolve an open decision.
