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

Stop after each stage until its check passes:

1. Define stable global IDs and a serial partition metadata object for a tiny
   mesh. Check ownership, ghosts, and reconstruction without MPI.
2. Add an optional MPI boundary—prefer an extension/isolated dependency path—
   and run the same metadata on one rank.
3. Implement one typed halo exchange for a nodal scalar field and validate it
   on two ranks, including an empty neighbor/partition edge case.
4. Distribute one simple assembly/solve path, likely the smallest scalar
   problem, with correct residual accumulation and global convergence.
5. Extend to mixed Stokes fields only after scalar ownership and reductions are
   stable.
6. Add partition-aware output/checkpointing and accelerator transport based on
   measured needs.

Do not add a partitioner, MPI wrapper hierarchy, asynchronous scheduler, and
parallel file format in the first change. Choose a partitioning dependency only
after mesh sizes, element weights, and deployment constraints are known.

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
- CPU first, then explicit tests for each claimed GPU-aware or staged path.

All tests need timeouts and collective-safe failure reporting to avoid hanging
CI. Record the exact launcher and MPI implementation once chosen.

## Open decisions

- Partitioning library and whether partitions are loaded or computed.
- MPI dependency placement and supported MPI implementations.
- Node/element ownership rule and pressure-field ownership.
- Host-staged versus device-aware communication.
- Parallel visualization and checkpoint layout.
- CI environment for multi-rank tests.

## Update this guide when

- any MPI dependency or distributed type is added;
- global IDs, ownership, ghost layers, communication, or reductions are
  specified or changed;
- a solver becomes distributed;
- a launcher, MPI implementation, GPU transport, or CI job is supported;
- measurements resolve an open decision.
