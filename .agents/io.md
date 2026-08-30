# I/O

## Current reality

Core I/O is intentionally small:

- `write_vtk` in `src/postprocess/postprocess.jl` writes legacy ASCII VTK
  unstructured grids for `Mesh` and `MixedMesh`.
- High-order elements are linearized to corner nodes for VTK output. A 3-D
  `MixedMesh` uses its T11 or Hex27 velocity connectivity; the separate
  pressure connectivity is interpolation data, not cell geometry.
- Scalar point fields and cell fields are supported and length-validated.
- `write_stokes_vtk` maps mixed pressure/velocity and derived stress/strain
  fields through the generic writer in both 2-D and 3-D.
- Core has no mesh reader, checkpoint/restart format, parallel writer, or
  general result schema.
- Examples create meshes with Gmsh.jl and sometimes depend on WriteVTK, but
  these paths are not a unified FEMTools I/O API.

Primary checks: `test/test_vtk.jl` and `test/test_postprocess_api.jl`. Public
documentation is currently under the Mesh and Stokes pages.

## Invariants

- Internal connectivity is 1-based; VTK connectivity is written 0-based.
- Preserve the distinction between point-associated and cell-associated data.
- Validate field lengths, supported topology, and indices before or during
  writing. Never emit a plausible but structurally inconsistent file.
- Moving output from a GPU to the host must be explicit and bounded. Writers
  should not trigger repeated scalar transfers.
- Node and element IDs/order matter for restart, partitioning, and comparison.
  Visualization-only remapping must be documented and must not be mistaken for
  a restart representation.
- File handles close on errors. Material output should use a temporary file and
  atomic replacement once restart/checkpoint data exists.
- Output functions return predictable values (`nothing` or a documented result)
  and do not silently choose changing working directories.

## Direction

Keep visualization output separate from durable simulation state.

Near-term priorities:

1. Consolidate repeated external-mesh conversion rules only when there is a
   concrete importer target. Validate dimension, supported element type, tag
   remapping, node ordering, and boundary groups at the boundary.
2. Extend VTK only for fields/topologies required by maintained examples. Use
   an installed dependency only if it replaces enough custom format surface to
   reduce maintenance.
3. Define checkpoint contents before choosing a format: mesh identity, global
   IDs, partition metadata, numeric type, time/step, solver state, material
   state, stress history, and schema/package version.
4. Design parallel output with the distributed ownership model. Avoid every
   rank independently writing ambiguous copies of ghost data.

A checkpoint is accepted only after a round-trip continuation test reproduces
the uninterrupted result. A visualization file is not a checkpoint.

## Acceptance checks

For writers:

- parse or inspect a tiny written file and check topology, indexing, field
  association, names, and values;
- cover unsupported topology and mismatched field lengths;
- cover high-order linearization when touched;
- use temporary directories/files and remove them after the test;
- verify host output from an affected device path where relevant.

For readers or checkpoints:

- round-trip a tiny mesh/state without changing IDs, types, fields, or time;
- reject malformed, unsupported, truncated, and version-incompatible input;
- restart for at least one step and compare with an uninterrupted solve;
- test empty partitions and ghost ownership once distributed state exists.

## Open decisions

- First supported mesh input format and required boundary/material tags.
- Whether high-order visualization should remain linearized or gain native
  high-order cell output.
- Checkpoint format and compatibility policy.
- Single-file versus per-rank parallel result layout.

## Update this guide when

- a format, field association, topology, reader, writer, or checkpoint changes;
- examples stop owning a conversion path and core/extension code takes it over;
- distributed ownership changes output semantics;
- persistence compatibility or versioning becomes supported behavior.
