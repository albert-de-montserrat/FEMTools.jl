# FEMTools.jl

FEMTools.jl provides finite-element utilities for structured meshes in 1-D, 2-D,
and 3-D: reference elements, shape functions, quadrature rules, connectivity
helpers, sparsity-pattern construction, mesh coloring, and boundary-condition
handling. Array operations are backend-agnostic via
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl), with
optional GPU support through the CUDA package extension.

```@contents
Pages = ["elements.md", "mesh.md", "boundary_conditions.md", "heat_diffusion.md", "lithostatic_pressure.md", "stokes.md", "api.md"]
Depth = 2
```

## Quick Start

Build a structured mesh from a reference element and a domain. The `×` product
operator is not exported by DomainSets, so import it explicitly:

```jldoctest
julia> using FEMTools, DomainSets

julia> using DomainSets: ×

julia> element = ReferenceElement(QuadraticElement{2, 9})
ReferenceElement{QuadraticElement{2, 9, Float64}}(order=2, nodes=9, nips=9)

julia> mesh = Mesh((0.0..1.0) × (0.0..1.0), element, (4, 4))
Mesh{2, 2}(nnodes=81, nels=16)
```

## Workflows

A solve needs a mesh, a material, boundary conditions, and physical controls.
The state takes its backend, precision, and field sizes from the mesh, and
`solve!` owns its scratch storage. One backward-Euler heat-diffusion step with
a uniform source and zero boundary temperature:

```@example workflow
using FEMTools, DomainSets
using DomainSets: ×

element = ReferenceElement(QuadraticElement{2, 9})
mesh = Mesh((0.0..1.0) × (0.0..1.0), element, (8, 8))
thermal = ThermalDiffusionDR(mesh, ThermalMaterial(; k = 1.0, α = 0.0, K = Inf))
fill!(thermal.source, 1.0)
bc = DirichletBoundaryCondition(mesh.Γnodes, zeros(length(mesh.Γnodes)))
stats = solve!(thermal, mesh, bc; dt = 0.1, verbose = false)
(stats.converged, maximum(thermal.T))
```

A dense, viscous inclusion sinking in an incompressible (`ηb = Inf`) no-slip
box, with T7 velocity and discontinuous P1 pressure. Phases are assigned per
cell:

```@example workflow
element_v = ReferenceElement(QuadraticElement{2, 7})
element_P = ReferenceElement(LinearElement{2, 3})
mesh_v = Mesh((0.0..1.0) × (0.0..1.0), element_v, (8, 8))
mesh = MixedMesh(mesh_v, element_P)
material = StokesMaterial(;
    η = (1.0, 10.0), ηb = Inf, ρ0 = (1.0, 2.0), g = (0.0, -1.0), Tref = 0.0,
)
stokes = StokesDR(mesh, material)

centroid(e) = sum(mesh.coords[mesh.el2nP[:, e]]) / 3
phases = reshape([hypot((centroid(e) .- 0.5)...) < 0.2 ? 2 : 1 for e in 1:mesh.nels], 1, :)
no_slip = DirichletBoundaryCondition(mesh_v.Γnodes, zeros(length(mesh_v.Γnodes)))
stats = solve!(stokes, mesh, (no_slip, no_slip);
               dt = 1.0, phases_v = phases, phases_P = phases, verbose = false)
(stats.converged, minimum(stokes.v.y))
```

`solve!` throws if the iteration does not converge; pass
`throw_on_failure = false` to inspect failed statistics instead. The
[Heat Diffusion](heat_diffusion.md) and [Stokes](stokes.md) pages describe the
controls, coupling, adjoints, and plasticity.

## GPU support

Load CUDA to activate the GPU backend extension:

```julia
using CUDA   # CuArray support
```

The `TA(backend)` helper returns the array constructor for a given backend so
that the rest of the code stays generic.

```@docs
FEMTools.TA
```

## Package

```@docs
FEMTools
```
