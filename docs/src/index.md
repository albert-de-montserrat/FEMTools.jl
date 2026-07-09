# FEMTools.jl

FEMTools.jl provides finite-element utilities for structured meshes in 1-D, 2-D,
and 3-D: reference elements, shape functions, quadrature rules, connectivity
helpers, sparsity-pattern construction, mesh coloring, and boundary-condition
handling. Array operations are backend-agnostic via
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl), with
optional GPU support through package extensions for CUDA, AMDGPU, and Metal.

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

## GPU support

Load one of the GPU back-end packages to activate the corresponding extension:

```julia
using CUDA   # CuArray support
using AMDGPU # ROCArray support
using Metal  # MtlArray support
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
