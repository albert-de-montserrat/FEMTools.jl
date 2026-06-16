# FEMTools.jl

FEMTools.jl provides finite-element utilities for structured meshes in 1-D, 2-D,
and 3-D: reference elements, shape functions, quadrature rules, connectivity
helpers, sparsity-pattern construction, mesh coloring, and boundary-condition
handling. Array operations are backend-agnostic via
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl), with
optional GPU support through package extensions for CUDA, AMDGPU, and Metal.

```@contents
Pages = ["elements.md", "mesh.md", "boundary_conditions.md", "heat_diffusion.md", "api.md"]
Depth = 2
```

## Quick Start

```julia
using FEMTools
using DomainSets

# 2-D quadrilateral mesh on the unit square
mesh = Mesh((0.0..1.0) × (0.0..1.0), (4, 4))

# Reference element with quadratic shape functions
element = ReferenceElement(QuadraticElement{2, 9})
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
