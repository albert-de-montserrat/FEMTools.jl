# FEMTools.jl

FEMTools.jl provides small finite-element utilities for reference elements, shape
functions, integration points, and simple one-dimensional mesh connectivity.

```@contents
Pages = ["elements.md", "mesh.md", "boundary_conditions.md", "api.md"]
Depth = 2
```

## Quick Start

```julia
using FEMTools
using DomainSets

element = ReferenceElement(QuadraticElement{1, 3})
coords = generate_coordinates(element, 0.0..1.0, 4)
mesh = Mesh(0.0..1.0, 4)
```

## Package

```@docs
FEMTools
```
