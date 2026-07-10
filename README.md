# FEMTools.jl

[![CI](https://github.com/albert-de-montserrat/FEMTools.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/albert-de-montserrat/FEMTools.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/albert-de-montserrat/FEMTools.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/albert-de-montserrat/FEMTools.jl)
[![docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://albert-de-montserrat.github.io/FEMTools.jl/dev/)

Finite-element utilities for structured and unstructured meshes: reference
elements, shape functions, integration points, sparsity construction, mesh
coloring, and boundary-condition handling. Built on these primitives are
pseudo-transient dynamic-relaxation solvers for heat diffusion, lithostatic
pressure, and incompressible (visco-elasto-plastic) Stokes flow. Array
operations are backend-agnostic via
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl),
with optional GPU support through package extensions.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/albert-de-montserrat/FEMTools.jl")
```

## Elements

Element types are lightweight tags parameterized by reference dimension and
number of local nodes. Supported elements:

| Type | Instances |
|:-----|:----------|
| `LinearElement{nDim, nVert}` | line (1×2), triangle (2×3), quad (2×4), hex (3×8) |
| `QuadraticElement{nDim, nVert}` | line (1×3), triangle (2×6), quad (2×9), hex (3×27) |

```julia
using FEMTools

line  = ReferenceElement(LinearElement{1, 2})
quad  = ReferenceElement(LinearElement{2, 4})
hex   = ReferenceElement(LinearElement{3, 8})
quad9 = ReferenceElement(QuadraticElement{2, 9})
```

## Shape Functions

`ShapeFunctions(element)` returns a container with shape functions `N` and
reference-coordinate gradients `∇N`, both ordered by local node number.

```julia
sf = ShapeFunctions(QuadraticElement{2, 6})

sf.N[4](0.5, 0.0)    # evaluate node-4 shape function at (ξ, η) = (0.5, 0.0)
sf.∇N[4](0.2, 0.3)   # gradient at (0.2, 0.3)
```

For a `ReferenceElement`, evaluate all shape functions at once into an `SVector`:

```julia
el = ReferenceElement(LinearElement{2, 3})

eval_shape_function(el, (0.2, 0.3))          # SVector of N values
eval_shape_function_gradient(el, (0.2, 0.3)) # SVector of ∇N tuples
eval_shape_function_jacobian(el, (0.2, 0.3)) # Jacobian matrix via ForwardDiff
```

## Integration Points

`IntegrationPoints(element)` returns the quadrature rule for the reference element:

```julia
ip = IntegrationPoints(LinearElement{2, 4})

ip.ξ  # SVector of ξ coordinates
ip.η  # SVector of η coordinates (nothing for 1D)
ip.ω  # SVector of weights
```

## Mesh

`Mesh` builds a structured mesh over a domain and assembles all connectivity data:

```julia
using FEMTools, DomainSets

Ω = (0.0..1.0) × (0.0..1.0)
element = ReferenceElement(LinearElement{2, 4})
mesh = Mesh(Ω, element, (16, 16))  # 16×16 elements
n2el = generate_node2element(mesh.el2n, mesh.nnodes)

mesh.coords   # node coordinates
mesh.el2n     # element-to-node connectivity  (N_local × N_elements)
n2el          # node-to-element adjacency
mesh.Γnodes   # boundary node ids
mesh.nnodes   # total node count
mesh.nels     # total element count
```

Sparsity pattern and mesh coloring for conflict-free parallel assembly:

```julia
K = generate_sparsity_pattern(mesh)
colors = color_mesh(mesh)
```

## Boundary Conditions

```julia
ΓD = DirichletBoundaryCondition(mesh.Γ, mesh.Γnodes, zeros(length(mesh.Γnodes)))

apply_bc!(rhs, ΓD)        # set constrained RHS entries
apply_bc!(A, ΓD)          # zero constrained rows, pin diagonal
apply_bc!(A, rhs, ΓD)     # symmetric elimination
```

## Solvers

Three matrix-free, backend-agnostic solvers are built on the mesh and element
primitives, each using a pseudo-transient dynamic-relaxation (DR) scheme:

| Solver | State type | Entry point |
|:-------|:-----------|:------------|
| Transient multi-phase heat diffusion | `ThermalDiffusionDR` | `solver!` |
| Lithostatic pressure | `LithostaticPressureDR` | `solver!` |
| Incompressible visco-elasto-plastic Stokes flow | `StokesDR` | `solve_stokes_dyrel!` |

Runnable scripts live under [`examples/`](examples/); see the
[documentation](https://albert-de-montserrat.github.io/FEMTools.jl/dev/) for the
physical models, solver-state fields, and worked examples.

## GPU backends

GPU support is loaded on demand. Install the relevant package and load it
alongside FEMTools to unlock the corresponding `TA` dispatch:

```julia
using CUDA, FEMTools
mesh = Mesh(CUDABackend(), Ω, element, (64, 64))  # arrays live on the GPU
```

| Package | Backend | Array type |
|:--------|:--------|:-----------|
| [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) | `CUDABackend()` | `CuArray` |
| [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) | `ROCBackend()` | `ROCArray` |
| [Metal.jl](https://github.com/JuliaGPU/Metal.jl) | `MetalBackend()` | `MtlArray` |
