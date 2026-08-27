# Field Containers

Solver states hold vector- and tensor-valued quantities as *struct-of-arrays*
containers: one array per component, bundled in a small immutable struct. A
velocity field is a `VectorField2D` holding `x` and `y` arrays rather than two
loose arrays passed side by side, and a deviatoric stress is a
`SymmetricTensor2D` holding its independent components in Voigt order.

The layout is chosen for kernels. Each component stays a plain contiguous array
that can be handed to a KernelAbstractions kernel unchanged, so nothing about
the container reaches GPU code; only the arrays it groups do. Gathering the
components of one sample into a value happens at the point of use, where
`A[i]` returns a stack-allocated `SVector`.

## Storage and indexing

A container's type parameter is the storage type of a single component. It may
be an array — the usual case, one entry per node or per integration point — or
a scalar, which describes a single sample rather than a field:

```julia
v = VectorField2D(backend, Float64, nnodes)   # x and y arrays of length nnodes
τ = SymmetricTensor2D(backend, Float64, nnodes)
```

Both support the read-only part of the array interface — `eltype`, `size`,
`length`, `axes`, `eachindex`, `firstindex`, `lastindex` — delegating to the
first component, so loops over samples read the way they would over an array.
Indexing selects a *sample*, not a component:

```julia
for i in eachindex(v)
    vi = v[i]           # SVector{2} of the components at sample i
    τi = τ[i]           # SVector{3}: (xx, yy, xy)
    τ[i] = 2 .* τi      # writes the components back
end
```

Components stored as `nq × nels` matrices of integration-point values are
indexed with both subscripts, `τ[q, iel]`, returning the same component vector.
Indices may be any `Integer`, which matters because GPU kernels commonly supply
`Int32`.

Voigt ordering is `(xx, yy, xy)` in two dimensions and
`(xx, yy, zz, yz, xz, xy)` in three. `SymmetricTensor2D` and
`SymmetricTensor3D` carry one further component, `II`, reserved for a derived
invariant. It is allocated with the others and left for the caller to fill;
component writes through `setindex!` do not touch it.

`Tuple` unpacks the independent components — dropping `II` — for assemblers
that take a plain `NTuple` of arrays, and `copyto!` copies a whole container
component-wise, which is how a stress history is advanced.

## A single tensor

`VoigtTensor2D` and `VoigtTensor3D` hold the components of *one* symmetric
tensor as scalars, in the same Voigt order. This is what a field yields at a
sample:

```julia
τi = VoigtTensor(τ, i)     # equivalently VoigtTensor(τ[i])
```

They convert to StaticArrays on demand — `SVector` for the Voigt vector,
`SMatrix` for the dense symmetric matrix — and support the arithmetic that goes
with that: addition and subtraction, negation, scaling by a number, the
matrix–vector product `τ * x`, the tensor product `τ1 * τ2`, and the solve
`τ \ x`.

Every result is immutable. A product with an `MVector` returns an `SVector`
rather than an `MVector`, because a mutable result is heap-allocated even where
it does not escape, which rules it out inside a GPU kernel.

Mixed component types promote, so `VoigtTensor2D(1, 0.0, 0.0)` gives a
`VoigtTensor2D{Float64}`; the parametric form `VoigtTensor2D{Float32}(1, 2, 3)`
converts to a chosen type instead. Building one from any other number of
components is an `ArgumentError`.

## Where they are used

[`StokesDR`](stokes.md) stores its velocity, pseudo-transient rate, momentum
residual and its snapshot, row-sum Jacobian estimate, and diagonal
preconditioner as vector fields, and its current and previous deviatoric stress
as symmetric tensor fields. Which pair it uses follows the solver state's
spatial dimension, so the same field names carry two or three velocity
components and three or six stress components. The 3-D sinking-block example
uses a bare `VectorField3D` for its velocity.

## Vector fields

```@docs
FEMTools.AbstractVectorField
FEMTools.VectorField2D
FEMTools.VectorField3D
FEMTools.VectorField
```

## Symmetric tensor fields

```@docs
FEMTools.AbstractTensor
FEMTools.AbstractSymmetricTensor
FEMTools.SymmetricTensor2D
FEMTools.SymmetricTensor3D
FEMTools.SymmetricTensor
```

## Single tensors

```@docs
FEMTools.AbstractVoigtTensor
FEMTools.VoigtTensor2D
FEMTools.VoigtTensor3D
FEMTools.VoigtTensor
```
