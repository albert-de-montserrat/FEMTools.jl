# FEMTools.jl

Small finite-element utilities for reference elements, shape functions, and
integration points.

## Elements

Element types are lightweight tags parameterized by reference dimension and
number of vertices:

```julia
import FEMTools

line = FEMTools.ReferenceElement(FEMTools.LinearElement{1, 2})
triangle = FEMTools.ReferenceElement(FEMTools.LinearElement{2, 3})
quad = FEMTools.ReferenceElement(FEMTools.LinearElement{2, 4})
quadratic_line = FEMTools.ReferenceElement(FEMTools.QuadraticElement{1, 3})
quadratic_triangle = FEMTools.ReferenceElement(FEMTools.QuadraticElement{2, 6})
quadratic_quad = FEMTools.ReferenceElement(FEMTools.QuadraticElement{2, 9})
```

## Shape Functions

`ShapeFunctions(element)` returns a container with shape functions `N` and
reference-coordinate gradients `∇N`, both ordered by local node number.

```julia
sf = FEMTools.ShapeFunctions(FEMTools.QuadraticElement{2, 6})

sf.N[4](0.5, 0.0)    # 1.0
sf.∇N[4](0.2, 0.3)   # (1.2, -0.8)
```

For a `ReferenceElement`, evaluate all shape functions at once:

```julia
el = FEMTools.ReferenceElement(FEMTools.LinearElement{2, 3})

FEMTools.eval_shape_function(el, (0.2, 0.3))
FEMTools.eval_shape_function_gradient(el, (0.2, 0.3))
FEMTools.eval_shape_function_jacobian(el, (0.2, 0.3))
```

## Integration Points

`IntegrationPoints(element)` returns reference-coordinate integration points:

```julia
ip = FEMTools.IntegrationPoints(FEMTools.LinearElement{2, 4})

ip.ξ
ip.η
ip.ζ
ip.ω
```
