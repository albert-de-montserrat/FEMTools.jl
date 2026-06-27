# Elements

Element types are lightweight tags parameterized by reference dimension and
number of vertices or nodes.

## Element Tags

```@docs
FEMTools.AbstractElement
FEMTools.AbstractLinearElement
FEMTools.AbstractQuadraticElement
LinearElement
QuadraticElement
CubicElement
ReferenceElement
order
Base.length(::ReferenceElement)
```

## Shape Functions

```@docs
FEMTools.AbstractShapeFunction
ShapeFunctions
eval_shape_function
eval_shape_function_gradient
eval_shape_function_jacobian
```

## Integration Points

```@docs
FEMTools.AbstractIntegrationPoints
IntegrationPoints
gauss_legendre_triangle
```
