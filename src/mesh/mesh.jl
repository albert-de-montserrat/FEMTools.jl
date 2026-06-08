"""
    AbstractMesh

Abstract supertype for mesh containers.
"""
abstract type AbstractMesh end

"""
    Mesh(Ω, nels)

Construct a one-dimensional quadratic mesh over the interval domain `Ω` with
`nels` elements.

The constructor derives the boundary `Γ`, coordinates, degrees of freedom,
element-to-node connectivity, node-to-element adjacency, boundary nodes,
boundary elements, and stores the node/element counts.
"""
struct Mesh{nDim, O, D, B, T1, T2, T3, T4, T5, T6} <: AbstractMesh
    Ω::D        # model domain
    Γ::B        # model boundary
    coords::T1  # vertex coordinates
    DoFs::T2    # degrees of freedom
    el2n::T3    # element-to-node connectivity
    n2el::T4    # node-to-elements connectivity
    Γnodes::T5  # boundary nodes
    Γels::T6    # boundary elements
    nnodes::Int # number of nodes
    nels::Int   # number of elements

    function Mesh(Ω::ClosedInterval, nels::Integer)
        element = ReferenceElement(QuadraticElement{1, 3})
        Γ = boundary(Ω)
        coords = generate_coordinates(element, Ω, nels)
        DoFs = generate_dofs(element, length(coords))
        el2n = generate_element2node(element, nels)
        n2el = generate_node2element(el2n, length(coords))
        Γnodes = DoFs[in.(coords, Γ)]
        Γels = generate_boundary_elements(Γnodes, n2el)
        nnodes = length(coords)

        return new{
            1,
            order(element),
            typeof(Ω),
            typeof(Γ),
            typeof(coords),
            typeof(DoFs),
            typeof(el2n),
            typeof(n2el),
            typeof(Γnodes),
            typeof(Γels),
        }(Ω, Γ, coords, DoFs, el2n, n2el, Γnodes, Γels, nnodes, Int(nels))
    end
end

"""
    generate_coordinates(element::ReferenceElement{1, 2}, Ω, nels)

Generate coordinates for a linear one-dimensional mesh over interval `Ω`.
"""
function generate_coordinates(::ReferenceElement{1, 2}, Ω::ClosedInterval, nels)
    domain = LinRange(leftendpoint(Ω), rightendpoint(Ω), nels + 1)
    return collect(domain)
end

"""
    generate_coordinates(element::ReferenceElement{1, 3}, Ω, nels)

Generate coordinates for a quadratic one-dimensional mesh over interval `Ω`.

Vertex nodes occupy odd indices and midpoint nodes occupy even indices.
"""
function generate_coordinates(::ReferenceElement{1, 3}, Ω::ClosedInterval, nels)
    domain = LinRange(leftendpoint(Ω), rightendpoint(Ω), nels + 1)
    npoints = 2 * length(domain) - 1
    coords = zeros(npoints)
    coords[1:2:npoints] .= domain
    for i in 2:2:length(coords)
        coords[i] = (coords[i - 1] + coords[i + 1]) / 2
    end
    return coords
end

"""
    generate_dofs(element::ReferenceElement{1}, npoints)

Generate one degree of freedom per mesh point for a one-dimensional element.
"""
generate_dofs(::ReferenceElement{1}, npoints) = [Int32(i) for i in 1:npoints]

generete_coordinates(args...) = generate_coordinates(args...)
generete_DoFs(args...) = generate_dofs(args...)
