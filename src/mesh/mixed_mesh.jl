"""
    MixedMesh{nDim, O1, O2, ...}
    MixedMesh(element, elementP, coords, DoFs, el2n, DoFsP, el2nP)

Mixed finite-element mesh storing separate connectivities for two fields.

`O1` is the polynomial order of the primary field (e.g. velocity) and `O2` is
the order of the secondary field (e.g. pressure). Both fields share the same
element geometry (`coords`, `nels`) but have independent node numbering and
degree-of-freedom maps.

Fields:
- `coords`  : vertex coordinates (`AbstractVector{SVector{nDim, T}}`).
- `nels`    : number of elements.
- `DoFs`    : primary-field degree-of-freedom indices.
- `el2n`    : primary element-to-node connectivity (`N1 × nels`).
- `nnodes`  : number of primary-field nodes.
- `DoFsP`   : secondary-field degree-of-freedom indices.
- `el2nP`   : secondary element-to-node connectivity (`N2 × nels`).
- `nnodesP` : number of secondary-field nodes.
"""
struct MixedMesh{nDim, O1, O2, T1, T2, T3, T4, T5} <: AbstractMesh
    coords::T1  # vertex coordinates
    nels::Int   # number of elements
    # Field 1: Velocity
    DoFs::T2    # degrees of freedom
    el2n::T3    # element-to-node connectivity
    nnodes::Int # number of nodes
    # Field 2: Pressure
    DoFsP::T4    # degrees of freedom
    el2nP::T5    # element-to-node connectivity
    nnodesP::Int # number of nodes

    function MixedMesh{nDim, O1, O2, T1, T2, T3, T4, T5}(
        coords::T1,
        DoFs::T2,
        el2n::T3,
        nnodes::Int,
        nels::Int,
        DoFsP::T4,
        el2nP::T5,
        nnodesP::Int,
    ) where {nDim, O1, O2, T1, T2, T3, T4, T5}
        size(el2n, 2) == nels || throw(ArgumentError("velocity connectivity has the wrong number of elements"))
        size(el2nP, 2) == nels || throw(ArgumentError("pressure connectivity has the wrong number of elements"))
        return new{nDim, O1, O2, T1, T2, T3, T4, T5}(
            coords, nels,
            DoFs, el2n, nnodes,
            DoFsP, el2nP, nnodesP,
        )
    end
end


function MixedMesh(
    element::ReferenceElement,
    elementP::ReferenceElement,
    coords::AbstractVector{<:SVector{nDim}},
    DoFs,
    el2n,
    DoFsP,
    el2nP,
) where {nDim}
    return MixedMesh{
        nDim,
        order(element),
        order(elementP),
        typeof(coords),
        typeof(DoFs),
        typeof(el2n),
        typeof(DoFsP),
        typeof(el2nP),
    }(
        coords,
        DoFs,
        el2n,
        length(coords),
        size(el2n, 2),
        DoFsP,
        el2nP,
        prod(size(el2nP)),
    )
end

# ---------------------------------------------------------------------------
# Mesh generation
# ---------------------------------------------------------------------------

"""
    build_discontinuous_linear_mesh(coords, el2n) -> (p_el2n, p_el2dof, p_dof_coords)

Build the linear triangle topology and element-to-DoF map for discontinuous
linear pressure elements.

`el2n` may be either T3 or T6 triangle connectivity. The returned `p_el2n`
uses rows 1:3, i.e. the corner nodes of each triangle. The returned
`p_el2dof` gives each element its own three pressure DoFs:
`p_el2dof[:, iel] == 3(iel - 1) .+ (1:3)`.

Returns `(p_el2n, p_el2dof, p_dof_coords)`, where `p_dof_coords` duplicates
corner coordinates per element so a discontinuous nodal pressure field can be
plotted or initialized directly on pressure DoFs.
"""
function build_discontinuous_linear_mesh(coords, el2n::AbstractMatrix{<:Integer})
    size(el2n, 1) >= 3 || throw(ArgumentError("triangle connectivity needs at least 3 local nodes"))

    nels      = size(el2n, 2)
    p_el2n    = Matrix{Int32}(el2n[1:3, :])
    p_el2dof  = Matrix{Int32}(undef, 3, nels)
    p_dof_coords = Vector{eltype(coords)}(undef, 3 * nels)

    for iel in 1:nels
        base = 3 * (iel - 1)
        for a in 1:3
            dof = base + a
            p_el2dof[a, iel] = Int32(dof)
            p_dof_coords[dof] = coords[p_el2n[a, iel]]
        end
    end

    return p_el2n, p_el2dof, p_dof_coords
end