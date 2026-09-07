# Element-averaged diagnostics, in the order the accumulator carries them.
const _DIAGNOSTIC_FIELDS = (:εxx, :εyy, :εzz, :εxy, :εII, :τxx, :τyy, :τzz, :τxy, :tauII)

"""
    _strain_rate_stress_diagnostics(vx, vy, el2n_v, geo_v, element_v, element_stress)

Volume-average strain-rate and deviatoric-stress diagnostics over each element.

`element_stress(iel, local_nodes)` returns the stress law for one element as a
callable `(q, Nv, ε_dev) -> (τxx, τyy, τzz, τxy)`, where `ε_dev` holds the
deviatoric strain rate `(εxx, εyy, εzz, εxy)` at quadrature point `q`. Gathering
the element's nodal data in `element_stress` keeps it out of the quadrature
loop.

Each field is integrated against `dΩ` and divided by the element volume.
"""
function _strain_rate_stress_diagnostics(
    vx, vy,
    el2n_v,
    geo_v,
    element_v::ReferenceElement{TV},
    element_stress,
) where {NV, FP, TV <: AbstractElement{2, NV, FP}}
    nels = size(el2n_v, 2)
    Nq = shape_function_values(element_v)
    fields = ntuple(_ -> zeros(FP, nels), length(_DIAGNOSTIC_FIELDS))

    for iel in 1:nels
        local_nodes = local_nodes_of(el2n_v, iel, Val(NV))
        vxloc = _gather_local(vx, local_nodes, Val(NV))
        vyloc = _gather_local(vy, local_nodes, Val(NV))
        stress_at_ip = element_stress(iel, local_nodes)
        geo_el = geo_v[iel]

        totals = ntuple(_ -> zero(FP), length(_DIAGNOSTIC_FIELDS))
        volume = zero(FP)

        for q in eachindex(geo_el)
            ∂N∂x, dΩ = geo_el[q]
            Nv = Nq[q]

            ∇vx = ∂N∂x' * vxloc
            ∇vy = ∂N∂x' * vyloc

            εxx_q = ∇vx[1]
            εyy_q = ∇vy[2]
            εzz_q = zero(εxx_q)
            εxy_q = (∇vx[2] + ∇vy[1]) / 2

            tr = (εxx_q + εyy_q + εzz_q) / 3
            ε_dev = (εxx_q - tr, εyy_q - tr, εzz_q - tr, εxy_q)

            τxx_q, τyy_q, τzz_q, τxy_q = stress_at_ip(q, Nv, ε_dev)

            εII_q = sqrt((ε_dev[1]^2 + ε_dev[2]^2 + ε_dev[3]^2) / 2 + εxy_q^2)
            τII_q = sqrt((τxx_q^2 + τyy_q^2 + τzz_q^2) / 2 + τxy_q^2)

            contribution = (
                εxx_q, εyy_q, εzz_q, εxy_q, εII_q,
                τxx_q, τyy_q, τzz_q, τxy_q, τII_q,
            )
            totals = map((total, value) -> total + value * dΩ, totals, contribution)
            volume += dΩ
        end

        for (field, total) in zip(fields, totals)
            field[iel] = total / volume
        end
    end

    return NamedTuple{_DIAGNOSTIC_FIELDS}(fields)
end

"""
    compute_strain_rate_stress_postprocess(vx, vy, el2n_v, geo_v, τ_ip, element_v)

Compute element-averaged strain-rate and current deviatoric-stress diagnostics
from integration-point stresses.

`τ_ip` holds `(τxx, τyy, τxy)` as `nq × nels` matrices; the out-of-plane
component follows from the deviatoric constraint `τzz = −(τxx + τyy)`.
"""
function compute_strain_rate_stress_postprocess(
    vx, vy,
    el2n_v,
    geo_v,
    τ_ip,
    element_v::ReferenceElement{TV},
) where {NV, FP, TV <: AbstractElement{2, NV, FP}}
    function element_stress(iel, _)
        return function (q, _, _)
            τxx_q = τ_ip[1][q, iel]
            τyy_q = τ_ip[2][q, iel]
            return τxx_q, τyy_q, -(τxx_q + τyy_q), τ_ip[3][q, iel]
        end
    end
    return _strain_rate_stress_diagnostics(vx, vy, el2n_v, geo_v, element_v, element_stress)
end

"""
    compute_strain_rate_stress_postprocess(vx, vy, el2n_v, geo_v, phases_v, τ_old, η, G, Δt, element_v)

Compute element-averaged strain-rate and viscoelastic deviatoric-stress
diagnostics from nodal old-stress fields.

Stress follows the Maxwell viscoelastic law
`τ = 2 ηeff (ε_dev + τ_old / (2 G Δt))` with `ηeff = (1/η + 1/(G Δt))⁻¹`.
Compliance `1/G` is interpolated rather than `G`, so the purely viscous limit
`G = Inf` stays finite where quadratic shape functions are negative.
"""
function compute_strain_rate_stress_postprocess(
    vx, vy,
    el2n_v,
    geo_v,
    phases_v,
    τ_old,
    η, G, Δt,
    element_v::ReferenceElement{TV},
) where {NV, FP, TV <: AbstractElement{2, NV, FP}}
    invG = map(inv, G)
    function element_stress(iel, local_nodes)
        τxx_old_loc = _gather_local(τ_old[1], local_nodes, Val(NV))
        τyy_old_loc = _gather_local(τ_old[2], local_nodes, Val(NV))
        τxy_old_loc = _gather_local(τ_old[3], local_nodes, Val(NV))
        phase_loc = _gather_phase(phases_v, local_nodes, iel, Val(NV))
        return function (_, Nv, ε_dev)
            ηq = interp2ip_phase(Nv, η, phase_loc)
            invGq = interp2ip_phase(Nv, invG, phase_loc)
            ηeff_q = inv(inv(ηq) + invGq / Δt)
            inv_2Gdt = invGq / (2 * Δt)

            τxx_old_q = dot(Nv, τxx_old_loc)
            τyy_old_q = dot(Nv, τyy_old_loc)
            τxy_old_q = dot(Nv, τxy_old_loc)
            τzz_old_q = -(τxx_old_q + τyy_old_q)

            return (
                2 * ηeff_q * (ε_dev[1] + τxx_old_q * inv_2Gdt),
                2 * ηeff_q * (ε_dev[2] + τyy_old_q * inv_2Gdt),
                2 * ηeff_q * (ε_dev[3] + τzz_old_q * inv_2Gdt),
                2 * ηeff_q * (ε_dev[4] + τxy_old_q * inv_2Gdt),
            )
        end
    end
    return _strain_rate_stress_diagnostics(vx, vy, el2n_v, geo_v, element_v, element_stress)
end

# Element-averaged three-dimensional diagnostics, in accumulator order.
const _DIAGNOSTIC_FIELDS_3D = (
    :εxx, :εyy, :εzz, :εxy, :εxz, :εyz, :εII,
    :τxx, :τyy, :τzz, :τxy, :τxz, :τyz, :tauII,
)

"""
    compute_strain_rate_stress_postprocess(v, el2n_v, geo_v, τ_ip, element_v)

Compute element-averaged strain-rate and current deviatoric-stress diagnostics
in three dimensions from integration-point stresses.

`v` holds one nodal velocity array per direction. `τ_ip` holds
`(τxx, τyy, τzz, τxy, τxz, τyz)` as `nq × nels` matrices, so a stress that has
been through a plastic return is carried through unchanged.

Normal strain rates are reported in full while the invariant `εII` is formed
from the deviatoric part, matching the plane-strain method.
"""
function compute_strain_rate_stress_postprocess(
    v::NTuple{3},
    el2n_v,
    geo_v,
    τ_ip::NTuple{6},
    element_v::ReferenceElement{TV},
) where {NV, FP, TV <: AbstractElement{3, NV, FP}}
    nels = size(el2n_v, 2)
    Nq = shape_function_values(element_v)
    fields = ntuple(_ -> zeros(FP, nels), length(_DIAGNOSTIC_FIELDS_3D))

    for iel in 1:nels
        local_nodes = local_nodes_of(el2n_v, iel, Val(NV))
        vloc = ntuple(i -> _gather_local(v[i], local_nodes, Val(NV)), Val(3))
        geo_el = geo_v[iel]

        totals = ntuple(_ -> zero(FP), length(_DIAGNOSTIC_FIELDS_3D))
        volume = zero(FP)

        for q in eachindex(geo_el)
            ∂N∂x, dΩ = geo_el[q]
            ∇v = ntuple(i -> ∂N∂x' * vloc[i], Val(3))

            ε = (
                ∇v[1][1], ∇v[2][2], ∇v[3][3],
                (∇v[1][2] + ∇v[2][1]) / 2,
                (∇v[1][3] + ∇v[3][1]) / 2,
                (∇v[2][3] + ∇v[3][2]) / 2,
            )
            tr = (ε[1] + ε[2] + ε[3]) / 3
            ε_dev = (ε[1] - tr, ε[2] - tr, ε[3] - tr, ε[4], ε[5], ε[6])
            τ = ntuple(c -> τ_ip[c][q, iel], Val(6))

            contribution = (ε..., second_invariant(ε_dev), τ..., second_invariant(τ))
            totals = map((total, value) -> total + value * dΩ, totals, contribution)
            volume += dΩ
        end

        for (field, total) in zip(fields, totals)
            field[iel] = total / volume
        end
    end

    return NamedTuple{_DIAGNOSTIC_FIELDS_3D}(fields)
end

# Cell-averaged stress components of a diagnostics NamedTuple, in the component
# order the solver's stress-history tuples use.
_cell_stress_components(post, ::Val{3}) = (post.τxx, post.τyy, post.τxy)
_cell_stress_components(post, ::Val{6}) =
    (post.τxx, post.τyy, post.τzz, post.τxy, post.τxz, post.τyz)

"""
    update_old_stress_from_cells!(τ_old, post, el2n_v, nnodes_v)

Project cell-averaged stress diagnostics back to nodal old-stress arrays.

`τ_old` holds three components in plane strain and six in three dimensions;
the matching components are taken from `post`.
"""
function update_old_stress_from_cells!(τ_old::NTuple{Nτ}, post, el2n_v, nnodes_v) where {Nτ}
    cells  = _cell_stress_components(post, Val(Nτ))
    nodal  = ntuple(c -> zeros(eltype(τ_old[c]), nnodes_v), Val(Nτ))
    counts = zeros(Int, nnodes_v)

    for iel in axes(el2n_v, 2)
        for a in axes(el2n_v, 1)
            inode = el2n_v[a, iel]
            ntuple(Val(Nτ)) do c
                nodal[c][inode] += cells[c][iel]
                nothing
            end
            counts[inode] += 1
        end
    end

    for inode in eachindex(counts)
        counts[inode] > 0 || continue
        ntuple(Val(Nτ)) do c
            nodal[c][inode] /= counts[inode]
            nothing
        end
    end

    foreach(copyto!, τ_old, nodal)
    return nothing
end

"""
    write_vtk(path, mesh; point_data=(;), cell_data=(;), title="FEMTools")

Write a legacy ASCII VTK unstructured-grid file for `Mesh` or `MixedMesh`.
The optional `coords` argument replaces the mesh coordinates in the written
geometry while preserving the mesh connectivity and field indexing.

High-order elements are linearized to their corner nodes. Scalar `point_data`
fields may have either one value per mesh coordinate node or one value per
written VTK point. Scalar `cell_data` fields must have one value per element.

A field given as a tuple of component arrays is written as a VTK vector or
tensor rather than as separate scalars, so a viewer can glyph it directly: two
or three components make a vector, six a symmetric tensor in the order
`(xx, yy, zz, xy, xz, yz)`, and nine a full row-major tensor.
"""
function write_vtk(path, mesh::Union{Mesh, MixedMesh}; point_data = (;), cell_data = (;), title = "FEMTools", coords = nothing)
    topo = _vtk_topology(mesh; coords_override = coords)
    point_fields = collect(pairs(point_data))
    cell_fields = collect(pairs(cell_data))

    open(path, "w") do io
        println(io, "# vtk DataFile Version 3.0")
        println(io, title)
        println(io, "ASCII")
        println(io, "DATASET UNSTRUCTURED_GRID")

        println(io, "POINTS $(length(topo.nodes)) float")
        for old_i in topo.nodes
            _vtk_print_point(io, topo.coords[old_i])
        end

        ncorner = size(topo.el2n, 1)
        println(io, "CELLS $(topo.nels) $((ncorner + 1) * topo.nels)")
        for iel in 1:topo.nels
            print(io, ncorner)
            for a in 1:ncorner
                print(io, " ", topo.node_map[topo.el2n[a, iel]] - 1)
            end
            println(io)
        end

        println(io, "CELL_TYPES $(topo.nels)")
        for _ in 1:topo.nels
            println(io, topo.cell_type)
        end

        if !isempty(point_fields)
            println(io, "POINT_DATA $(length(topo.nodes))")
            for (name, field) in point_fields
                _vtk_write_field(io, name, _vtk_point_values(field, topo, name))
            end
        end

        if !isempty(cell_fields)
            println(io, "CELL_DATA $(topo.nels)")
            for (name, field) in cell_fields
                _vtk_write_field(io, name, _vtk_cell_values(field, topo, name))
            end
        end
    end
    return nothing
end

_vtk_mesh_arrays(mesh::Mesh{nDim}) where {nDim} =
    (Array(mesh.coords), Array(mesh.el2n), mesh.nels, Val(nDim))

_vtk_mesh_arrays(mesh::MixedMesh{2}) =
    (Array(mesh.coords), Array(mesh.el2nP), mesh.nels, Val(2))

# The 3-D pressure connectivity identifies modal interpolation points, not a
# geometric tetrahedron. Write the velocity Hex27 topology instead.
_vtk_mesh_arrays(mesh::MixedMesh{3}) =
    (Array(mesh.coords), Array(mesh.el2n), mesh.nels, Val(3))

function _vtk_topology(mesh; coords_override = nothing)
    mesh_coords, el2n, nels, dim = _vtk_mesh_arrays(mesh)
    coords = coords_override === nothing ? mesh_coords : coords_override
    length(coords) == length(mesh_coords) || throw(DimensionMismatch(
        "coords has length $(length(coords)); expected $(length(mesh_coords))",
    ))
    corner_rows = _vtk_corner_rows(dim, size(el2n, 1))
    corner_el2n = Matrix{Int}(el2n[corner_rows, :])
    nodes = Int.(sort!(unique(vec(corner_el2n))))
    node_map = zeros(Int, length(coords))

    for (new_i, old_i) in enumerate(nodes)
        checkbounds(coords, old_i)
        node_map[old_i] = new_i
    end

    return (;
        coords,
        el2n = corner_el2n,
        nodes,
        node_map,
        nels,
        cell_type = _vtk_cell_type(dim, length(corner_rows)),
    )
end

function _vtk_corner_rows(::Val{1}, nlocal)
    nlocal == 2 && return [1, 2]
    nlocal == 3 && return [1, 3]
    throw(ArgumentError("cannot write VTK line cells with $nlocal local nodes"))
end

function _vtk_corner_rows(::Val{2}, nlocal)
    nlocal in (3, 6, 7) && return [1, 2, 3]
    nlocal in (4, 8, 9) && return [1, 2, 3, 4]
    throw(ArgumentError("cannot write VTK 2D cells with $nlocal local nodes"))
end

function _vtk_corner_rows(::Val{3}, nlocal)
    nlocal in (4, 10, 11) && return [1, 2, 3, 4]
    nlocal in (8, 27) && return [1, 2, 3, 4, 5, 6, 7, 8]
    throw(ArgumentError("cannot write VTK 3D cells with $nlocal local nodes"))
end

_vtk_cell_type(::Val{1}, ::Int) = 3
_vtk_cell_type(::Val{2}, ncorner::Int) = ncorner == 3 ? 5 : 9
_vtk_cell_type(::Val{3}, ncorner::Int) = ncorner == 4 ? 10 : 12

function _vtk_print_point(io, c)
    if c isa Number
        println(io, "$c 0.0 0.0")
    elseif length(c) == 1
        println(io, "$(c[1]) 0.0 0.0")
    elseif length(c) == 2
        println(io, "$(c[1]) $(c[2]) 0.0")
    else
        println(io, "$(c[1]) $(c[2]) $(c[3])")
    end
end

_vtk_point_values(field::Tuple, topo, name) =
    map(component -> _vtk_point_values(component, topo, name), field)

function _vtk_point_values(field, topo, name)
    values = vec(Array(field))
    if length(values) == length(topo.nodes)
        return values
    elseif length(values) == length(topo.coords)
        return values[topo.nodes]
    end
    throw(DimensionMismatch(
        "point_data[$(string(name))] has length $(length(values)); expected $(length(topo.nodes)) or $(length(topo.coords))",
    ))
end

_vtk_cell_values(field::Tuple, topo, name) =
    map(component -> _vtk_cell_values(component, topo, name), field)

function _vtk_cell_values(field, topo, name)
    values = vec(Array(field))
    length(values) == topo.nels && return values
    throw(DimensionMismatch(
        "cell_data[$(string(name))] has length $(length(values)); expected $(topo.nels)",
    ))
end

_vtk_write_field(io, name, values) = _vtk_write_scalar_field(io, name, values)

function _vtk_write_field(io, name, components::Tuple)
    ncomp = length(components)
    ncomp in (2, 3) && return _vtk_write_vector_field(io, name, components)
    ncomp in (6, 9) && return _vtk_write_tensor_field(io, name, components)
    throw(ArgumentError(
        "field $(string(name)) has $ncomp components; a tuple field must hold 2 or 3 " *
        "(vector), 6 (symmetric tensor), or 9 (full tensor) of them",
    ))
end

function _vtk_write_scalar_field(io, name, values)
    println(io, "SCALARS $(string(name)) float 1")
    println(io, "LOOKUP_TABLE default")
    for value in values
        println(io, value)
    end
end

function _vtk_write_vector_field(io, name, components::Tuple)
    println(io, "VECTORS $(string(name)) float")
    # The format has no two-component vector: a plane field pads the third slot.
    pad = zero(eltype(first(components)))
    for i in eachindex(components...)
        third = length(components) == 3 ? components[3][i] : pad
        println(io, components[1][i], " ", components[2][i], " ", third)
    end
end

# Row-major 3 × 3 positions of the components of a symmetric tensor given in the
# order (xx, yy, zz, xy, xz, yz), which is the order the stress and strain-rate
# diagnostics carry.
const _VTK_SYMMETRIC_TENSOR_LAYOUT = (1, 4, 5, 4, 2, 6, 5, 6, 3)

function _vtk_write_tensor_field(io, name, components::Tuple)
    # A legacy VTK tensor is always a full 3 × 3, so a symmetric field is
    # mirrored into the lower triangle as it is written.
    layout = length(components) == 6 ? _VTK_SYMMETRIC_TENSOR_LAYOUT : ntuple(identity, 9)
    println(io, "TENSORS $(string(name)) float")
    for i in eachindex(components...)
        for row in 1:3
            entries = ntuple(c -> components[layout[3 * (row - 1) + c]][i], 3)
            println(io, entries[1], " ", entries[2], " ", entries[3])
        end
    end
end

"""
    write_stokes_vtk(vtk_path, mesh_stokes, coords_v, el2nP_cpu, DoFsP_cpu,
                     P_cpu, vx_cpu, vy_cpu, post; title, cell_data, Q_cpu)

Write pressure, velocity, strain-rate, and stress diagnostics to an ASCII VTK
unstructured-grid file using pressure triangle corners. The 2-D velocity is
written both as component scalars (`Vx`, `Vy`) and as the vector field `V`, so
viewers can glyph it directly. An optional pressure-node source array `Q_cpu`
is averaged to cells and written as `Q`; extra per-cell fields can be supplied
with `cell_data`.
"""
function write_stokes_vtk(
    vtk_path,
    mesh_stokes,
    coords_v,
    el2nP_cpu,
    DoFsP_cpu,
    P_cpu,
    vx_cpu,
    vy_cpu,
    post;
    title = "FEMTools Stokes 2D",
    cell_data = (;),
    Q_cpu = nothing,
)
    topo = _vtk_topology(mesh_stokes; coords_override = coords_v)
    NP = size(el2nP_cpu, 1)

    vtk_P = zeros(eltype(P_cpu), length(topo.nodes))
    vtk_P_count = zeros(Int, length(topo.nodes))
    for iel in 1:mesh_stokes.nels
        for a in 1:NP
            inode = topo.node_map[el2nP_cpu[a, iel]]
            vtk_P[inode] += P_cpu[DoFsP_cpu[a, iel]]
            vtk_P_count[inode] += 1
        end
    end
    @. vtk_P /= vtk_P_count

    vtk_Vx = [vx_cpu[old_i] for old_i in topo.nodes]
    vtk_Vy = [vy_cpu[old_i] for old_i in topo.nodes]
    vtk_cell_data = merge(cell_data, (;
        strain_xx = post.εxx,
        strain_yy = post.εyy,
        strain_zz = post.εzz,
        strain_xy = post.εxy,
        strain_II = post.εII,
        tau_xx = post.τxx,
        tau_yy = post.τyy,
        tau_zz = post.τzz,
        tau_xy = post.τxy,
        tau_II = post.tauII,
    ))
    if Q_cpu !== nothing
        vtk_cell_data = merge(vtk_cell_data, (Q = [
            sum(Q_cpu[DoFsP_cpu[a, iel]] for a in axes(DoFsP_cpu, 1)) /
            size(DoFsP_cpu, 1) for iel in axes(DoFsP_cpu, 2)
        ],))
    end

    return write_vtk(
        vtk_path,
        mesh_stokes;
        point_data = (; P = vtk_P, Vx = vtk_Vx, Vy = vtk_Vy, V = (vtk_Vx, vtk_Vy)),
        cell_data = vtk_cell_data,
        coords = coords_v,
        title,
    )
end

"""
    write_stokes_vtk(vtk_path, mesh_stokes, coords_v, el2nP_cpu, DoFsP_cpu,
                     P_cpu, v_cpu::NTuple{3}, post; kwargs...)

Write a three-dimensional Stokes solution to a legacy VTK file. `v_cpu` holds
one nodal velocity array per direction, and `post` the diagnostics returned by
the three-dimensional `compute_strain_rate_stress_postprocess`.

Velocity is written as a VTK vector and the strain-rate and deviatoric-stress
diagnostics as VTK tensors, so a viewer can glyph them or take their principal
values without first recombining components. The second invariants are kept as
the separate scalars `strain_II` and `tau_II`.

The point elevation is written as the scalar `z`, in the same units as the
written coordinates, so a viewer can colour or warp a surface by it.

Pressure is written as cell data. For tetrahedra it is evaluated at the centroid
by averaging the four P1 values; for Hex27 it uses the cell-center value.
"""
function write_stokes_vtk(
    vtk_path, mesh_stokes, coords_v, el2nP_cpu, DoFsP_cpu, P_cpu,
    v_cpu::NTuple{3}, post;
    title = "FEMTools Stokes 3D",
    cell_data = (;),
)
    topo = _vtk_topology(mesh_stokes; coords_override = coords_v)
    vtk_V = ntuple(c -> [v_cpu[c][old_i] for old_i in topo.nodes], 3)
    cell_P = if size(mesh_stokes.el2n, 1) in (10, 11)
        [sum(P_cpu[DoFsP_cpu[a, iel]] for a in 1:4) / 4 for iel in 1:mesh_stokes.nels]
    else
        [P_cpu[DoFsP_cpu[1, iel]] for iel in 1:mesh_stokes.nels]
    end

    return write_vtk(
        vtk_path,
        mesh_stokes;
        point_data = (; velocity = vtk_V, z = [topo.coords[i][3] for i in topo.nodes]),
        cell_data = merge(cell_data, (;
            P = cell_P,
            strain = (post.εxx, post.εyy, post.εzz, post.εxy, post.εxz, post.εyz),
            strain_II = post.εII,
            tau = (post.τxx, post.τyy, post.τzz, post.τxy, post.τxz, post.τyz),
            tau_II = post.tauII,
        )),
        coords = coords_v,
        title,
    )
end
