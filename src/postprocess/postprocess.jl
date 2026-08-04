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

"""
    update_old_stress_from_cells!(τ_old, post, el2n_v, nnodes_v)

Project cell-averaged stress diagnostics back to nodal old-stress arrays.
"""
function update_old_stress_from_cells!(τ_old, post, el2n_v, nnodes_v)
    τxx_nodes = zeros(eltype(τ_old[1]), nnodes_v)
    τyy_nodes = zeros(eltype(τ_old[2]), nnodes_v)
    τxy_nodes = zeros(eltype(τ_old[3]), nnodes_v)
    counts = zeros(Int, nnodes_v)

    for iel in axes(el2n_v, 2)
        for a in axes(el2n_v, 1)
            inode = el2n_v[a, iel]
            τxx_nodes[inode] += post.τxx[iel]
            τyy_nodes[inode] += post.τyy[iel]
            τxy_nodes[inode] += post.τxy[iel]
            counts[inode] += 1
        end
    end

    for inode in eachindex(counts)
        if counts[inode] > 0
            τxx_nodes[inode] /= counts[inode]
            τyy_nodes[inode] /= counts[inode]
            τxy_nodes[inode] /= counts[inode]
        end
    end

    copyto!(τ_old[1], τxx_nodes)
    copyto!(τ_old[2], τyy_nodes)
    copyto!(τ_old[3], τxy_nodes)
    return nothing
end

"""
    write_vtk(path, mesh; point_data=(;), cell_data=(;), title="FEMTools")

Write a legacy ASCII VTK unstructured-grid file for `Mesh` or `MixedMesh`.

High-order elements are linearized to their corner nodes. Scalar `point_data`
fields may have either one value per mesh coordinate node or one value per
written VTK point. Scalar `cell_data` fields must have one value per element.
"""
function write_vtk(path, mesh::Union{Mesh, MixedMesh}; point_data = (;), cell_data = (;), title = "FEMTools")
    topo = _vtk_topology(mesh)
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
                _vtk_write_scalar_field(io, name, _vtk_point_values(field, topo, name))
            end
        end

        if !isempty(cell_fields)
            println(io, "CELL_DATA $(topo.nels)")
            for (name, field) in cell_fields
                _vtk_write_scalar_field(io, name, _vtk_cell_values(field, topo, name))
            end
        end
    end
    return nothing
end

_vtk_mesh_arrays(mesh::Mesh{nDim}) where {nDim} =
    (Array(mesh.coords), Array(mesh.el2n), mesh.nels, Val(nDim))

_vtk_mesh_arrays(mesh::MixedMesh{nDim}) where {nDim} =
    (Array(mesh.coords), Array(mesh.el2nP), mesh.nels, Val(nDim))

function _vtk_topology(mesh)
    coords, el2n, nels, dim = _vtk_mesh_arrays(mesh)
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
    nlocal == 4 && return [1, 2, 3, 4]
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

function _vtk_cell_values(field, topo, name)
    values = vec(Array(field))
    length(values) == topo.nels && return values
    throw(DimensionMismatch(
        "cell_data[$(string(name))] has length $(length(values)); expected $(topo.nels)",
    ))
end

function _vtk_write_scalar_field(io, name, values)
    println(io, "SCALARS $(string(name)) float 1")
    println(io, "LOOKUP_TABLE default")
    for value in values
        println(io, value)
    end
end

"""
    write_stokes_vtk(vtk_path, mesh_stokes, coords_v, el2nP_cpu, DoFsP_cpu, P_cpu, vx_cpu, vy_cpu, post; title, cell_data)

Write pressure, velocity, strain-rate, and stress diagnostics to an ASCII VTK
unstructured-grid file using pressure triangle corners. Extra per-cell fields
can be supplied with `cell_data`.
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
)
    topo = _vtk_topology(mesh_stokes)
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
    vtk_V = hypot.(vtk_Vx, vtk_Vy)

    return write_vtk(
        vtk_path,
        mesh_stokes;
        point_data = (; P = vtk_P, Vx = vtk_Vx, Vy = vtk_Vy, V = vtk_V),
        cell_data = merge(cell_data, (;
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
        )),
        title,
    )
end
