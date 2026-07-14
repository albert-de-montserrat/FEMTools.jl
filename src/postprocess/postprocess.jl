"""
    compute_strain_rate_stress_postprocess(vx, vy, el2n_v, geo_v, τ_ip, element_v)

Compute element-averaged strain-rate and current deviatoric-stress diagnostics
from integration-point stresses.
"""
function compute_strain_rate_stress_postprocess(
    vx, vy,
    el2n_v,
    geo_v,
    τ_ip,
    element_v::ReferenceElement{TV},
) where {NV, FP, TV <: AbstractElement{2, NV, FP}}
    nels = size(el2n_v, 2)
    Nq = shape_function_values(element_v)

    εxx = zeros(FP, nels)
    εyy = zeros(FP, nels)
    εzz = zeros(FP, nels)
    εxy = zeros(FP, nels)
    εII = zeros(FP, nels)
    τxx = zeros(FP, nels)
    τyy = zeros(FP, nels)
    τzz = zeros(FP, nels)
    τxy = zeros(FP, nels)
    τII = zeros(FP, nels)

    for iel in 1:nels
        local_nodes = SVector{NV}(ntuple(i -> el2n_v[i, iel], Val(NV)))
        vxloc = SVector{NV}(ntuple(i -> vx[local_nodes[i]], Val(NV)))
        vyloc = SVector{NV}(ntuple(i -> vy[local_nodes[i]], Val(NV)))
        geo_el = geo_v[iel]
        volume = zero(FP)

        for q in eachindex(geo_el)
            ∂N∂x, dΩ = geo_el[q]

            ∇vx = ∂N∂x' * vxloc
            ∇vy = ∂N∂x' * vyloc

            εxx_q = ∇vx[1]
            εyy_q = ∇vy[2]
            εzz_q = zero(εxx_q)
            εxy_q = (∇vx[2] + ∇vy[1]) / 2

            tr = (εxx_q + εyy_q + εzz_q) / 3
            εxx_dev = εxx_q - tr
            εyy_dev = εyy_q - tr
            εzz_dev = εzz_q - tr

            τxx_q = τ_ip[1][q, iel]
            τyy_q = τ_ip[2][q, iel]
            τxy_q = τ_ip[3][q, iel]
            τzz_q = -(τxx_q + τyy_q)

            εII_q = sqrt((εxx_dev^2 + εyy_dev^2 + εzz_dev^2) / 2 + εxy_q^2)
            τII_q = sqrt((τxx_q^2 + τyy_q^2 + τzz_q^2) / 2 + τxy_q^2)

            εxx[iel] += εxx_q * dΩ
            εyy[iel] += εyy_q * dΩ
            εzz[iel] += εzz_q * dΩ
            εxy[iel] += εxy_q * dΩ
            εII[iel] += εII_q * dΩ
            τxx[iel] += τxx_q * dΩ
            τyy[iel] += τyy_q * dΩ
            τzz[iel] += τzz_q * dΩ
            τxy[iel] += τxy_q * dΩ
            τII[iel] += τII_q * dΩ
            volume += dΩ
        end

        εxx[iel] /= volume
        εyy[iel] /= volume
        εzz[iel] /= volume
        εxy[iel] /= volume
        εII[iel] /= volume
        τxx[iel] /= volume
        τyy[iel] /= volume
        τzz[iel] /= volume
        τxy[iel] /= volume
        τII[iel] /= volume
    end

    return (;
        εxx, εyy, εzz, εxy, εII,
        τxx, τyy, τzz, τxy,
        tauII = τII,
    )
end

@inline _phase_at_postprocess(phases::AbstractMatrix, _, i, iel) =
    Int(phases[size(phases, 1) == 1 ? 1 : i, iel])
@inline _phase_at_postprocess(phases, local_nodes, i, _) = Int(phases[local_nodes[i]])
@inline _phase_loc_postprocess(phases, local_nodes, iel, ::Val{N}) where {N} =
    SVector{N}(ntuple(i -> _phase_at_postprocess(phases, local_nodes, i, iel), Val(N)))

"""
    compute_strain_rate_stress_postprocess(vx, vy, el2n_v, geo_v, phases_v, τ_old, η, G, Δt, element_v)

Compute element-averaged strain-rate and viscoelastic deviatoric-stress
diagnostics from nodal old-stress fields.
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
    nels = size(el2n_v, 2)
    Nq = shape_function_values(element_v)

    εxx = zeros(FP, nels)
    εyy = zeros(FP, nels)
    εzz = zeros(FP, nels)
    εxy = zeros(FP, nels)
    εII = zeros(FP, nels)
    τxx = zeros(FP, nels)
    τyy = zeros(FP, nels)
    τzz = zeros(FP, nels)
    τxy = zeros(FP, nels)
    τII = zeros(FP, nels)

    for iel in 1:nels
        local_nodes = SVector{NV}(ntuple(i -> el2n_v[i, iel], Val(NV)))
        vxloc = SVector{NV}(ntuple(i -> vx[local_nodes[i]], Val(NV)))
        vyloc = SVector{NV}(ntuple(i -> vy[local_nodes[i]], Val(NV)))
        τxx_old_loc = SVector{NV}(ntuple(i -> τ_old[1][local_nodes[i]], Val(NV)))
        τyy_old_loc = SVector{NV}(ntuple(i -> τ_old[2][local_nodes[i]], Val(NV)))
        τxy_old_loc = SVector{NV}(ntuple(i -> τ_old[3][local_nodes[i]], Val(NV)))
        phase_loc = _phase_loc_postprocess(phases_v, local_nodes, iel, Val(NV))
        geo_el = geo_v[iel]
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
            εxx_dev = εxx_q - tr
            εyy_dev = εyy_q - tr
            εzz_dev = εzz_q - tr

            ηq = interp2ip_phase(Nv, η, phase_loc)
            invGq = interp2ip_phase(Nv, map(inv, G), phase_loc)
            ηeff_q = inv(inv(ηq) + invGq / Δt)
            inv_2Gdt = invGq / (2 * Δt)

            τxx_old_q = dot(Nv, τxx_old_loc)
            τyy_old_q = dot(Nv, τyy_old_loc)
            τxy_old_q = dot(Nv, τxy_old_loc)
            τzz_old_q = -(τxx_old_q + τyy_old_q)

            τxx_q = 2 * ηeff_q * (εxx_dev + τxx_old_q * inv_2Gdt)
            τyy_q = 2 * ηeff_q * (εyy_dev + τyy_old_q * inv_2Gdt)
            τzz_q = 2 * ηeff_q * (εzz_dev + τzz_old_q * inv_2Gdt)
            τxy_q = 2 * ηeff_q * (εxy_q + τxy_old_q * inv_2Gdt)

            εII_q = sqrt((εxx_dev^2 + εyy_dev^2 + εzz_dev^2) / 2 + εxy_q^2)
            τII_q = sqrt((τxx_q^2 + τyy_q^2 + τzz_q^2) / 2 + τxy_q^2)

            εxx[iel] += εxx_q * dΩ
            εyy[iel] += εyy_q * dΩ
            εzz[iel] += εzz_q * dΩ
            εxy[iel] += εxy_q * dΩ
            εII[iel] += εII_q * dΩ
            τxx[iel] += τxx_q * dΩ
            τyy[iel] += τyy_q * dΩ
            τzz[iel] += τzz_q * dΩ
            τxy[iel] += τxy_q * dΩ
            τII[iel] += τII_q * dΩ
            volume += dΩ
        end

        εxx[iel] /= volume
        εyy[iel] /= volume
        εzz[iel] /= volume
        εxy[iel] /= volume
        εII[iel] /= volume
        τxx[iel] /= volume
        τyy[iel] /= volume
        τzz[iel] /= volume
        τxy[iel] /= volume
        τII[iel] /= volume
    end

    return (;
        εxx, εyy, εzz, εxy, εII,
        τxx, τyy, τzz, τxy,
        tauII = τII,
    )
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

High-order elements are linearized to their corner nodes. Scalar or vector
`point_data` fields may have either one value per mesh coordinate node or one
value per written VTK point. Two-component vectors are padded with a zero
z-component. `cell_data` fields must have one value per element.
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

function _vtk_write_field(io, name, values)
    isempty(values) || first(values) isa Number || return _vtk_write_vector_field(io, name, values)
    return _vtk_write_scalar_field(io, name, values)
end

function _vtk_write_vector_field(io, name, values)
    println(io, "VECTORS $(string(name)) float")
    for value in values
        length(value) in (2, 3) || throw(DimensionMismatch(
            "vector field $(string(name)) must have two or three components"))
        println(io, length(value) == 2 ? "$(value[1]) $(value[2]) 0.0" :
                                        "$(value[1]) $(value[2]) $(value[3])")
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
