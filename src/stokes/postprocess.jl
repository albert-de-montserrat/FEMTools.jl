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
) where {NV, TV <: AbstractElement{2, NV}}
    nels = size(el2n_v, 2)
    Nq = shape_function_values(element_v)

    εxx = zeros(Float64, nels)
    εyy = zeros(Float64, nels)
    εzz = zeros(Float64, nels)
    εxy = zeros(Float64, nels)
    εII = zeros(Float64, nels)
    τxx = zeros(Float64, nels)
    τyy = zeros(Float64, nels)
    τzz = zeros(Float64, nels)
    τxy = zeros(Float64, nels)
    τII = zeros(Float64, nels)

    for iel in 1:nels
        local_nodes = SVector{NV}(ntuple(i -> el2n_v[i, iel], Val(NV)))
        vxloc = SVector{NV}(ntuple(i -> vx[local_nodes[i]], Val(NV)))
        vyloc = SVector{NV}(ntuple(i -> vy[local_nodes[i]], Val(NV)))
        geo_el = geo_v[iel]
        volume = 0.0

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
        τxx, τyy, τzz, τxy, τII,
        tauII = τII,
    )
end

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
) where {NV, TV <: AbstractElement{2, NV}}
    nels = size(el2n_v, 2)
    Nq = shape_function_values(element_v)

    εxx = zeros(Float64, nels)
    εyy = zeros(Float64, nels)
    εzz = zeros(Float64, nels)
    εxy = zeros(Float64, nels)
    εII = zeros(Float64, nels)
    τxx = zeros(Float64, nels)
    τyy = zeros(Float64, nels)
    τzz = zeros(Float64, nels)
    τxy = zeros(Float64, nels)
    τII = zeros(Float64, nels)

    for iel in 1:nels
        local_nodes = SVector{NV}(ntuple(i -> el2n_v[i, iel], Val(NV)))
        vxloc = SVector{NV}(ntuple(i -> vx[local_nodes[i]], Val(NV)))
        vyloc = SVector{NV}(ntuple(i -> vy[local_nodes[i]], Val(NV)))
        τxx_old_loc = SVector{NV}(ntuple(i -> τ_old[1][local_nodes[i]], Val(NV)))
        τyy_old_loc = SVector{NV}(ntuple(i -> τ_old[2][local_nodes[i]], Val(NV)))
        τxy_old_loc = SVector{NV}(ntuple(i -> τ_old[3][local_nodes[i]], Val(NV)))
        phase_loc = SVector{NV}(ntuple(i -> Int(phases_v[local_nodes[i]]), Val(NV)))
        geo_el = geo_v[iel]
        volume = 0.0

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
        τxx, τyy, τzz, τxy, τII,
        tauII = τII,
    )
end

"""
    update_old_stress_from_cells!(τ_old, post, el2n_v, nnodes_v)

Project cell-averaged stress diagnostics back to nodal old-stress arrays.
"""
function update_old_stress_from_cells!(τ_old, post, el2n_v, nnodes_v)
    τxx_nodes = zeros(Float64, nnodes_v)
    τyy_nodes = zeros(Float64, nnodes_v)
    τxy_nodes = zeros(Float64, nnodes_v)
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
    write_stokes_vtk(vtk_path, mesh_stokes, coords_v, el2nP_cpu, DoFsP_cpu, P_cpu, vx_cpu, vy_cpu, post; title)

Write pressure, velocity, strain-rate, and stress diagnostics to an ASCII VTK
unstructured-grid file using pressure triangle corners.
"""
function write_stokes_vtk(vtk_path, mesh_stokes, coords_v, el2nP_cpu, DoFsP_cpu, P_cpu, vx_cpu, vy_cpu, post; title = "FEMTools Stokes 2D")
    NP = size(el2nP_cpu, 1)
    vtk_nodes = sort!(unique(vec(el2nP_cpu)))
    vtk_node_map = zeros(Int32, length(coords_v))
    for (new_i, old_i) in enumerate(vtk_nodes)
        vtk_node_map[old_i] = Int32(new_i)
    end

    vtk_P = zeros(Float64, length(vtk_nodes))
    vtk_P_count = zeros(Int, length(vtk_nodes))
    for iel in 1:mesh_stokes.nels
        for a in 1:NP
            inode = vtk_node_map[el2nP_cpu[a, iel]]
            vtk_P[inode] += P_cpu[DoFsP_cpu[a, iel]]
            vtk_P_count[inode] += 1
        end
    end
    @. vtk_P /= vtk_P_count

    vtk_Vx = [vx_cpu[old_i] for old_i in vtk_nodes]
    vtk_Vy = [vy_cpu[old_i] for old_i in vtk_nodes]
    vtk_V = hypot.(vtk_Vx, vtk_Vy)

    open(vtk_path, "w") do io
        println(io, "# vtk DataFile Version 3.0")
        println(io, title)
        println(io, "ASCII")
        println(io, "DATASET UNSTRUCTURED_GRID")

        println(io, "POINTS $(length(vtk_nodes)) float")
        for old_i in vtk_nodes
            c = coords_v[old_i]
            println(io, "$(c[1]) $(c[2]) 0.0")
        end

        println(io, "CELLS $(mesh_stokes.nels) $(4 * mesh_stokes.nels)")
        for iel in 1:mesh_stokes.nels
            i1 = vtk_node_map[el2nP_cpu[1, iel]] - 1
            i2 = vtk_node_map[el2nP_cpu[2, iel]] - 1
            i3 = vtk_node_map[el2nP_cpu[3, iel]] - 1
            println(io, "3 $i1 $i2 $i3")
        end

        println(io, "CELL_TYPES $(mesh_stokes.nels)")
        for _ in 1:mesh_stokes.nels
            println(io, "5")
        end

        println(io, "POINT_DATA $(length(vtk_nodes))")
        for (name, field) in (("P", vtk_P), ("Vx", vtk_Vx), ("Vy", vtk_Vy), ("V", vtk_V))
            println(io, "SCALARS $name float 1")
            println(io, "LOOKUP_TABLE default")
            for value in field
                println(io, value)
            end
        end

        println(io, "CELL_DATA $(mesh_stokes.nels)")
        for (name, field) in (
            ("strain_xx", post.εxx),
            ("strain_yy", post.εyy),
            ("strain_zz", post.εzz),
            ("strain_xy", post.εxy),
            ("strain_II", post.εII),
            ("tau_xx", post.τxx),
            ("tau_yy", post.τyy),
            ("tau_zz", post.τzz),
            ("tau_xy", post.τxy),
            ("tau_II", post.τII),
        )
            println(io, "SCALARS $name float 1")
            println(io, "LOOKUP_TABLE default")
            for value in field
                println(io, value)
            end
        end
    end
    return nothing
end
