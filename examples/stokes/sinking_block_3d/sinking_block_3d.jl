using FEMTools

import Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using KernelAbstractions
using LinearAlgebra
using StaticArrays

include("mesh_q2q1.jl")

"""Compute cell-averaged divergence, strain-rate invariant, and stress invariant."""
function cell_diagnostics(mesh, cache, dr, element_v, cell_phase)
    v = map(Array, dr.v)
    el2n = Array(mesh.el2n)
    geo = Array(cache.geo_v)
    η = dr.η
    nels = mesh.nels
    divergence = zeros(nels)
    strain_II = zeros(nels)
    stress_II = zeros(nels)
    Nq = shape_function_values(element_v)

    for iel in 1:nels
        nodes = el2n[:, iel]
        vloc = ntuple(a -> SVector{27}(v[a][nodes]), 3)
        phase = cell_phase[iel]
        volume = 0.0
        for q in eachindex(Nq)
            ∂N∂x, dΩ = geo[iel][q]
            grad_v = SMatrix{3, 3}(ntuple(k -> begin
                a = (k - 1) % 3 + 1
                b = (k - 1) ÷ 3 + 1
                dot(∂N∂x[:, b], vloc[a])
            end, 9))
            ε = (grad_v + transpose(grad_v)) / 2
            εdev = ε - SMatrix{3, 3}(I) * (tr(ε) / 3)
            τ = 2η[phase] * εdev
            divergence[iel] += tr(grad_v) * dΩ
            strain_II[iel] += sqrt(sum(abs2, εdev) / 2) * dΩ
            stress_II[iel] += sqrt(sum(abs2, τ) / 2) * dΩ
            volume += dΩ
        end
        divergence[iel] /= volume
        strain_II[iel] /= volume
        stress_II[iel] /= volume
    end
    return (; divergence, strain_II, stress_II)
end

"""
    main(; half_width=0.1, max_area=1/16^3, backend=CPU(), ...)

Run one forward, purely viscous 3D sinking-block solve on a conforming
Hex27/Q1 mesh. The element residuals, pressure mass, AD Jacobian diagnostics,
and Powell--Hestenes/DYREL iteration are provided by FEMTools' dimension-generic
Stokes implementation; this file contains only problem setup and output.
"""
function main(;
    half_width = 0.1,
    max_area = 1 / 16^3,
    backend = CPU(),
    workgroup = 64,
    Δt = 1.0,
    η = (1.0, 1.0),
    ρ0 = (1.0, 2.0),
    γfact = 0.5,
    ϵ_tol = 1e-5,
    ncheck = 25,
    iterMax = 50_000,
    total_iterMax = 20_000,
    max_ph_iterations = 100,
    rel_drop = 0.1,
    verbose = true,
    output_dir = joinpath(@__DIR__, "output"),
)
    host = build_sinking_block_q2q1_mesh(; half_width, max_area)
    TDev = FEMTools.TA(backend)

    element_v = ReferenceElement(QuadraticElement{3, 27, Float64})
    element_P = ReferenceElement(LinearElement{3, 8, Float64})
    velocity_dofs = TDev(Int32.(eachindex(host.coords_v)))
    # MixedMesh stores the element-to-DoF map separately from the geometric
    # Q1 connectivity. Pressure DoFs are compact, whereas geometry must index
    # the matching corner nodes in the Q2 coordinate array.
    pressure_dofs = TDev(host.el2n_p)
    pressure_geometry = TDev(reshape(
        host.pressure_velocity_nodes[vec(host.el2n_p)], size(host.el2n_p),
    ))
    mesh = MixedMesh(
        element_v, element_P,
        TDev(host.coords_v), velocity_dofs, TDev(host.el2n_v),
        pressure_dofs, pressure_geometry,
    )
    cache = MixedMeshCache(backend, workgroup, mesh, element_v, element_P)

    phases = TDev(reshape(Int.(host.phase), 1, :))
    α = (0.0, 0.0)
    K = (Inf, Inf)
    ηb = (Inf, Inf)
    g = (0.0, 0.0, -1.0)
    dr = StokesDRND(
        backend, Val(3), mesh.nnodes, mesh.nnodesP, η, ηb, α;
        ρ0, K, g, Tref = 0.0, CFL_v = 0.9, c_fact = 0.7, ϵ = ϵ_tol,
    )

    # Hydrostatic warm start measured downward from the top boundary z=0.5.
    P0 = Float64[ρ0[1] * abs(g[3]) * (0.5 - x[3]) for x in host.coords_p]
    copyto!(dr.P, P0)
    copyto!(dr.P0, P0)

    # Free slip: constrain only the component normal to each box face.
    tol = 100eps(Float64)
    nodes = ntuple(a -> TDev(Int32[
        i for i in eachindex(host.coords_v)
        if isapprox(abs(host.coords_v[i][a]), 0.5; atol = tol)
    ]), 3)
    values = ntuple(a -> KernelAbstractions.zeros(backend, Float64, length(nodes[a])), 3)
    for a in 1:3
        FEMTools.apply_dirichlet!(dr.v[a], nodes[a], values[a], backend, workgroup)
    end

    # The weak pressure residual is divided by its lumped mass inside the
    # solver. A constant viscosity scale is sufficient for this first case.
    γP = KernelAbstractions.zeros(backend, Float64, mesh.nnodesP)
    fill!(γP, γfact * sum(η) / length(η))

    stats = solve_stokes_dyrel_nd!(
        dr, mesh, cache, element_v, element_P, phases, phases,
        Δt, γP, nodes, values, backend, workgroup;
        ncheck, ϵ_tol, iterMax, total_iterMax, max_ph_iterations,
        rel_drop, verbose,
    )

    velocity = map(Array, dr.v)
    pressure = Array(dr.P)
    pressure_on_velocity = zeros(length(host.coords_v))
    pressure_on_velocity[host.pressure_velocity_nodes] .= pressure
    speed = sqrt.(velocity[1].^2 .+ velocity[2].^2 .+ velocity[3].^2)
    diagnostics = cell_diagnostics(mesh, cache, dr, element_v, host.phase)

    mkpath(output_dir)
    output_mesh = Mesh(
        element_v, nothing, nothing, host.coords_v,
        Int32.(eachindex(host.coords_v)), host.el2n_v, host.boundary_nodes,
    )
    vtk_path = joinpath(output_dir, "sinking_block_q2q1_step_001.vtk")
    write_vtk(
        vtk_path, output_mesh;
        point_data = (;
            Vx = velocity[1], Vy = velocity[2], Vz = velocity[3],
            velocity_magnitude = speed,
            pressure = pressure_on_velocity,
        ),
        cell_data = (;
            phase = host.phase,
            density = [ρ0[p] for p in host.phase],
            viscosity = [η[p] for p in host.phase],
            divergence = diagnostics.divergence,
            strain_rate_II = diagnostics.strain_II,
            stress_II = diagnostics.stress_II,
        ),
        title = "3D Q2/Q1 sinking block after one Stokes step",
    )

    @info "Completed 3D sinking-block forward solve" stats vtk_path
    # return (; dr, mesh, cache, stats, diagnostics, vtk_path)
    nothing
end

@time main()
