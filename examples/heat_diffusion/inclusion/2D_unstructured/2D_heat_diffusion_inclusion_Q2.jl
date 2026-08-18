using Gmsh
using KernelAbstractions
using StaticArrays
using Printf
using FEMTools

const backend   = CPU()
const workgroup = 1

function precompute_geometry(coords, el2n, nels, element::ReferenceElement{T}) where T<:AbstractElement{2, N} where N
    ip    = element.integration_points
    NQ    = length(ip.ω)
    ξq    = ntuple(q -> SVector(ip.ξ[q], ip.η[q]), NQ)
    ∂N∂ξq = ntuple(q -> eval_shape_function_jacobian(element, ξq[q]), NQ)
    geo   = KernelAbstractions.allocate(backend, NTuple{NQ, Tuple{SMatrix{N, 2, Float64, 2N}, Float64}}, nels)
    FEMTools.precompute_geometry_kernel!(backend, workgroup)(geo, coords, el2n, ∂N∂ξq, ip.ω, Val(N); ndrange = nels)
    KernelAbstractions.synchronize(backend)
    return geo
end

function build_mesh_Q2(; L, mesh_size)
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
        gmsh.model.add("heat_inclusion_q2")

        s = gmsh.model.occ.addRectangle(-L, -L, 0.0, 2L, 2L)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
        gmsh.model.mesh.setRecombine(2, s)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(2)

        nodetags, coords_flat, _ = gmsh.model.mesh.getNodes()
        tag2idx = Dict(Int(tag) => Int32(i) for (i, tag) in enumerate(nodetags))
        coords = [SVector{2, Float64}(coords_flat[3(i - 1) + 1], coords_flat[3(i - 1) + 2]) for i in eachindex(nodetags)]

        elemtypes, _, elemnodetags = gmsh.model.mesh.getElements(2)
        quad_idx = findfirst(==(10), elemtypes)
        isnothing(quad_idx) && error("No 9-node quadrilateral elements found in Gmsh mesh.")
        quad_flat = elemnodetags[quad_idx]
        nels = length(quad_flat) ÷ 9
        el2n = Matrix{Int32}(undef, 9, nels)
        for iel in 1:nels, i in 1:9
            el2n[i, iel] = tag2idx[Int(quad_flat[9(iel - 1) + i])]
        end

        tol = L * 1e-8
        Γ = Int32[i for i in eachindex(coords) if
            abs(coords[i][1] + L) < tol || abs(coords[i][1] - L) < tol ||
            abs(coords[i][2] + L) < tol || abs(coords[i][2] - L) < tol]

        return coords, el2n, Γ
    finally
        gmsh.finalize()
    end
end

function main(nels; do_plot = true, nsteps = 50, verbose = true)
    TDev = FEMTools.TA(backend)

    L        = 50e3
    h        = 2L / nels[1]
    element  = ReferenceElement(QuadraticElement{2, 9, Float64})
    coords_cpu, el2n_cpu, Γ = build_mesh_Q2(; L, mesh_size = h)
    mesh = FEMTools.Mesh(backend, coords_cpu, el2n_cpu; order = 2)

    k   = (3.0,)
    Cp  = (1200.0,)
    ρ0  = (3300.0,)
    α   = (3e-5,)
    K   = (1e11,)
    Tref = 273.0
    Δt  = 100e3 * 365 * 24 * 3600

    T_bg = 273.0
    ΔT   = 500.0
    σ    = L / 8

    Γ_dofs = TDev(Γ)
    Γ_vals = TDev(fill(T_bg, length(Γ)))
    Γ_zero = zero(Γ_vals)

    geo = precompute_geometry(mesh.coords, mesh.el2n, mesh.nels, element)
    dr = ThermalDiffusionDR(backend, mesh.nnodes, k, Cp, ρ0, α, K; CFL=0.9, ϵ=1e-8)

    copyto!(dr.T, TDev(Float64[T_bg + ΔT * exp(-(p[1]^2 + p[2]^2) / (2σ^2)) for p in coords_cpu]))
    apply_dirichlet!(dr.T, Γ_dofs, Γ_vals, backend, workgroup)
    copyto!(dr.T0, dr.T)

    history    = Tuple{Vector{Int}, Vector{Float64}}[]
    solve_time = 0.0
    for step in 1:nsteps
        verbose && @printf("─── time step %d / %d  (t = %.3g s) ───\n", step, nsteps, step * Δt)
        copyto!(dr.T0, dr.T)
        fill!(dr.∂T∂τ, 0)
        Δwall = @elapsed hist = solver!(dr, Δt, mesh, geo, element, Γ_dofs, Γ_zero, Γ_vals, backend, workgroup; ncheck = 5, Tref = Tref, verbose = verbose)
        solve_time += Δwall
        push!(history, hist)
    end

    if do_plot
        @eval using GLMakie
        @eval using GeometryBasics
        pts2d = [Point2f(p[1] / 1e3, p[2] / 1e3) for p in coords_cpu]
        faces = [QuadFace(Int(el2n_cpu[1, i]), Int(el2n_cpu[2, i]), Int(el2n_cpu[3, i]), Int(el2n_cpu[4, i])) for i in axes(el2n_cpu, 2)]
        fig = Figure(size = (600, 600))
        ax = Axis(fig[1, 1]; aspect = DataAspect())
        mesh!(ax, pts2d, faces; color = Array(dr.T), colormap = :thermal, colorrange = (T_bg, T_bg + ΔT), shading = NoShading)
        display(fig)
    end

    return (; nels = mesh.nels, ndofs = mesh.nnodes, history, solve_time, dr)
end

if !(@isdefined RUN_SWEEP)
    main((15, 15))
end
