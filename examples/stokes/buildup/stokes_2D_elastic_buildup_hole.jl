import Pkg
Pkg.activate(@__DIR__)

using FEMTools
using StaticArrays
using Printf
using Statistics
using LinearAlgebra
using DomainSets
using DomainSets: ×
using KernelAbstractions
using Gmsh
using GLMakie: Figure, Axis, Colorbar, poly!, scatterlines!, lines!, Point2f, DataAspect

const backend = CPU()
const workgroup = 128
const YEAR = 365.25 * 24 * 3600
const KYR = 1.0e3 * YEAR

"""
    build_gmsh_t7_tunnel_mesh(; lx, ly, cx, cy, r, max_area = nothing)

Build an unstructured T7 velocity mesh of the rectangle `[0,lx]×[0,ly]` with a
circular tunnel of radius `r` centred at `(cx, cy)`, using Gmsh's OCC geometry
kernel. Returns `(coords, el2n, outer_nodes)`, where `outer_nodes` are the nodes
on the outer rectangle edges (used to impose the velocity BCs). The tunnel-rim
nodes are intentionally left unconstrained → free surface.
"""
function build_gmsh_t7_tunnel_mesh(; lx, ly, cx, cy, r, max_area = nothing)
    mesh_size = min(sqrt(2 * something(max_area, lx * ly / (2 * 48^2) * 10)), r / 2)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("stokes_tunnel")
    try
        rect = gmsh.model.occ.addRectangle(0, 0, 0, lx, ly)
        disk = gmsh.model.occ.addDisk(cx, cy, 0, r, r)
        gmsh.model.occ.cut([(2, rect)], [(2, disk)])
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(2)

        nodetags, coords_flat, _ = gmsh.model.mesh.getNodes()
        tag2idx = Dict{Int, Int32}(Int(tag) => Int32(i) for (i, tag) in enumerate(nodetags))
        coords = [SVector{2, Float64}(coords_flat[3(i - 1) + 1], coords_flat[3(i - 1) + 2])
                  for i in eachindex(nodetags)]

        elemtypes, _, elemnodetags = gmsh.model.mesh.getElements(2)
        tri_idx = findfirst(==(9), elemtypes) # 6-node second-order triangle
        isnothing(tri_idx) && error("No Gmsh T6 triangle elements found")
        tri_flat = elemnodetags[tri_idx]
        nels = length(tri_flat) ÷ 6

        n_t6 = length(coords)
        el2n = Matrix{Int32}(undef, 7, nels)
        for iel in 1:nels
            base = 6 * (iel - 1)
            for a in 1:6
                el2n[a, iel] = tag2idx[Int(tri_flat[base + a])]
            end
        end

        inside = count(iel -> begin
            c = (coords[el2n[1, iel]] + coords[el2n[2, iel]] + coords[el2n[3, iel]]) / 3
            hypot(c[1] - cx, c[2] - cy) < r
        end, 1:nels)
        inside == 0 || error("Gmsh returned $inside elements inside the tunnel")

        sizehint!(coords, n_t6 + nels)
        for iel in 1:nels
            c1 = coords[el2n[1, iel]]
            c2 = coords[el2n[2, iel]]
            c3 = coords[el2n[3, iel]]
            push!(coords, (c1 + c2 + c3) / 3)
            el2n[7, iel] = Int32(n_t6 + iel)
        end

        tol = 1e-8 * max(lx, ly)
        outer_nodes = Int32[
            i for i in 1:n_t6
            if abs(coords[i][1]) ≤ tol ||
               abs(coords[i][1] - lx) ≤ tol ||
               abs(coords[i][2]) ≤ tol ||
               abs(coords[i][2] - ly) ≤ tol
        ]

        return coords, el2n, sort!(unique!(outer_nodes))
    finally
        gmsh.finalize()
    end
end

function precompute_geometry(coords, el2n, nels, element::ReferenceElement{T}) where T <: AbstractElement{2, N} where N
    ip    = element.integration_points
    NQ    = length(ip.ω)
    ξq    = ntuple(q -> SVector(ip.ξ[q], ip.η[q]), NQ)
    ∂N∂ξq = ntuple(q -> eval_shape_function_jacobian(element, ξq[q]), NQ)
    geo   = KernelAbstractions.allocate(backend, NTuple{NQ, Tuple{SMatrix{N, 2, Float64, 2N}, Float64}}, nels)
    FEMTools.precompute_geometry_kernel!(backend, workgroup)(geo, coords, el2n, ∂N∂ξq, ip.ω, Val(N); ndrange = nels)
    KernelAbstractions.synchronize(backend)
    return geo
end

function main(;
    max_area = nothing,
    lx = 4.0e3,
    ly = 2.0e3,
    tunnel_depth = 500.0,
    r_hole = 150.0,
    endtime_kyr = 100.0,
    η0 = 1.0e22,
    G0 = 1.0e10,
    ρ0_mat = 2700.0,
    g0 = 9.81,
    γfact = 20.0,
    first_dt_kyr = 1.0e-4,  # 0.1 yr: keep explicit free-surface advection tame
    later_dt_kyr = 1.0e-3,  # 1 yr
    dt_switch_kyr = 0.1,
    ncheck = 50,
    ϵ_tol = 1.0e-6,
    niter_PH = 20,
    niter_inner = 25_000,
    total_iterMax = 150_000,
    rel_drop0 = 0.75,
    vtk_every = 1,
    show_plot = true,
    verbose = true,
)
    t_end = endtime_kyr * KYR
    cx = lx / 2
    cy = ly - tunnel_depth
    max_area = isnothing(max_area) ? (lx * ly) / (2 * 48^2) * 10 : max_area

    η = (Float64(η0), Float64(η0))
    ηb = (Float64(η0 * γfact), Float64(η0 * γfact))
    α = (0.0, 0.0)
    ρ0 = (Float64(ρ0_mat), Float64(ρ0_mat))
    # Keep the Stokes solve incompressible for PH/DYREL stability. For finite
    # compressibility: K0 = 2 * G0 * (1 + ν) / (3 * (1 - 2ν)).
    K = (Inf, Inf)
    G = (Float64(G0), Float64(G0))
    g = (0.0, -Float64(g0))   # gravity points downward → self-weight loading
    Tref = 0.0
    plastic = nothing

    element_v = ReferenceElement(QuadraticElement{2, 7, Float64})
    element_P = ReferenceElement(LinearElement{2, 3, Float64})

    coords_v_cpu, el2n_v_cpu, outer_nodes = build_gmsh_t7_tunnel_mesh(;
        lx, ly, cx, cy, r = r_hole, max_area,
    )
    DoFs_v_cpu = Int32.(1:length(coords_v_cpu))
    mesh_v = FEMTools.Mesh(
        element_v, nothing, nothing,
        coords_v_cpu, DoFs_v_cpu, el2n_v_cpu, outer_nodes,
    )
    mesh_stokes = MixedMesh(mesh_v, element_P)

    @info "Gravitational-loading FEM model (tunnel below flat terrain)" nnodes_v=mesh_stokes.nnodes nnodes_P=mesh_stokes.nnodesP nels=mesh_stokes.nels tunnel_depth r_hole n_outer=length(outer_nodes)

    cache = MixedMeshCache(backend, workgroup, mesh_stokes, element_v, element_P)
    TDev = FEMTools.TA(backend)
    NV = length(element_v)
    NP = length(element_P)

    dr = StokesDR(
        backend,
        mesh_stokes.nnodes,
        mesh_stokes.nnodesP,
        η, ηb, α;
        ρ0,
        K,
        g,
        Tref,
        CFL_v = 1 / sqrt(2.1),
        CFL_P = 1 / sqrt(2.1),
        c_fact = 0.9,
    )
    τ_old = (dr.τxx_old, dr.τyy_old, dr.τxy_old)

    coords_v = Array(mesh_stokes.coords)
    el2n_v_cpu = Array(mesh_stokes.el2n)
    el2nP_cpu = Array(mesh_stokes.el2nP)
    DoFsP_cpu = Array(mesh_stokes.DoFsP)
    edge_count = Dict{Tuple{Int32, Int32}, Int}()
    for iel in axes(el2nP_cpu, 2), (a, b) in ((1, 2), (2, 3), (3, 1))
        edge = minmax(Int32(el2nP_cpu[a, iel]), Int32(el2nP_cpu[b, iel]))
        edge_count[edge] = get(edge_count, edge, 0) + 1
    end
    tunnel_corner_nodes = Int32[]
    for ((a, b), n) in edge_count
        if n == 1 &&
           abs(norm(coords_v[Int(a)] - SVector(cx, cy)) - r_hole) ≤ 1e-6 &&
           abs(norm(coords_v[Int(b)] - SVector(cx, cy)) - r_hole) ≤ 1e-6
            push!(tunnel_corner_nodes, a, b)
        end
    end
    isempty(tunnel_corner_nodes) && error("No tunnel boundary edges found")
    tunnel_corner_nodes = sort!(
        unique!(tunnel_corner_nodes),
        by = n -> atan(coords_v[Int(n)][2] - cy, coords_v[Int(n)][1] - cx),
    )

    # Single phase everywhere: the tunnel is an absence of material, not a second
    # phase, so every element stays phase 1.
    phases_v_cpu = ones(Int, NV, mesh_stokes.nels)
    phases_P_cpu = ones(Int, NP, mesh_stokes.nels)

    # The body starts at rest; gravity is the only driving force, so leave the
    # velocity fields zero-initialised (StokesDR already zeros them).

    # Gravitational-loading boundary conditions:
    #   • no-slip bottom (y≈0):      vx = vy = 0
    #   • free-slip sides (x≈0/lx):  vx = 0, vy free
    #   • free surface (top y≈ly):   no constraint (natural traction-free BC)
    #   • tunnel rim:                no constraint (free surface)
    # So vx is pinned on the two sides plus the bottom; vy is pinned on the bottom only.
    # Node sets are selected geometrically from the outer boundary, so the
    # tunnel-rim nodes are never picked up and stay traction-free.
    Γnodes = Array(mesh_v.Γnodes)
    coords = Array(mesh_v.coords)
    tol_x = max(lx, ly) * eps(Float64) * 32
    on_bottom(n) = abs(coords[n][2]) ≤ tol_x
    on_side(n) = abs(coords[n][1]) ≤ tol_x || abs(coords[n][1] - lx) ≤ tol_x
    vx_nodes = sort!(unique!(Int32[n for n in Γnodes if on_side(n) || on_bottom(n)]))
    vy_nodes = Int32[n for n in Γnodes if on_bottom(n)]

    vx_bc = zeros(Float64, length(vx_nodes))
    vy_bc = zeros(Float64, length(vy_nodes))

    apply_bc!(dr.vx, DirichletBoundaryCondition(nothing, vx_nodes, vx_bc))
    apply_bc!(dr.vy, DirichletBoundaryCondition(nothing, vy_nodes, vy_bc))

    @info "Gravitational-loading BCs (no-slip bottom, free-slip sides, free surface top+tunnel)" n_vx=length(vx_nodes) n_vy=length(vy_nodes)

    # Seed pressure from a lithostatic solve on the continuous linear corner
    # mesh, then sample it to the discontinuous P1 pressure DoFs. The analytical
    # ρg depth profile is only the warm start for the lithostatic DR solve.
    corner_nodes = sort!(unique(vec(el2nP_cpu)))
    corner_id = Dict{Int32, Int32}(old => Int32(i) for (i, old) in enumerate(corner_nodes))
    coords_litho = coords_v[Int.(corner_nodes)]
    el2n_litho = Matrix{Int32}(undef, 3, mesh_stokes.nels)
    for iel in 1:mesh_stokes.nels, a in 1:3
        el2n_litho[a, iel] = corner_id[Int32(el2nP_cpu[a, iel])]
    end
    mesh_litho = FEMTools.Mesh(backend, coords_litho, el2n_litho)
    geo_litho = precompute_geometry(mesh_litho.coords, mesh_litho.el2n, mesh_litho.nels, element_P)

    top_nodes_litho = Int32[
        corner_id[Int32(n)] for n in outer_nodes
        if haskey(corner_id, Int32(n)) && abs(coords[n][2] - ly) ≤ tol_x
    ]
    lp_dr = LithostaticPressureDR(backend, mesh_litho.nnodes, ρ0, α, K; CFL = 0.9, ϵ = 1e-2)
    T_stokes = Array(dr.T)
    copyto!(lp_dr.T, Float64[T_stokes[Int(n)] for n in corner_nodes])
    P0_litho = Float64[ρ0_mat * g0 * (ly - coords_litho[i][2]) for i in eachindex(coords_litho)]
    copyto!(lp_dr.P, P0_litho)
    Γ_P_dofs = TDev(top_nodes_litho)
    Γ_P_zero_vals = TDev(zeros(Float64, length(top_nodes_litho)))
    solver!(lp_dr, mesh_litho, geo_litho, element_P, Γ_P_dofs, Γ_P_zero_vals, Γ_P_zero_vals,
        backend, workgroup; ncheck = 50, verbose = false, Tref = Tref, g = g)

    P_litho_l = Array(lp_dr.P)
    P_litho_P = zeros(Float64, mesh_stokes.nnodesP)
    for iel in 1:mesh_stokes.nels, a in 1:3
        P_litho_P[DoFsP_cpu[a, iel]] = P_litho_l[corner_id[Int32(el2nP_cpu[a, iel])]]
    end
    copyto!(dr.P, P_litho_P)
    copyto!(dr.P0, P_litho_P)

    h = sqrt(2 * max_area)
    ηmax = maximum(η)
    Δτ_V_seed = dr.CFL_v * h^2 / (4 * ηmax)
    fill!(dr.PC_vx, 1 / Δτ_V_seed)
    fill!(dr.PC_vy, 1 / Δτ_V_seed)

    γP = KernelAbstractions.zeros(backend, Float64, mesh_stokes.nnodesP)

    _λmin(step, rate, ΔR, PC) = begin
        dV = step .* rate
        denom = sum(dV .^ 2)
        denom == 0 ? 0.0 : abs(sum(dV .* (ΔR ./ PC))) / denom
    end

    _cheb(Δτ, λmin, c_fact) = begin
        c = min(2 * sqrt(λmin) * c_fact, 2.0 / Δτ)
        (2 * Δτ^2 / (2 + c * Δτ), (2 - c * Δτ) / (2 + c * Δτ))
    end

    out_dir = joinpath(@__DIR__, "output_stokes_tunnel")
    mkpath(out_dir)

    time_kyr = Float64[]
    τyy_max = Float64[]
    vmag_max = Float64[]
    post = nothing
    t = 0.0
    istep = 0

    @info "Starting gravitational-loading viscoelastic solve (tunnel)" endtime_kyr η0 G0 ρ0_mat g0 maxwell_kyr=(η0 / G0 / KYR)

    while t < t_end - eps(t_end)
        istep += 1
        Δt_kyr = t < dt_switch_kyr * KYR ? first_dt_kyr : later_dt_kyr
        Δt = min(Δt_kyr * KYR, t_end - t)
        t += Δt

        # Pressure mass + viscosity-weighted scaling, recomputed each step on the
        # current (advected) geometry. The K-aware form sets γ_phy = K·Δt, so the
        # finite bulk modulus caps the pressure penalty (γ_eff ≈ K·Δt ≪ γfact·η),
        # making the incompressible coupling far less stiff and Powell-Hestenes
        # converge much faster than the K=Inf hard-incompressible limit.
        FEMTools.assemble_viscosity_weighted_pressure_scaling!(
            dr.M_P, γP,
            mesh_stokes.el2n, mesh_stokes.DoFsP, cache.geo_P, mesh_stokes.nels,
            element_v, element_P,
            phases_v_cpu, dr.η, γfact, dr.K, Δt,
            backend, workgroup,
        )

        # Residual normalization scales for self-weight loading:
        # lithostatic stress ρgH, and the strain rate it drives through η.
        τ_ref = max(ρ0_mat * g0 * ly, eps(Float64))
        Rv_ref = max(τ_ref * h, eps(Float64))
        RP_ref = max(τ_ref / η0, eps(Float64))

        copyto!(dr.P0, dr.P)
        copyto!(dr.T0, dr.T)
        fill!(dr.∂vx∂τ, 0)
        fill!(dr.∂vy∂τ, 0)
        fill!(dr.Rv_x0, 0)
        fill!(dr.Rv_y0, 0)

        FEMTools.assemble_augmented_momentum_jacobian_matrices_atomix!(
            dr.∂Rv_x∂vx, dr.PC_vx, dr.∂Rv_y∂vy, dr.PC_vy,
            dr.vx, dr.vy, dr.P, dr.P0, dr.T, dr.T0,
            mesh_stokes.el2n, mesh_stokes.DoFsP, cache.geo_v, cache.geo_P, mesh_stokes.nels,
            element_v, element_P,
            phases_v_cpu, phases_P_cpu, τ_old, plastic, dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref,
            dr.ηb, Δt, γP, dr.M_P,
            backend, workgroup,
        )
        λmax_vx0 = max(maximum(dr.∂Rv_x∂vx ./ dr.PC_vx), eps(Float64))
        λmax_vy0 = max(maximum(dr.∂Rv_y∂vy ./ dr.PC_vy), eps(Float64))
        Δτ_vx0 = 2 / sqrt(λmax_vx0) * dr.CFL_v
        Δτ_vy0 = 2 / sqrt(λmax_vy0) * dr.CFL_v

        α_vx, β_vx = _cheb(Δτ_vx0, 0.0, dr.c_fact)
        α_vy, β_vy = _cheb(Δτ_vy0, 0.0, dr.c_fact)
        err_min = Inf
        err = 2ϵ_tol
        err_v0 = 0.0
        err_P0 = 0.0
        err_v00 = 0.0
        iter = 0
        rel_drop = Float64(rel_drop0)

        verbose && @printf("step = %04d, time = %.3f kyr, dt = %.3f kyr\n", istep, t / KYR, Δt / KYR)

        for itPH in 1:niter_PH
            FEMTools.assemble_momentum_residual_matrices_atomix!(
                dr.Rv_x, dr.Rv_y,
                dr.vx, dr.vy, dr.P, dr.T, nothing,
                mesh_stokes.el2n, mesh_stokes.DoFsP, cache.geo_v, mesh_stokes.nels,
                element_v, element_P,
                phases_v_cpu, τ_old, plastic, dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, Δt,
                backend, workgroup,
            )
            FEMTools.apply_dirichlet!(dr.Rv_x, vx_nodes, vx_bc, backend, workgroup)
            FEMTools.apply_dirichlet!(dr.Rv_y, vy_nodes, vy_bc, backend, workgroup)

            FEMTools.assemble_pressure_residual_matrices_atomix!(
                dr.RP,
                dr.vx, dr.vy, dr.P, dr.P0, dr.T, dr.T0,
                mesh_stokes.el2n, mesh_stokes.DoFsP, cache.geo_v, cache.geo_P, mesh_stokes.nels,
                element_v, element_P,
                phases_P_cpu, dr.α, dr.ηb, Δt,
                backend, workgroup,
            )

            err_P = norm(dr.RP ./ dr.M_P) / (sqrt(mesh_stokes.nnodesP) * RP_ref)
            err_v = (norm(dr.Rv_x) + norm(dr.Rv_y)) / (2 * sqrt(mesh_stokes.nnodes) * Rv_ref)
            if itPH == 1
                err_P0 = err_P + eps(err_P)
                err_v0 = err_v + eps(err_v)
            elseif itPH == 2
                err_P0 = err_P + eps(err_P)
            end

            # Absolute convergence: err_v and err_P are already normalized by the
            # physical scales Rv_ref / RP_ref, so use them directly rather than
            # ratios to the itPH=1 residual. Anchoring to err_v0 fails here because
            # the pre-pressure rest-state residual is tiny, so the ratio can never
            # reach ϵ_tol. (err_v0/err_P0 kept only for the diagnostic print.)
            err_v_rel = err_v
            err_P_rel = err_P
            err = max(err_v_rel, err_P_rel)

            isnan(err) && error("NaN detected in outer loop at step=$istep PH=$itPH")
            err > 1e10 && error("Kaboom! Error > 1e10 in outer loop at step=$istep PH=$itPH")

            verbose && @printf("  itPH = %02d iter = %06d err = %.3e - norm[Rv=%.3e %.3e, Rp=%.3e %.3e]\n",
                itPH, iter, err, err_v, err_v / err_v0, err_P, err_P / err_P0)

            err < ϵ_tol && break

            if err > err_min * 1.05
                rel_drop = max(rel_drop * 0.1, 1e-3)
            end
            err_min = min(err_min, err)

            ϵ_vel = err * rel_drop
            itPT = 0

            while err > ϵ_vel && itPT ≤ niter_inner
                itPT += 1
                iter += 1
                do_jac = (mod(itPT, ncheck) == 0) || (itPT == 1)

                FEMTools.assemble_pressure_residual_matrices_atomix!(
                    dr.RP,
                    dr.vx, dr.vy, dr.P, dr.P0, dr.T, dr.T0,
                    mesh_stokes.el2n, mesh_stokes.DoFsP, cache.geo_v, cache.geo_P, mesh_stokes.nels,
                    element_v, element_P,
                    phases_P_cpu, dr.α, dr.ηb, Δt,
                    backend, workgroup,
                )

                @. dr.Pnum = γP * dr.RP / dr.M_P

                FEMTools.assemble_momentum_residual_matrices_atomix!(
                    dr.Rv_x, dr.Rv_y,
                    dr.vx, dr.vy, dr.P, dr.T, dr.Pnum,
                    mesh_stokes.el2n, mesh_stokes.DoFsP, cache.geo_v, mesh_stokes.nels,
                    element_v, element_P,
                    phases_v_cpu, τ_old, plastic, dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, Δt,
                    backend, workgroup,
                )

                if do_jac
                    FEMTools.assemble_augmented_momentum_jacobian_matrices_atomix!(
                        dr.∂Rv_x∂vx, dr.PC_vx, dr.∂Rv_y∂vy, dr.PC_vy,
                        dr.vx, dr.vy, dr.P, dr.P0, dr.T, dr.T0,
                        mesh_stokes.el2n, mesh_stokes.DoFsP, cache.geo_v, cache.geo_P, mesh_stokes.nels,
                        element_v, element_P,
                        phases_v_cpu, phases_P_cpu, τ_old, plastic, dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref,
                        dr.ηb, Δt, γP, dr.M_P,
                        backend, workgroup,
                    )
                end

                FEMTools.apply_dirichlet!(dr.Rv_x, vx_nodes, vx_bc, backend, workgroup)
                FEMTools.apply_dirichlet!(dr.∂vx∂τ, vx_nodes, vx_bc, backend, workgroup)
                FEMTools.apply_dirichlet!(dr.Rv_y, vy_nodes, vy_bc, backend, workgroup)
                FEMTools.apply_dirichlet!(dr.∂vy∂τ, vy_nodes, vy_bc, backend, workgroup)

                FEMTools.stokes_update_rate!(dr.∂vx∂τ, dr.Rv_x, dr.PC_vx, β_vx, mesh_stokes.nnodes, backend, workgroup)
                FEMTools.stokes_update_variable!(dr.vx, dr.∂vx∂τ, -α_vx, mesh_stokes.nnodes, backend, workgroup)
                FEMTools.stokes_update_rate!(dr.∂vy∂τ, dr.Rv_y, dr.PC_vy, β_vy, mesh_stokes.nnodes, backend, workgroup)
                FEMTools.stokes_update_variable!(dr.vy, dr.∂vy∂τ, -α_vy, mesh_stokes.nnodes, backend, workgroup)

                FEMTools.apply_dirichlet!(dr.vx, vx_nodes, vx_bc, backend, workgroup)
                FEMTools.apply_dirichlet!(dr.vy, vy_nodes, vy_bc, backend, workgroup)

                if do_jac
                    err_v_inner = (norm(dr.Rv_x) + norm(dr.Rv_y)) / (2 * sqrt(mesh_stokes.nnodes) * Rv_ref)
                    if iter == 1
                        err_v00 = err_v_inner + eps(err_v_inner)
                    end
                    # Absolute (physically normalized) inner velocity residual.
                    err = err_v_inner
                    isnan(err) && error("NaN detected in inner loop step=$istep PH=$itPH PT=$itPT")
                    err > 1e10 && error("Kaboom! Error > 1e10 in inner loop step=$istep PH=$itPH PT=$itPT")

                    λmax_vx = max(maximum(dr.∂Rv_x∂vx ./ dr.PC_vx), eps(Float64))
                    λmax_vy = max(maximum(dr.∂Rv_y∂vy ./ dr.PC_vy), eps(Float64))
                    Δτ_vx = 2 / sqrt(λmax_vx) * dr.CFL_v
                    Δτ_vy = 2 / sqrt(λmax_vy) * dr.CFL_v

                    λmin_vx = itPT == 1 ? 0.0 : _λmin(α_vx, dr.∂vx∂τ, dr.Rv_x .- dr.Rv_x0, dr.PC_vx)
                    λmin_vy = itPT == 1 ? 0.0 : _λmin(α_vy, dr.∂vy∂τ, dr.Rv_y .- dr.Rv_y0, dr.PC_vy)

                    α_vx, β_vx = _cheb(Δτ_vx, λmin_vx, dr.c_fact)
                    α_vy, β_vy = _cheb(Δτ_vy, λmin_vy, dr.c_fact)

                    copyto!(dr.Rv_x0, dr.Rv_x)
                    copyto!(dr.Rv_y0, dr.Rv_y)
                end

                itPT == niter_inner && @printf("  inner: max iters (%d) reached at PH=%d\n", niter_inner, itPH)
                iter > total_iterMax && break
            end

            @. dr.P += γP * dr.RP / dr.M_P
            FEMTools.remove_pressure_mean!(dr.P, dr.M_P)
            iter > total_iterMax && break
        end

        P_cpu = Array(dr.P)
        vx_cpu = Array(dr.vx)
        vy_cpu = Array(dr.vy)
        post = FEMTools.compute_strain_rate_stress_postprocess(
            vx_cpu, vy_cpu,
            el2n_v_cpu,
            Array(cache.geo_v),
            phases_v_cpu,
            τ_old,
            dr.η, G, Δt,
            element_v,
        )
        update_old_stress_from_cells!(τ_old, post, el2n_v_cpu, mesh_stokes.nnodes)

        vmag = maximum(hypot.(vx_cpu, vy_cpu))
        push!(time_kyr, t / KYR)
        push!(τyy_max, maximum(abs, post.τyy))
        push!(vmag_max, vmag)

        verbose && @printf("  max|τyy| = %.6e Pa, max|v| = %.6e m/yr\n",
            τyy_max[end], vmag * YEAR)

        if vtk_every > 0 && (mod(istep, vtk_every) == 0 || t ≥ t_end - eps(t_end))
            vtk_path = joinpath(out_dir, @sprintf("stokes_2D_elastic_buildup_tunnel_%04d.vtk", istep))
            write_stokes_vtk(vtk_path, mesh_stokes, coords_v, el2nP_cpu, DoFsP_cpu, P_cpu, vx_cpu, vy_cpu, post)
            @info "Wrote VTK file" vtk_path
        end

        # --- Lagrangian advection: move the mesh with the flow (explicit Euler).
        # Every velocity node is displaced by v·Δt, so the free top surface
        # subsides and the tunnel deforms. Free-slip keeps boundary nodes on their
        # walls (vy=0 on the base, vx=0 on the sides), so the node sets and the
        # domain box stay valid; the geometry cache is rebuilt on the new config.
        @inbounds for i in eachindex(coords_v)
            coords_v[i] += Δt * SVector(vx_cpu[i], vy_cpu[i])
        end
        copyto!(mesh_stokes.coords, coords_v)
        cache = MixedMeshCache(backend, workgroup, mesh_stokes, element_v, element_P)
    end

    isnothing(post) && error("No time steps were executed")

    # Final velocity magnitude, reduced from velocity nodes to one value per
    # element (mean over the 3 corner nodes) so it colours the same polygons as
    # the stress panel. Re-fetched here because dr.vx/vy hold the final state.
    vx_cpu = Array(dr.vx)
    vy_cpu = Array(dr.vy)
    vmag_node = hypot.(vx_cpu, vy_cpu)
    vmag_el = Float64[
        (vmag_node[el2nP_cpu[1, i]] + vmag_node[el2nP_cpu[2, i]] + vmag_node[el2nP_cpu[3, i]]) / 3
        for i in 1:mesh_stokes.nels
    ] .* YEAR   # m/yr

    pts = [Point2f(c[1] / 1e3, c[2] / 1e3) for c in coords_v]
    polys = [[pts[el2nP_cpu[1, i]], pts[el2nP_cpu[2, i]], pts[el2nP_cpu[3, i]]]
             for i in 1:mesh_stokes.nels]

    tunnel_ring = [Point2f(coords_v[Int(n)][1] / 1e3, coords_v[Int(n)][2] / 1e3) for n in tunnel_corner_nodes]
    push!(tunnel_ring, first(tunnel_ring))
    hole_x = getindex.(tunnel_ring, 1)
    hole_y = getindex.(tunnel_ring, 2)

    fig = Figure(size = (1300, 520))

    # Panel 1 — velocity magnitude field
    clims_v = extrema(vmag_el)
    ax1 = Axis(fig[1, 1];
        aspect = DataAspect(),
        title = "Velocity magnitude (m/yr)",
        xlabel = "x (km)",
        ylabel = "y (km)",
    )
    poly!(ax1, polys; color = vmag_el, colormap = :batlow, colorrange = clims_v, strokewidth = 0)
    poly!(ax1, tunnel_ring; color = :white, strokewidth = 0)
    lines!(ax1, hole_x, hole_y; color = :black, linewidth = 2)
    Colorbar(fig[1, 2]; colormap = :batlow, limits = clims_v, label = "|v| (m/yr)", width = 15, tellheight = false)

    # Panel 2 — deviatoric stress second invariant τII field
    τII_mpa = post.tauII ./ 1e6
    clims = extrema(τII_mpa)
    ax2 = Axis(fig[1, 3];
        aspect = DataAspect(),
        title = "Final τII (MPa)",
        xlabel = "x (km)",
        ylabel = "y (km)",
    )
    poly!(ax2, polys; color = τII_mpa, colormap = :magma, colorrange = clims, strokewidth = 0)
    poly!(ax2, tunnel_ring; color = :white, strokewidth = 0)
    lines!(ax2, hole_x, hole_y; color = :black, linewidth = 2)
    Colorbar(fig[1, 4]; colormap = :magma, limits = clims, label = "τII (MPa)", width = 15, tellheight = false)

    show_plot && display(fig)

    return (;
        time_kyr,
        τyy_max,
        vmag_max,
        post,
        figure = fig,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
