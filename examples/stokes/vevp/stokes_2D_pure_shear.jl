import Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using Printf
using Statistics
using StaticArrays
using LinearAlgebra
using DomainSets
using DomainSets: ×
using KernelAbstractions
using FEMTools
using GLMakie: Figure, Axis, Colorbar, poly!, scatterlines!, lines!, Point2f, DataAspect

const backend   = CPU()
const workgroup = 128

function precompute_geometry!(geo, coords, el2n, ∂N∂ξq, ω, ::Val{N}, nels) where N
    FEMTools.precompute_geometry_kernel!(backend, workgroup)(
        geo, coords, el2n, ∂N∂ξq, ω, Val(N);
        ndrange = nels,
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

function update_rate!(∂u∂τ, R, PC, β, ndofs)
    FEMTools.update_rate_kernel!(backend, workgroup)(
        ∂u∂τ, R, PC, β;
        ndrange = ndofs,
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

function update_variable!(u, ∂u∂τ, α_dr, ndofs)
    FEMTools.update_variable_kernel!(backend, workgroup)(
        u, ∂u∂τ, α_dr;
        ndrange = ndofs,
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

function main(; nsteps = 15, mesh_cells = (32, 32) .* 2, Δt = 1/6, show_plot = true)
    # Domain
    Lx, Ly = 1.0, 1.0
    nx, ny  = mesh_cells        # quad cells per direction (each splits into 2 triangles)

    # Background pure-shear strain rate (non-dimensional)
    ε̇_bg = 1.0

    # Material (2 phases: matrix + inclusion)
    η     = (1.0,     1.0)   # shear viscosity
    γfact = 20.0
    α     = (0.0,     0.0)   # thermal expansivity  (zero → isothermal)
    ρ0    = (1.0,     1.0)   # reference density
    K     = (4e0,     4e0)   # bulk modulus  (Inf → incompressible)
    ηb    = K                # pressure storage modulus; residual uses ηb * Δt
    G     = (1e0,     0.5)   # Shear modulus
    G_stokes = G
    # Cohesion chosen so the yield stress C·cosϕ = 1.6 at zero pressure.
    # Background deviatoric stress in pure shear is 2η·ε̇_bg = 2, so the
    # inclusion (lower G) will enter the plastic regime after a few steps.
    τy      = 1.6 / cosd(30)                   # cohesion C; yield stress = C·cosϕ = 1.6
    plastic = DruckerPrager(
        (π/6, π/6),                            # friction angle ϕ = 30° [rad]
        (0.0, 0.0),                            # dilation angle Ψ = 0°  [rad] (non-associated)
        (τy, τy),                              # cohesion C [same for both phases]
        (8.0e-3, 8.0e-3),                      # plastic regularisation viscosity η_reg
        K,                               # Kb (passed separately from elastic K)
    )
    g     = (0.0,     0.0)   # gravity vector
    Tref  = 0.0

    # DR solver
    ncheck = 50          # convergence check interval
    ϵ_tol  = 1e-6         # relative residual tolerance

    # ---------------------------------------------------------------------------
    # Meshes
    # ---------------------------------------------------------------------------

    Ω         = (0.0..Lx) × (0.0..Ly)
    element_v = ReferenceElement(QuadraticElement{2, 7, Float64})   # T7 (bubble)
    element_P = ReferenceElement(LinearElement{2, 3, Float64})      # P1-disc

    mesh_v      = Mesh(backend, Ω, element_v, (nx, ny))
    mesh_stokes = MixedMesh(mesh_v, element_P)

    @info "Mixed mesh (T7/P1-disc)" nnodes_v=mesh_stokes.nnodes nnodes_P=mesh_stokes.nnodesP nels=mesh_stokes.nels

    # ---------------------------------------------------------------------------
    # Geometry precompute  (both fields evaluated at velocity integration points)
    # ---------------------------------------------------------------------------

    ip_v  = element_v.integration_points
    NQ_v  = length(ip_v.ω)
    NV    = length(element_v)
    NP    = length(element_P)

    ξq_v    = ntuple(q -> SVector(ip_v.ξ[q], ip_v.η[q]), NQ_v)
    ∂N∂ξq_v = ntuple(q -> eval_shape_function_jacobian(element_v, ξq_v[q]), NQ_v)
    ∂N∂ξq_P = ntuple(q -> eval_shape_function_jacobian(element_P, ξq_v[q]), NQ_v)

    geo_v = Vector{NTuple{NQ_v, Tuple{SMatrix{NV, 2, Float64, 2NV}, Float64}}}(undef, mesh_stokes.nels)
    geo_P = Vector{NTuple{NQ_v, Tuple{SMatrix{NP, 2, Float64, 2NP}, Float64}}}(undef, mesh_stokes.nels)

    precompute_geometry!(geo_v, mesh_stokes.coords, mesh_stokes.el2n, ∂N∂ξq_v, ip_v.ω, Val(NV), mesh_stokes.nels)
    precompute_geometry!(geo_P, mesh_stokes.coords, mesh_stokes.el2nP, ∂N∂ξq_P, ip_v.ω, Val(NP), mesh_stokes.nels)

    # ---------------------------------------------------------------------------
    # StokesDR struct
    # ---------------------------------------------------------------------------

    material = StokesMaterial(; η, ηb, G = G_stokes, α, ρ0, K, g = Tuple(g), Tref)
    dr = StokesDR(
        backend, mesh_stokes.nnodes, mesh_stokes.nnodesP, material;
        CFL_v = 0.9, CFL_P = 0.9, c_fact = 0.9,
        stress_size = (NQ_v, mesh_stokes.nels),
        # CFL_v = 0.03, CFL_P = 0.9, c_fact = 0.5,
    )
    M_P = pressure_mass(dr)
    τ = (dr.τxx, dr.τyy, dr.τxy)
    τ_old = (dr.τxx_old, dr.τyy_old, dr.τxy_old)

    # ---------------------------------------------------------------------------
    # Phase assignment — circular inclusion
    # ---------------------------------------------------------------------------

    r_incl = 0.1
    cx     = Lx / 2
    cy     = Ly / 2

    in_incl(c) = (c[1] - cx)^2 + (c[2] - cy)^2 ≤ r_incl^2

    coords_v     = Array(mesh_stokes.coords)
    el2nP_cpu    = Array(mesh_stokes.el2nP)
    DoFsP_cpu    = Array(mesh_stokes.DoFsP)
    cell_phase = Int[
        in_incl(sum(a -> coords_v[mesh_stokes.el2n[a, iel]], 1:NV) / NV) ? 2 : 1
        for iel in 1:mesh_stokes.nels
    ]
    phases_v_cpu = repeat(reshape(cell_phase, 1, :), NV, 1)
    phases_P_cpu = repeat(reshape(cell_phase, 1, :), NP, 1)

    @info "Phases" n_incl_v=count(==(2), phases_v_cpu) n_incl_P=count(==(2), phases_P_cpu)

    # ---------------------------------------------------------------------------
    # Boundary conditions — pure shear
    #   vx = +ε̇_bg * (x - Lx/2),   vy = -ε̇_bg * (y - Ly/2)
    # ---------------------------------------------------------------------------

    Γnodes = Array(mesh_v.Γnodes)
    coords = Array(mesh_v.coords)
    tol = max(Lx, Ly) * eps(Float64) * 32
    vx_nodes = Int32[n for n in Γnodes if abs(coords[n][1]) ≤ tol || abs(coords[n][1] - Lx) ≤ tol]
    vy_nodes = Int32[n for n in Γnodes if abs(coords[n][2]) ≤ tol || abs(coords[n][2] - Ly) ≤ tol]

    bc_vx_vals = Float64[ ε̇_bg * (coords[n][1] - Lx / 2) for n in vx_nodes]
    bc_vy_vals = Float64[-ε̇_bg * (coords[n][2] - Ly / 2) for n in vy_nodes]

    # Seed the full interior with the analytical pure-shear field so the
    # solver starts with a good initial guess (boundary nodes are overwritten
    # by apply_bc! below; the result is identical on those nodes).
    copyto!(dr.vx, Float64[ ε̇_bg * (c[1] - Lx / 2) for c in coords_v])
    copyto!(dr.vy, Float64[-ε̇_bg * (c[2] - Ly / 2) for c in coords_v])

    apply_bc!(dr.vx, DirichletBoundaryCondition(nothing, vx_nodes, bc_vx_vals))
    apply_bc!(dr.vy, DirichletBoundaryCondition(nothing, vy_nodes, bc_vy_vals))

    zero_vx_bc = zero(bc_vx_vals)
    zero_vy_bc = zero(bc_vy_vals)

    @info "BCs" n_vx = length(vx_nodes) n_vy = length(vy_nodes) max_vx = maximum(abs, bc_vx_vals) max_vy = maximum(abs, bc_vy_vals)

    # FEM pressure residuals are assembled in weak form:
    #
    #     RP_i = ∫ N_i (-∇⋅v) dΩ
    #
    # The Arrow-Hurwicz pressure step and numerical pressure
    # correction are calibrated for that pointwise residual.  If we feed the weak
    # residual directly into Pnum or P += γP*RP/M_P, the update is scaled by element
    # volume and pressure convergence stalls/refines incorrectly.
    #
    # Use the helper to assemble both:
    #   dr.M_P = ∫ N_i dΩ
    #   γP      = local viscosity-weighted pressure update scale
    # Then γP * RP/M_P matches the pointwise FD-style pressure correction, but
    # adapts the pressure step to viscosity contrasts.
    γP = KernelAbstractions.zeros(backend, Float64, mesh_stokes.nnodesP)
    assemble_viscosity_weighted_pressure_scaling!(
        γP, dr, mesh_stokes, geo_P, element_v, element_P,
        γfact, Δt, backend, workgroup; phases_v = phases_v_cpu,
    )

    Δt = Δt === nothing ? 0.5 / max(abs(ε̇_bg), eps(Float64)) : Float64(Δt)
    time_history = zeros(Float64, nsteps)
    mean_tauII_history = zeros(Float64, nsteps)

    # ---------------------------------------------------------------------------
    # Powell-Hestenes / DYREL-style Stokes solver
    #
    # Structure (mirrors DYREL / JustRelax solver_DR_FD.jl):
    #
    #   Outer Powell-Hestenes loop (itPH)
    #   ├─ assemble physical Rv and RP
    #   ├─ compute err and set inner tolerance ϵ_vel = err * rel_drop
    #   ├─ Inner DR loop for velocity while err > ϵ_vel
    #   │   ├─ copy previous residuals
    #   │   ├─ assemble RP from current v
    #   │   ├─ Pnum = γP·RP/M_P
    #   │   ├─ assemble augmented Rv with Pnum
    #   │   ├─ update v with damped pseudo-transient DR
    #   │   └─ every ncheck: update λmin, λmax, αV, βV
    #   └─ P += γP·RP/M_P  (Arrow-Hurwicz pressure update)
    # ---------------------------------------------------------------------------

    iterMax       = 50_000   # max inner DR iterations per PH step
    total_iterMax = 50_000   # max total inner DR iterations
    nout          = ncheck   # residual / spectral update cadence
    rel_drop0     = 1e-2     # inner convergence: velocity residual drops by this factor
    verbose_PH    = true
    verbose_DR    = false

    @info "Starting PH/DYREL-style Stokes solver" nsteps Δt iterMax total_iterMax nout ϵ_tol

    # λ_min helper (Rayleigh quotient from consecutive residual snapshots).
    # JustRelax computes this with the actual velocity increment, dV = βV*dτV*dVdτ.
    _λmin(step, rate, ΔR, PC) = begin
        dV = step .* rate
        denom = sum(dV .^ 2)
        denom == 0 ? 0.0 : abs(sum(dV .* (ΔR ./ PC))) / denom
    end

    # Damped DYREL/Chebyshev step from spectral step Δτ and damping λmin.
    _cheb(Δτ, λmin, c_fact) = begin
        c = 2 * √(λmin) * c_fact
        (2 * Δτ^2 / (2 + c * Δτ), (2 - c * Δτ) / (2 + c * Δτ))
    end

    el2n_v_cpu = Array(mesh_stokes.el2n)
    out_dir = joinpath(@__DIR__, "output_stokes")
    mkpath(out_dir)
    post = nothing

    for istep in 1:nsteps
        t = istep * Δt
        time_history[istep] = t
        copyto!(dr.P0, dr.P)
        copyto!(dr.T0, dr.T)
        fill!(dr.∂vx∂τ, 0)
        fill!(dr.∂vy∂τ, 0)
        fill!(dr.Rv_x0, 0)
        fill!(dr.Rv_y0, 0)
        @info "Physical time step" istep nsteps t

        FEMTools.assemble_augmented_momentum_jacobian_matrices_atomix!(
            dr.∂Rv_x∂vx, dr.PC_vx, dr.∂Rv_y∂vy, dr.PC_vy,
            dr.vx, dr.vy, dr.P, dr.P0, dr.T, dr.T0,
            mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P, mesh_stokes.nels,
            element_v, element_P,
            phases_v_cpu, phases_P_cpu, τ_old, plastic, nothing, dr.η, G_stokes, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref,
            dr.ηb, Δt, γP, M_P,
            backend, workgroup,
        )
        λmax_vx = maximum(dr.∂Rv_x∂vx ./ dr.PC_vx)
        λmax_vy = maximum(dr.∂Rv_y∂vy ./ dr.PC_vy)
        Δτ_vx   = 2 / √(λmax_vx) * dr.CFL_v
        Δτ_vy   = 2 / √(λmax_vy) * dr.CFL_v
        α_vx, β_vx = _cheb(Δτ_vx, 0.0, dr.c_fact)
        α_vy, β_vy = _cheb(Δτ_vy, 0.0, dr.c_fact)
        @info "Initial momentum preconditioner" λmax_vx λmax_vy Δτ_vx Δτ_vy

        err_min = Inf
        ϵ = Float64(ϵ_tol)
        err = 2 * ϵ
        err_v0 = 1.0
        err_P0 = 1.0
        err_v00 = 1.0
        iter = 0
        rel_drop = rel_drop0

        for itPH in 1:1000

            # ── Outer residuals (fresh momentum + pressure) for convergence check ────
            FEMTools.assemble_pressure_residual_matrices_atomix!(
                dr.RP,
                dr.vx, dr.vy, dr.P, dr.P0, dr.T, dr.T0,
                mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P, mesh_stokes.nels,
                element_v, element_P,
                phases_P_cpu, dr.α, dr.ηb, Δt,
                backend, workgroup,
            )

            FEMTools.assemble_momentum_residual_matrices_atomix!(
                dr.Rv_x, dr.Rv_y,
                dr.vx, dr.vy, dr.P, dr.T, nothing,
                mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, mesh_stokes.nels,
                element_v, element_P,
                phases_v_cpu, τ_old, plastic, nothing, dr.η, G_stokes, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, Δt,
                backend, workgroup,
            )
            FEMTools.apply_dirichlet!(dr.Rv_x, vx_nodes, zero_vx_bc, backend, workgroup)
            FEMTools.apply_dirichlet!(dr.Rv_y, vy_nodes, zero_vy_bc, backend, workgroup)

            # ── Outer convergence check ─────────────────────────────────────────────
            # Compare the FD-like, pointwise pressure residual RP/M_P, not the weak
            # integrated residual RP.
            err_P = norm(dr.RP ./ M_P) / √mesh_stokes.nnodesP
            err_v = max(norm(dr.Rv_x), norm(dr.Rv_y)) / (2 * √mesh_stokes.nnodes)
            if itPH == 1
                err_P0 = err_P + eps(err_P)
                err_v0 = err_v + eps(err_v)
            end
            if itPH == 2
                err_P0 = err_P + eps(err_P)
            end
            err_v_rel = err_v / err_v0
            err_P_rel = err_P / err_P0
            err_abs = max(err_v, err_P)
            err_rel = max(err_v_rel, err_P_rel)
            err = min(err_abs, err_rel)

            isnan(err) && error("NaN detected in outer loop at PH=$itPH")
            err > 1e10 && error("Kaboom! Error > 1e10 in outer loop at PH=$itPH")

            if verbose_PH
                @printf("itPH = %02d iter = %06d err = %.3e abs = %.3e rel = %.3e - norm[Rv=%.3e %.3e, Rp=%.3e %.3e]\n",
                        itPH, iter, err, err_abs, err_rel, err_v, err_v_rel, err_P, err_P_rel)
            end
            err < ϵ && break

            if err > err_min * 1.05
                rel_drop = max(rel_drop * 0.1, 1e-3)
            end
            err_min = min(err_min, err)

            ϵ_vel = err * rel_drop
            itPT  = 0

            # ── Inner DR loop for velocity (P held fixed) ───────────────────────────
            while err > ϵ_vel && itPT ≤ iterMax
                itPT += 1
                iter += 1

                copyto!(dr.Rv_x0, dr.Rv_x)
                copyto!(dr.Rv_y0, dr.Rv_y)

                # Reassemble pressure residual (v is changing → ∇·v changes → RP changes)
                FEMTools.assemble_pressure_residual_matrices_atomix!(
                    dr.RP,
                    dr.vx, dr.vy, dr.P, dr.P0, dr.T, dr.T0,
                    mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P, mesh_stokes.nels,
                    element_v, element_P,
                    phases_P_cpu, dr.α, dr.ηb, Δt,
                    backend, workgroup,
                )

                # Numerical pressure correction, matching JustRelax's pointwise DYREL
                # P_num = γP * RP.  Here RP is weak, so use γP*RP/M_P.
                @. dr.Pnum = γP * dr.RP / M_P

                # Momentum residuals with pressure correction
                FEMTools.assemble_momentum_residual_matrices_atomix!(
                    dr.Rv_x, dr.Rv_y,
                    dr.vx, dr.vy, dr.P, dr.T, dr.Pnum,
                    mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, mesh_stokes.nels,
                    element_v, element_P,
                    phases_v_cpu, τ_old, plastic, nothing, dr.η, G_stokes, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, Δt,
                    backend, workgroup,
                )

                # Enforce Dirichlet BCs on residuals and rates
                FEMTools.apply_dirichlet!(dr.Rv_x, vx_nodes, zero_vx_bc, backend, workgroup)
                FEMTools.apply_dirichlet!(dr.∂vx∂τ, vx_nodes, zero_vx_bc, backend, workgroup)
                FEMTools.apply_dirichlet!(dr.Rv_y, vy_nodes, zero_vy_bc, backend, workgroup)
                FEMTools.apply_dirichlet!(dr.∂vy∂τ, vy_nodes, zero_vy_bc, backend, workgroup)

                # DYREL-style velocity update
                update_rate!(dr.∂vx∂τ, dr.Rv_x, dr.PC_vx, β_vx, mesh_stokes.nnodes)
                update_variable!(dr.vx, dr.∂vx∂τ, -α_vx, mesh_stokes.nnodes)
                update_rate!(dr.∂vy∂τ, dr.Rv_y, dr.PC_vy, β_vy, mesh_stokes.nnodes)
                update_variable!(dr.vy, dr.∂vy∂τ, -α_vy, mesh_stokes.nnodes)

                # Re-pin Dirichlet values
                FEMTools.apply_dirichlet!(dr.vx, vx_nodes, bc_vx_vals, backend, workgroup)
                FEMTools.apply_dirichlet!(dr.vy, vy_nodes, bc_vy_vals, backend, workgroup)

                # Inner convergence check + damped step-size update
                if iszero(iter % nout)
                    err_v_inner = max(norm(dr.Rv_x), norm(dr.Rv_y)) / (2 * √mesh_stokes.nnodes)
                    if iter == nout
                        err_v00 = err_v_inner + eps(err_v_inner)
                    end
                    err = max(err_v_inner / err_v00, err_v_inner)
                    isnan(err) && error("NaN detected in inner loop PH=$itPH PT=$itPT")
                    err > 1e10 && error("Kaboom! Error > 1e10 in inner loop PH=$itPH PT=$itPT")

                    verbose_DR && @printf("  it = %d, iter = %d, err = %.3e\n", itPT, iter, err)

                    λmin_vx = _λmin(α_vx, dr.∂vx∂τ, dr.Rv_x .- dr.Rv_x0, dr.PC_vx)
                    λmin_vy = _λmin(α_vy, dr.∂vy∂τ, dr.Rv_y .- dr.Rv_y0, dr.PC_vy)

                    FEMTools.assemble_augmented_momentum_jacobian_matrices_atomix!(
                        dr.∂Rv_x∂vx, dr.PC_vx, dr.∂Rv_y∂vy, dr.PC_vy,
                        dr.vx, dr.vy, dr.P, dr.P0, dr.T, dr.T0,
                        mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P, mesh_stokes.nels,
                        element_v, element_P,
                        phases_v_cpu, phases_P_cpu, τ_old, plastic, nothing, dr.η, G_stokes, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref,
                        dr.ηb, Δt, γP, M_P,
                        backend, workgroup,
                    )

                    # λmax → Δτ → damped step for velocity.
                    λmax_vx = maximum(dr.∂Rv_x∂vx ./ dr.PC_vx)
                    λmax_vy = maximum(dr.∂Rv_y∂vy ./ dr.PC_vy)
                    Δτ_vx   = 2 / √(λmax_vx) * dr.CFL_v
                    Δτ_vy   = 2 / √(λmax_vy) * dr.CFL_v

                    α_vx, β_vx = _cheb(Δτ_vx, λmin_vx, dr.c_fact)
                    α_vy, β_vy = _cheb(Δτ_vy, λmin_vy, dr.c_fact)
                end

                itPT == iterMax && @printf("  inner: max iters (%d) reached at PH=%d\n", iterMax, itPH)
                iter > total_iterMax && break
            end  # inner PT loop

            # ── Arrow-Hurwicz pressure update (after inner velocity convergence) ─────
            # Same mass-lumped residual as Pnum: pressure is updated from the
            # pointwise divergence residual, not from the weak residual integral.
            @. dr.P += γP * dr.RP / M_P
            FEMTools.remove_pressure_mean!(dr.P, M_P)

            iter > total_iterMax && break
        end  # outer PH loop

        FEMTools.assemble_momentum_residual_matrices_atomix!(
            dr.Rv_x, dr.Rv_y,
            dr.vx, dr.vy, dr.P, dr.T, nothing,
            mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, mesh_stokes.nels,
            element_v, element_P,
            phases_v_cpu, τ_old, plastic, τ, dr.η, G_stokes, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, Δt,
            backend, workgroup,
        )

        P_cpu  = Array(dr.P)
        vx_cpu = Array(dr.vx)
        vy_cpu = Array(dr.vy)
        post = compute_strain_rate_stress_postprocess(
            vx_cpu, vy_cpu,
            el2n_v_cpu,
            Array(geo_v),
            τ,
            element_v,
        )
        mean_tauII_history[istep] = mean(post.tauII)
        copyto!(dr.τxx_old, dr.τxx)
        copyto!(dr.τyy_old, dr.τyy)
        copyto!(dr.τxy_old, dr.τxy)

        vtk_path = joinpath(out_dir, @sprintf("stokes_2D_pure_shear_%04d.vtk", istep))
        write_stokes_vtk(vtk_path, mesh_stokes, coords_v, el2nP_cpu, DoFsP_cpu, P_cpu, vx_cpu, vy_cpu, post)
        @info "Wrote VTK file" vtk_path mean_tauII=mean_tauII_history[istep]
    end  # physical time step loop

    P_cpu  = Array(dr.P)
    vx_cpu = Array(dr.vx)
    vy_cpu = Array(dr.vy)

    # ---------------------------------------------------------------------------
    # Visualisation
    # ---------------------------------------------------------------------------

    # Per-element average pressure (mean of 3 pressure DoFs per element)
    el_P = [mean(P_cpu[DoFsP_cpu[:, i]]) for i in 1:mesh_stokes.nels]

    pts   = [Point2f(c) for c in coords_v]
    polys = [[pts[el2nP_cpu[1, i]], pts[el2nP_cpu[2, i]], pts[el2nP_cpu[3, i]]]
            for i in 1:mesh_stokes.nels]

    fig = Figure(size = (1200, 520))

    # Pressure
    clims_P = extrema(el_P)
    ax1 = Axis(fig[1, 1]; aspect = DataAspect(),
            title = "Pressure  (T7/P1-disc, pure shear)", xlabel = "x", ylabel = "y")
    poly!(ax1, polys; color = el_P, colormap = :vik, colorrange = clims_P, strokewidth = 0)
    Colorbar(fig[1, 2]; colormap = :vik, limits = clims_P,
            label = "P", width = 15, tellheight = false)

    ax2 = Axis(fig[1, 3];
            title = "Mean τII history", xlabel = "time", ylabel = "mean(post.tauII)")
    scatterlines!(ax2, time_history, mean_tauII_history; color = :black, linewidth = 2)

    θ    = LinRange(0, 2π, 300)
    xs_c = cx .+ r_incl .* cos.(θ)
    ys_c = cy .+ r_incl .* sin.(θ)
    lines!(ax1, xs_c, ys_c; color = :white, linewidth = 1.5, linestyle = :dash)

    show_plot && display(fig)
    return (; time = time_history, mean_tauII = mean_tauII_history, post)
end

main()
