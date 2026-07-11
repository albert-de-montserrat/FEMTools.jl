import Pkg
Pkg.activate(@__DIR__)

using Printf
using Statistics
using StaticArrays
using LinearAlgebra
using DomainSets
using DomainSets: ×
using KernelAbstractions
using FEMTools
using GLMakie: Figure, Axis, Colorbar, poly!, scatterlines!, lines!, Point2f, DataAspect

const backend = CPU()
const workgroup = 128
const YEAR = 365.25 * 24 * 3600
const KYR = 1.0e3 * YEAR

elastic_buildup_solution(ε̇, t, G, η) = 2 * ε̇ * η * (1 - exp(-G * t / η))

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

function main(;
    mesh_cells = (32, 32),
    lx = 100.0e3,
    ly = 100.0e3,
    endtime_kyr = 100.0,
    η0 = 1.0e22,
    ε̇_bg = 1.0e-14,
    G0 = 1.0e10,
    γfact = 20.0,
    first_dt_kyr = 0.05,
    later_dt_kyr = 1.0,
    dt_switch_kyr = 10.0,
    ncheck = 50,
    ϵ_tol = 1.0e-6,
    niter_PH = 20,
    niter_inner = 5_000,
    total_iterMax = 150_000,
    rel_drop0 = 0.75,
    vtk_every = 0,
    show_plot = true,
    verbose = true,
)
    nx, ny = mesh_cells
    t_end = endtime_kyr * KYR

    η = (Float64(η0), Float64(η0))
    ηb = (Float64(η0 * γfact), Float64(η0 * γfact))
    α = (0.0, 0.0)
    ρ0 = (0.0, 0.0)
    K = (Inf, Inf)
    G = (Float64(G0), Float64(G0))
    g = (0.0, 0.0)
    Tref = 0.0
    plastic = nothing

    Ω = (0.0..lx) × (0.0..ly)
    element_v = ReferenceElement(QuadraticElement{2, 7, Float64})
    element_P = ReferenceElement(LinearElement{2, 3, Float64})

    mesh_v = Mesh(backend, Ω, element_v, (nx, ny))
    mesh_stokes = MixedMesh(mesh_v, element_P)

    @info "Elastic buildup FEM benchmark" nnodes_v=mesh_stokes.nnodes nnodes_P=mesh_stokes.nnodesP nels=mesh_stokes.nels

    ip_v = element_v.integration_points
    NQ_v = length(ip_v.ω)
    NV = length(element_v)
    NP = length(element_P)

    ξq_v = ntuple(q -> SVector(ip_v.ξ[q], ip_v.η[q]), NQ_v)
    ∂N∂ξq_v = ntuple(q -> eval_shape_function_jacobian(element_v, ξq_v[q]), NQ_v)
    ∂N∂ξq_P = ntuple(q -> eval_shape_function_jacobian(element_P, ξq_v[q]), NQ_v)

    geo_v = Vector{NTuple{NQ_v, Tuple{SMatrix{NV, 2, Float64, 2NV}, Float64}}}(undef, mesh_stokes.nels)
    geo_P = Vector{NTuple{NQ_v, Tuple{SMatrix{NP, 2, Float64, 2NP}, Float64}}}(undef, mesh_stokes.nels)

    precompute_geometry!(geo_v, mesh_stokes.coords, mesh_stokes.el2n, ∂N∂ξq_v, ip_v.ω, Val(NV), mesh_stokes.nels)
    precompute_geometry!(geo_P, mesh_stokes.coords, mesh_stokes.el2nP, ∂N∂ξq_P, ip_v.ω, Val(NP), mesh_stokes.nels)

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

    phases_v_cpu = ones(Int, NV, mesh_stokes.nels)
    phases_P_cpu = ones(Int, NP, mesh_stokes.nels)

    vx0 = Float64[ε̇_bg * (c[1] - lx / 2) for c in coords_v]
    vy0 = Float64[-ε̇_bg * (c[2] - ly / 2) for c in coords_v]
    copyto!(dr.vx, vx0)
    copyto!(dr.vy, vy0)

    Γnodes = Array(mesh_v.Γnodes)
    coords = Array(mesh_v.coords)
    tol_x = max(lx, ly) * eps(Float64) * 32
    lr_nodes = Int32[n for n in Γnodes if abs(coords[n][1]) ≤ tol_x || abs(coords[n][1] - lx) ≤ tol_x]
    tb_nodes = Int32[n for n in Γnodes if abs(coords[n][2]) ≤ tol_x || abs(coords[n][2] - ly) ≤ tol_x]

    bc_vx_lr = Float64[ε̇_bg * (coords[n][1] - lx / 2) for n in lr_nodes]
    bc_vy_tb = Float64[-ε̇_bg * (coords[n][2] - ly / 2) for n in tb_nodes]
    zero_vx_lr = zero(bc_vx_lr)
    zero_vy_tb = zero(bc_vy_tb)

    apply_bc!(dr.vx, DirichletBoundaryCondition(nothing, lr_nodes, bc_vx_lr))
    apply_bc!(dr.vy, DirichletBoundaryCondition(nothing, tb_nodes, bc_vy_tb))

    @info "Pure-shear free-slip BCs" n_lr=length(lr_nodes) n_tb=length(tb_nodes) max_vx=maximum(abs, bc_vx_lr) max_vy=maximum(abs, bc_vy_tb)

    h = min(lx / nx, ly / ny)
    ηmax = maximum(η)
    Δτ_V_seed = dr.CFL_v * h^2 / (4 * ηmax)
    fill!(dr.PC_vx, 1 / Δτ_V_seed)
    fill!(dr.PC_vy, 1 / Δτ_V_seed)

    γP = KernelAbstractions.zeros(backend, Float64, mesh_stokes.nnodesP)
    assemble_viscosity_weighted_pressure_scaling!(
        γP, dr, mesh_stokes, geo_P, element_v, element_P,
        γfact, Δt, backend, workgroup; phases_v = phases_v_cpu,
    )

    _λmin(step, rate, ΔR, PC) = begin
        dV = step .* rate
        denom = sum(dV .^ 2)
        denom == 0 ? 0.0 : abs(sum(dV .* (ΔR ./ PC))) / denom
    end

    _cheb(Δτ, λmin, c_fact) = begin
        c = min(2 * sqrt(λmin) * c_fact, 2.0 / Δτ)
        (2 * Δτ^2 / (2 + c * Δτ), (2 - c * Δτ) / (2 + c * Δτ))
    end

    out_dir = joinpath(@__DIR__, "output_stokes")
    vtk_every > 0 && mkpath(out_dir)

    time_kyr = Float64[]
    τyy_max = Float64[]
    τyy_exact = Float64[]
    rel_error = Float64[]
    post = nothing
    t = 0.0
    istep = 0

    @info "Starting elastic buildup solve" endtime_kyr η0 ε̇_bg G0 relaxation_kyr=(η0 / G0 / KYR)

    while t < t_end - eps(t_end)
        istep += 1
        Δt_kyr = t < dt_switch_kyr * KYR ? first_dt_kyr : later_dt_kyr
        Δt = min(Δt_kyr * KYR, t_end - t)
        t += Δt
        τ_ref = max(abs(elastic_buildup_solution(abs(ε̇_bg), t, G0, η0)), eps(Float64))
        Rv_ref = max(τ_ref * h, eps(Float64))
        RP_ref = max(abs(ε̇_bg), eps(Float64))

        copyto!(dr.P0, dr.P)
        copyto!(dr.T0, dr.T)
        fill!(dr.∂vx∂τ, 0)
        fill!(dr.∂vy∂τ, 0)
        fill!(dr.Rv_x0, 0)
        fill!(dr.Rv_y0, 0)

        FEMTools.assemble_augmented_momentum_jacobian_matrices_atomix!(
            dr.∂Rv_x∂vx, dr.PC_vx, dr.∂Rv_y∂vy, dr.PC_vy,
            dr.vx, dr.vy, dr.P, dr.P0, dr.T, dr.T0,
            mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P, mesh_stokes.nels,
            element_v, element_P,
            phases_v_cpu, phases_P_cpu, τ_old, plastic, nothing, dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref,
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
                mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, mesh_stokes.nels,
                element_v, element_P,
                phases_v_cpu, τ_old, plastic, nothing, dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, Δt,
                backend, workgroup,
            )
            FEMTools.apply_dirichlet!(dr.Rv_x, lr_nodes, zero_vx_lr, backend, workgroup)
            FEMTools.apply_dirichlet!(dr.Rv_y, tb_nodes, zero_vy_tb, backend, workgroup)

            FEMTools.assemble_pressure_residual_matrices_atomix!(
                dr.RP,
                dr.vx, dr.vy, dr.P, dr.P0, dr.T, dr.T0,
                mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P, mesh_stokes.nels,
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

            err_v_rel = max(err_v / err_v0, err_v)
            err_P_rel = max(err_P / err_P0, err_P)
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
                    mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P, mesh_stokes.nels,
                    element_v, element_P,
                    phases_P_cpu, dr.α, dr.ηb, Δt,
                    backend, workgroup,
                )

                @. dr.Pnum = γP * dr.RP / dr.M_P

                FEMTools.assemble_momentum_residual_matrices_atomix!(
                    dr.Rv_x, dr.Rv_y,
                    dr.vx, dr.vy, dr.P, dr.T, dr.Pnum,
                    mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, mesh_stokes.nels,
                    element_v, element_P,
                    phases_v_cpu, τ_old, plastic, nothing, dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, Δt,
                    backend, workgroup,
                )

                if do_jac
                    FEMTools.assemble_augmented_momentum_jacobian_matrices_atomix!(
                        dr.∂Rv_x∂vx, dr.PC_vx, dr.∂Rv_y∂vy, dr.PC_vy,
                        dr.vx, dr.vy, dr.P, dr.P0, dr.T, dr.T0,
                        mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P, mesh_stokes.nels,
                        element_v, element_P,
                        phases_v_cpu, phases_P_cpu, τ_old, plastic, nothing, dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref,
                        dr.ηb, Δt, γP, dr.M_P,
                        backend, workgroup,
                    )
                end

                FEMTools.apply_dirichlet!(dr.Rv_x, lr_nodes, zero_vx_lr, backend, workgroup)
                FEMTools.apply_dirichlet!(dr.∂vx∂τ, lr_nodes, zero_vx_lr, backend, workgroup)
                FEMTools.apply_dirichlet!(dr.Rv_y, tb_nodes, zero_vy_tb, backend, workgroup)
                FEMTools.apply_dirichlet!(dr.∂vy∂τ, tb_nodes, zero_vy_tb, backend, workgroup)

                update_rate!(dr.∂vx∂τ, dr.Rv_x, dr.PC_vx, β_vx, mesh_stokes.nnodes)
                update_variable!(dr.vx, dr.∂vx∂τ, -α_vx, mesh_stokes.nnodes)
                update_rate!(dr.∂vy∂τ, dr.Rv_y, dr.PC_vy, β_vy, mesh_stokes.nnodes)
                update_variable!(dr.vy, dr.∂vy∂τ, -α_vy, mesh_stokes.nnodes)

                FEMTools.apply_dirichlet!(dr.vx, lr_nodes, bc_vx_lr, backend, workgroup)
                FEMTools.apply_dirichlet!(dr.vy, tb_nodes, bc_vy_tb, backend, workgroup)

                if do_jac
                    err_v_inner = (norm(dr.Rv_x) + norm(dr.Rv_y)) / (2 * sqrt(mesh_stokes.nnodes) * Rv_ref)
                    if iter == 1
                        err_v00 = err_v_inner + eps(err_v_inner)
                    end
                    err = max(err_v_inner / err_v00, err_v_inner)
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
        post = compute_strain_rate_stress_postprocess(
            vx_cpu, vy_cpu,
            el2n_v_cpu,
            Array(geo_v),
            phases_v_cpu,
            τ_old,
            dr.η, G, Δt,
            element_v,
        )
        update_old_stress_from_cells!(τ_old, post, el2n_v_cpu, mesh_stokes.nnodes)

        push!(time_kyr, t / KYR)
        push!(τyy_max, maximum(abs, post.τyy))
        push!(τyy_exact, elastic_buildup_solution(ε̇_bg, t, G0, η0))
        push!(rel_error, abs(τyy_max[end] - τyy_exact[end]) / max(abs(τyy_exact[end]), eps(Float64)))

        verbose && @printf("  τyy_max = %.6e Pa, exact = %.6e Pa, relerr = %.3e\n",
            τyy_max[end], τyy_exact[end], rel_error[end])

        if vtk_every > 0 && (mod(istep, vtk_every) == 0 || t ≥ t_end - eps(t_end))
            vtk_path = joinpath(out_dir, @sprintf("stokes_2D_elastic_buildup_%04d.vtk", istep))
            write_stokes_vtk(vtk_path, mesh_stokes, coords_v, el2nP_cpu, DoFsP_cpu, P_cpu, vx_cpu, vy_cpu, post)
            @info "Wrote VTK file" vtk_path
        end
    end

    isnothing(post) && error("No time steps were executed")

    pts = [Point2f(c[1] / 1e3, c[2] / 1e3) for c in coords_v]
    polys = [[pts[el2nP_cpu[1, i]], pts[el2nP_cpu[2, i]], pts[el2nP_cpu[3, i]]]
             for i in 1:mesh_stokes.nels]

    fig = Figure(size = (1200, 520))
    ax1 = Axis(fig[1, 1];
        title = "Elastic stress buildup",
        xlabel = "time (kyr)",
        ylabel = "max |τyy| (MPa)",
    )
    scatterlines!(ax1, time_kyr, τyy_max ./ 1e6; color = :black, linewidth = 2)
    lines!(ax1, time_kyr, τyy_exact ./ 1e6; color = :red, linewidth = 2)

    τyy_mpa = post.τyy ./ 1e6
    clims = extrema(τyy_mpa)
    ax2 = Axis(fig[1, 2];
        aspect = DataAspect(),
        title = "Final τyy (MPa)",
        xlabel = "x (km)",
        ylabel = "y (km)",
    )
    poly!(ax2, polys; color = τyy_mpa, colormap = :vik, colorrange = clims, strokewidth = 0)
    Colorbar(fig[1, 3]; colormap = :vik, limits = clims, label = "τyy (MPa)", width = 15, tellheight = false)

    show_plot && display(fig)

    return (;
        time_kyr,
        τyy_max,
        τyy_exact,
        rel_error,
        post,
        figure = fig,
    )
end

main()
