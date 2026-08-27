# ---------------------------------------------------------------------------
# Stress post-processing for the cantilever: pressure P and the deviatoric
# stresses τxx, τyy, τxy recovered at the nodes from the converged total
# displacement field.
#
# Stresses are evaluated at the quadrature points (where they are most
# accurate) and projected to the nodes with a volume-weighted (lumped L2)
# average:   τ_node = Σ_e Σ_q N_i τ_q dΩ / Σ_e Σ_q N_i dΩ.
# Strains are taken w.r.t. the *reference* configuration, consistent with the
# small-strain incremental scheme of the solver (evaluating on the advected
# configuration instead would differ at O(ε), the order the scheme neglects).
# In plane strain σzz = λ tr(ε) ≠ 0 and enters the pressure.
# ---------------------------------------------------------------------------
using Atomix

# runs the DR simulation and leaves its result in `state`
include(joinpath(@__DIR__, "..", "2D_Elasticiy_DR_KA", "2D_Elasticiy_DR_KA.jl"))

@kernel function stress_kernel!(τxx, τyy, τxy, P, wt, @Const(Ux), @Const(Uy), @Const(el2n), @Const(geo), λ, μ, Nq, ::Val{N}) where N
    iel = @index(Global)
    local_nodes = local_nodes_of(el2n, iel, Val(N))
    geo_el = geo[iel]
    uxloc = SVector{N}(ntuple(i -> Ux[local_nodes[i]], Val(N)))
    uyloc = SVector{N}(ntuple(i -> Uy[local_nodes[i]], Val(N)))
    for q in eachindex(geo_el)
        ∂N∂x, dΩ = geo_el[q]
        Nv = Nq[q]

        ∇ux = ∂N∂x' * uxloc
        ∇uy = ∂N∂x' * uyloc
        εxx = ∇ux[1]
        εyy = ∇uy[2]
        γxy = ∇ux[2] + ∇uy[1]
        tr  = εxx + εyy
        σxx = λ * tr + 2μ * εxx
        σyy = λ * tr + 2μ * εyy
        σzz = λ * tr                   # plane strain
        σxy = μ * γxy
        p   = -(σxx + σyy + σzz) / 3

        for (i, inod) in enumerate(local_nodes)
            w = Nv[i] * dΩ
            Atomix.@atomic :monotonic τxx[inod] += (σxx + p) * w   # deviatoric
            Atomix.@atomic :monotonic τyy[inod] += (σyy + p) * w
            Atomix.@atomic :monotonic τxy[inod] += σxy * w
            Atomix.@atomic :monotonic P[inod]   += p * w
            Atomix.@atomic :monotonic wt[inod]  += w
        end
    end
end

function postprocess_stresses(state)
    (; mesh, element, el2n, Ux, Uy, λ, μ) = state
    N = length(element)
    Nq = shape_function_values(element)

    # strains w.r.t. the reference configuration (mesh.coords is unadvected)
    geo_ref = precompute_geometry(to_backend(mesh.coords), el2n, mesh.nels, element)

    τxx = KA.zeros(backend, FP, mesh.nnodes)
    τyy = KA.zeros(backend, FP, mesh.nnodes)
    τxy = KA.zeros(backend, FP, mesh.nnodes)
    P   = KA.zeros(backend, FP, mesh.nnodes)
    wt  = KA.zeros(backend, FP, mesh.nnodes)

    stress_kernel!(backend, workgroup)(τxx, τyy, τxy, P, wt, Ux, Uy, el2n, geo_ref, λ, μ, Nq, Val(N); ndrange = mesh.nels)
    KA.synchronize(backend)

    τxx ./= wt
    τyy ./= wt
    τxy ./= wt
    P   ./= wt
    return τxx, τyy, τxy, P
end

τxx, τyy, τxy, P = postprocess_stresses(state)

# sanity: maximum bending stress σxx at the clamped end vs beam theory,
# σxx_max ≈ M c / I with M = q Lx² / 2, q = |by| Ly, c = Ly/2, I = Ly³/12
let
    (; Lx, Ly, by_total) = state
    σxx = Array(τxx) .- Array(P)
    σxx_beam = abs(by_total) * Ly * Lx^2 / 2 * (Ly / 2) / (Ly^3 / 12)
    @printf("σxx extrema = (%.4e, %.4e)   beam-theory root stress ≈ ±%.4e\n",
            minimum(σxx), maximum(σxx), σxx_beam)
    @printf("τxy extrema = (%.4e, %.4e)   beam-theory max shear  ≈ %.4e\n",
            minimum(Array(τxy)), maximum(Array(τxy)),
            1.5 * abs(by_total) * Ly * Lx / Ly)
end

# render on the advected grid: deviatoric stresses (symmetric diverging
# colormap) and the vertical displacement
let
    coords_def = Array(state.coords)
    vertices = [GLMakie.Point2f(p[1], p[2]) for p in coords_def]
    faces = plot_triangles(state.mesh, state.element)
    fields = (("τxx", τxx, :RdBu), ("τyy", τyy, :RdBu), ("τxy", τxy, :RdBu), ("Uy", state.Uy, :viridis))
    fig = Figure(size = (1100, 900))
    for (k, (name, field, cmap)) in enumerate(fields)
        c = Array(field)
        colorrange = cmap === :RdBu ? (-1, 1) .* maximum(abs.(c)) : extrema(c)
        ax = Axis(fig[k, 1]; xlabel = "x", ylabel = "y", title = name, aspect = DataAspect())
        m = GLMakie.mesh!(ax, vertices, faces; color = c, colormap = cmap, colorrange, shading = NoShading)
        Colorbar(fig[k, 2], m)
    end
    display(fig)
end
