"""
    solver_colored!(dr, Δt, mesh, geo, element, Γ_dofs, Γ_zero, Γ_vals,
                    el_groups, backend, workgroup;
                    ncheck = 100, iterMax = 10_000, verbose = true, Tref = 273)

Run the pseudo-transient dynamic-relaxation (DR) solver on `dr` for one time
step of size `Δt` using graph-colored assembly.

`Γ_dofs` is an integer array of constrained DOF indices. `Γ_zero` and `Γ_vals`
are float arrays of the same length: zero values (for zeroing the residual and
rate at constrained nodes) and the prescribed Dirichlet values respectively.
`el_groups` is a vector of device integer arrays, one per color, produced by
`generate_element_groups`. Elements in the same color group share no nodes,
so the colored assembly kernels can scatter into `R`, `∂R∂T`, and `PC` without
atomics.

The solver modifies `dr.T` in-place. `dr.T0` must be set to the temperature at
the previous time step before calling. Returns `(iterations, residuals)`, where
both vectors contain the convergence checks performed during the solve.
"""
function solver_colored!(dr::ThermalDiffusionDR, Δt, mesh, geo, element,
                         Γ_dofs, Γ_zero, Γ_vals,
                         el_groups, backend, workgroup;
                         ncheck = 100,
                         iterMax = 10_000,
                         verbose = true,
                         Tref = eltype(dr.T)(273))
    (; R, R0, ∂R∂T, PC, T, T0, ∂T∂τ,
       phases, k, Cp, ρ0, α, K, P, source,
       CFL, c_fact, ϵ) = dr

    # α_dr: pseudo-transient step (distinct from thermal expansivity α in dr)
    α_dr = zero(eltype(R))
    β    = zero(eltype(R))
    nr0  = zero(eltype(R))

    last_rel = NaN
    λmax = zero(eltype(R))
    iterations = Int[]
    residuals = Float64[]

    for it in 1:iterMax
        do_∂R∂T = (mod(it, ncheck) == 0) || (it == 1)
        do_∂R∂T && copyto!(R0, R)

        assemble_diffusion_matrices_colored!(
            R, ∂R∂T, PC, T, T0, mesh.el2n, geo, el_groups,
            element, phases, k, Cp, ρ0, α, K, P, Δt, source, Tref,
            backend, workgroup; compute_jacobian = do_∂R∂T,
        )

        # Constrain residual and rate *before* the update so that boundary
        # reaction forces do not corrupt the λ_min spectral estimate.
        apply_dirichlet!(R,    Γ_dofs, Γ_zero, backend, workgroup)
        apply_dirichlet!(∂T∂τ, Γ_dofs, Γ_zero, backend, workgroup)

        do_∂R∂T && (λmax = _checked_λmax(∂R∂T, PC, "thermal diffusion"))

        update_rate_kernel!(backend, workgroup)(∂T∂τ, R, PC, β; ndrange = mesh.nnodes)
        update_variable_kernel!(backend, workgroup)(T, ∂T∂τ, α_dr; ndrange = mesh.nnodes)

        apply_dirichlet!(T, Γ_dofs, Γ_vals, backend, workgroup)

        if do_∂R∂T || it == 1
            nr = norm(R)
            it == 1 && (nr0 = max(nr, eps(nr)))   # guard against exact-zero initial residual
            isnan(nr / nr0) && error("NaNs at PT iter $it")

            Δτ    = 2 / √(λmax) * CFL
            denom = sum((Δτ .* ∂T∂τ) .^ 2)
            λmin  = (it == 1 || denom == 0) ? zero(eltype(R)) :
                    abs(sum(Δτ .* ∂T∂τ .* ((R .- R0) ./ PC))) / denom
            c    = 2 * √(λmin) * c_fact
            α_dr = 2 * Δτ^2 / (2 + c * Δτ)
            β    = (2 - c * Δτ) / (2 + c * Δτ)
            last_rel = nr / nr0
            push!(iterations, it)
            push!(residuals, last_rel)
            verbose && @printf("  PT %05d  res = %6.2e\n", it, last_rel)
            last_rel < ϵ && return iterations, residuals
        end
    end
    error("Thermal diffusion DR solver did not converge after $iterMax pseudo-transient iterations (relative residual = $last_rel)")
end
