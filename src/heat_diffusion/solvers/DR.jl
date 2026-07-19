"""
    solver!(dr::ThermalDiffusionDR, Δt, mesh, geo, element,
            Γ_dofs, Γ_zero, Γ_vals, backend, workgroup; kwargs...)

Run the pseudo-transient dynamic-relaxation (DR) solver on `dr` for one time
step of size `Δt` using explicitly supplied geometry, element, boundary arrays,
backend, and workgroup size.

`ncheck` controls how often spectral
estimates and convergence are recomputed, and `iterMax` limits the number of
pseudo-transient iterations. Set `verbose=false` to suppress residual output.
`Tref` is the reference temperature used by the density equation of state.

The solver modifies `dr.T` in-place. `dr.T0` must be set to the temperature at
the previous physical time step before calling.

"""
function solver!(dr::ThermalDiffusionDR, Δt, mesh, geo, element,
                 Γ_dofs, Γ_zero, Γ_vals,
                 backend, workgroup;
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

    for it in 1:iterMax
        do_∂R∂T = (mod(it, ncheck) == 0) || (it == 1)
        do_∂R∂T && copyto!(R0, R)

        assemble_diffusion_matrices_atomix!(
            R, ∂R∂T, PC, T, T0, mesh.el2n, geo, mesh.nels,
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
            verbose && @printf("  PT %05d  res = %6.2e\n", it, last_rel)
            last_rel < ϵ && return nothing
        end
    end
    error("Thermal diffusion DR solver did not converge after $iterMax pseudo-transient iterations (relative residual = $last_rel)")
end

"""
    solver!(dr, Δt, mesh, bc; workgroup=256, kwargs...)

Solve one thermal-diffusion time step using the element and geometry stored in
`mesh` and the prescribed values in `bc`. The backend is inferred from
`mesh.coords`; remaining keywords are forwarded to the low-level solver.
"""
function solver!(
    dr::ThermalDiffusionDR,
    Δt,
    mesh::Mesh,
    bc::DirichletBoundaryCondition;
    workgroup = 256,
    kwargs...,
)
    isnothing(mesh.geometry) && throw(ArgumentError("mesh has no geometry; construct it with Mesh(backend, coords, el2n, element)"))
    return solver!(
        dr, Δt, mesh, mesh.geometry, mesh.element,
        bc.DoFs, bc.zero_vals, bc.vals, KA.get_backend(mesh.coords), workgroup;
        kwargs...,
    )
end

# ---------------------------------------------------------------------------
# Dirichlet enforcement
# ---------------------------------------------------------------------------

@kernel function dirichlet_kernel!(v, @Const(dofs), @Const(vals))
    i = @index(Global)
    v[dofs[i]] = vals[i]
end

"""
    apply_dirichlet!(v, dofs, vals, backend, workgroup)

Overwrite `v[dofs[i]] = vals[i]` for all `i` on the target backend.

Used inside `solver!` to pin boundary nodes at Dirichlet values before and
after pseudo-transient updates. Returns `nothing`.
"""
function apply_dirichlet!(v, dofs, vals, backend, workgroup)
    isempty(dofs) && return nothing
    dirichlet_kernel!(backend, workgroup)(v, dofs, vals; ndrange = length(dofs))
    KA.synchronize(backend)
    return nothing
end

# ---------------------------------------------------------------------------
# Pseudo-transient update kernels
# ---------------------------------------------------------------------------

"""
    update_rate_kernel!(∂u∂τ, R, PC, β)

KernelAbstractions kernel for the Chebyshev-accelerated pseudo-transient rate update.

Sets `∂u∂τ[i] = R[i] / PC[i] + β * ∂u∂τ[i]`. With `β = 0` this is a plain
preconditioned gradient-descent step; with `β > 0` it is the momentum term of
the Chebyshev recurrence. `PC` is the diagonal preconditioner (units of
stiffness); `R` is the assembled residual.
"""
@kernel function update_rate_kernel!(∂u∂τ, @Const(R), @Const(PC), β)
    i = @index(Global)
    ∂u∂τ[i] = R[i] / PC[i] + β * ∂u∂τ[i]
end

"""
    update_variable_kernel!(u, ∂u∂τ, α_dr)

KernelAbstractions kernel that advances the solution by one pseudo-transient step.

Sets `u[i] += α_dr * ∂u∂τ[i]`, where `∂u∂τ` holds the current rate (as
produced by `update_rate_kernel!`) and `α_dr` is the Chebyshev step size.
"""
@kernel function update_variable_kernel!(u, @Const(∂u∂τ), α_dr)
    i = @index(Global)
    u[i] += α_dr * ∂u∂τ[i]
end
