"""
    solver!(dr, Δt, mesh, geo, element, Γ_dofs, Γ_zero, Γ_vals, backend, workgroup;
            ncheck = 100, Tref = 273)

Run the pseudo-transient dynamic-relaxation (DR) solver on `dr` for one time
step of size `Δt`.

`Γ_dofs` is an integer array of constrained DOF indices. `Γ_zero` and `Γ_vals`
are float arrays of the same length: zero values (for zeroing the residual and
rate at constrained nodes) and the prescribed Dirichlet values respectively.
`backend` and `workgroup` are forwarded to all KernelAbstractions kernel
launches. `ncheck` controls how often the spectral estimates and convergence
criterion are recomputed (every `ncheck` PT iterations).
`Tref` is the reference temperature used in the density equation of state.

The solver modifies `dr.T` in-place. `dr.T0` must be set to the temperature at
the previous time step before calling.
"""
function solver!(dr::ThermalDiffusionDR, Δt, mesh, geo, element,
                 Γ_dofs, Γ_zero, Γ_vals,
                 backend, workgroup;
                 ncheck = 100,
                 Tref = eltype(dr.T)(273))
    (; R, R0, ∂R∂T, PC, T, T0, ∂T∂τ,
       phases, k, Cp, ρ0, α, K, P, source,
       CFL, c_fact, ϵ) = dr

    # α_dr: pseudo-transient step (distinct from thermal expansivity α in dr)
    α_dr = zero(eltype(R))
    β    = zero(eltype(R))
    nr0  = zero(eltype(R))

    for it in 1:10_000
        do_∂R∂T = (mod(it, ncheck) == 0) || (it == 1)
        do_∂R∂T && copyto!(R0, R)

        assemble_diffusion_matrices_atomix!(
            R, ∂R∂T, PC, T, T0, mesh.el2n, geo, mesh.nels,
            element, phases, k, Cp, ρ0, α, K, P, Δt, source, Tref, do_∂R∂T,
            backend, workgroup,
        )

        # Constrain residual and rate *before* the update so that boundary
        # reaction forces do not corrupt the λ_min spectral estimate.
        apply_dirichlet!(R,    Γ_dofs, Γ_zero, backend, workgroup)
        apply_dirichlet!(∂T∂τ, Γ_dofs, Γ_zero, backend, workgroup)

        update_rate_kernel!(backend, workgroup)(∂T∂τ, R, PC, β; ndrange = mesh.nnodes)
        update_variable_kernel!(backend, workgroup)(T, ∂T∂τ, α_dr; ndrange = mesh.nnodes)

        apply_dirichlet!(T, Γ_dofs, Γ_vals, backend, workgroup)

        if do_∂R∂T || it == 1
            nr = norm(R)
            it == 1 && (nr0 = max(nr, eps(nr)))   # guard against exact-zero initial residual
            isnan(nr / nr0) && error("NaNs at PT iter $it")

            λmax  = maximum(∂R∂T ./ PC)
            Δτ    = 2 / √(λmax) * CFL
            denom = sum((Δτ .* ∂T∂τ) .^ 2)
            λmin  = (it == 1 || denom == 0) ? zero(eltype(R)) :
                    abs(sum(Δτ .* ∂T∂τ .* ((R .- R0) ./ PC))) / denom
            c    = 2 * √(λmin) * c_fact
            α_dr = 2 * Δτ^2 / (2 + c * Δτ)
            β    = (2 - c * Δτ) / (2 + c * Δτ)
            @printf("  PT %05d  res = %6.2e\n", it, nr / nr0)
            nr / nr0 < ϵ && break
        end
    end
    return nothing
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
