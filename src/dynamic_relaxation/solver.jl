"""
    AbstractDRProblem

Supertype for solver states advanced by the pseudo-transient
dynamic-relaxation (DR) scheme in [`solve_dynamic_relaxation!`](@ref).

A subtype supplies its work arrays through [`dr_fields`](@ref) and its name
through [`dr_name`](@ref), and carries the scalar solver parameters `CFL`,
`c_fact`, and `ϵ` as fields of those names.
"""
abstract type AbstractDRProblem end

"""
    dr_fields(problem) -> (; R, R0, ∂R∂u, PC, u, ∂u∂τ)

Return the dynamic-relaxation work arrays of `problem` under the names the
solver uses: current residual, residual snapshot, Jacobian row-sum estimate,
diagonal preconditioner, the unknown being solved for, and its
pseudo-transient rate.
"""
function dr_fields end

"""
    dr_name(problem) -> String

Return the lower-case name of the physical problem, used in solver
diagnostics. `"thermal diffusion"` renders as `"thermal diffusion
preconditioner produced invalid λmax"` and as `"Thermal diffusion DR solver
did not converge"`.
"""
function dr_name end

"""
    solve_dynamic_relaxation!(problem, assemble!, nnodes,
                              Γ_dofs, Γ_zero, Γ_vals, backend, workgroup;
                              ncheck=100, iterMax=10_000, verbose=true)

Advance `problem` to steady state with a Chebyshev-accelerated
pseudo-transient iteration, and return `nothing` on convergence.

`assemble!(compute_jacobian::Bool)` fills the problem's residual, and its
Jacobian diagnostics when asked. Spectral estimates and convergence are
recomputed every `ncheck` iterations; the Jacobian is assembled only on those
iterations. Dirichlet nodes `Γ_dofs` are pinned to `Γ_vals`, while the
residual and rate are constrained with `Γ_zero` before each update so that
boundary reaction forces do not corrupt the λ_min estimate.

Throws if the iteration produces NaNs, if the preconditioner admits no valid
λmax, or if `iterMax` iterations pass without reaching the tolerance `ϵ`.
"""
function solve_dynamic_relaxation!(
        problem::AbstractDRProblem, assemble!, nnodes,
        Γ_dofs, Γ_zero, Γ_vals, backend, workgroup;
        ncheck = 100,
        iterMax = 10_000,
        verbose = true,
    )
    (; R, R0, ∂R∂u, PC, u, ∂u∂τ) = dr_fields(problem)
    name = dr_name(problem)

    # α_dr, β: Chebyshev step and momentum, distinct from any material property
    # the problem happens to call α or β.
    α_dr = zero(eltype(R))
    β = zero(eltype(R))
    nr0 = zero(eltype(R))
    last_rel = NaN
    λmax = zero(eltype(R))

    for it in 1:iterMax
        do_jacobian = (mod(it, ncheck) == 0) || (it == 1)
        do_jacobian && copyto!(R0, R)

        assemble!(do_jacobian)

        # Constrain residual and rate *before* the update so that boundary
        # reaction forces do not corrupt the λ_min spectral estimate.
        apply_dirichlet!(R, Γ_dofs, Γ_zero, backend, workgroup)
        apply_dirichlet!(∂u∂τ, Γ_dofs, Γ_zero, backend, workgroup)

        do_jacobian && (λmax = _checked_λmax(∂R∂u, PC, name))

        update_rate_kernel!(backend, workgroup)(∂u∂τ, R, PC, β; ndrange = nnodes)
        update_variable_kernel!(backend, workgroup)(u, ∂u∂τ, α_dr; ndrange = nnodes)

        apply_dirichlet!(u, Γ_dofs, Γ_vals, backend, workgroup)

        if do_jacobian
            nr = norm(R)
            it == 1 && (nr0 = max(nr, eps(nr)))   # guard against exact-zero initial residual
            isnan(nr / nr0) && error("NaNs at PT iter $it")

            Δτ = 2 / √(λmax) * problem.CFL
            denom = sum((Δτ .* ∂u∂τ) .^ 2)
            λmin = (it == 1 || denom == 0) ? zero(eltype(R)) :
                abs(sum(Δτ .* ∂u∂τ .* ((R .- R0) ./ PC))) / denom
            c = 2 * √(λmin) * problem.c_fact
            α_dr = 2 * Δτ^2 / (2 + c * Δτ)
            β = (2 - c * Δτ) / (2 + c * Δτ)
            last_rel = nr / nr0
            verbose && @printf("  PT %05d  res = %6.2e\n", it, last_rel)
            last_rel < problem.ϵ && return nothing
        end
    end
    error("$(uppercasefirst(name)) DR solver did not converge after $iterMax pseudo-transient iterations (relative residual = $last_rel)")
end

# Geometry, element, and boundary arrays a `Mesh`/`DirichletBoundaryCondition`
# pair supplies to the low-level `solver!` methods.
@inline function _mesh_solver_arguments(mesh::Mesh, bc::DirichletBoundaryCondition, workgroup)
    isnothing(mesh.geometry) && throw(ArgumentError("mesh has no geometry; construct it with Mesh(backend, coords, el2n, element)"))
    return (
        mesh, mesh.geometry, mesh.element,
        bc.DoFs, bc.zero_vals, bc.vals,
        KA.get_backend(mesh.coords), workgroup,
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
