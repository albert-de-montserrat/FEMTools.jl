"""
    solver!(dr::LithostaticPressureDR, mesh, geo, element,
            Γ_dofs, Γ_zero, Γ_vals, backend, workgroup;
            ncheck=100, verbose=true, Tref=273, g=SVector(0, -9.81))

Run the pseudo-transient dynamic-relaxation (DR) solver for the
lithostatic-pressure problem `∫ ∇P·∇v dΩ = ∫ ρ(T) g·∇v dΩ`.

`dr.T` must be set to the current temperature field before calling.
`Γ_dofs`, `Γ_zero`, `Γ_vals` enforce Dirichlet boundary conditions on `P`.
`Tref` and `g` control the density equation of state and body-force vector.
`g` must be an `SVector` whose length matches the spatial dimension of the
mesh (e.g. `SVector(0, -9.81)` for 2-D, `SVector(0, 0, -9.81)` for 3-D).
The default value `SVector(0, -9.81)` is only appropriate for 2-D problems.
`ncheck` controls how often spectral estimates and convergence are recomputed.
Set `verbose = false` to suppress per-iteration residual output.

Modifies `dr.P` in-place. Returns `nothing`.
"""
function solver!(dr::LithostaticPressureDR, mesh, geo, element,
                 Γ_dofs, Γ_zero, Γ_vals,
                 backend, workgroup;
                 ncheck = 100,
                 verbose = true,
                 Tref = eltype(dr.P)(273),
                 g = SVector(zero(eltype(dr.P)), -eltype(dr.P)(9.81)))
    (; R, R0, ∂R∂P, PC, P, ∂P∂τ, T,
       phases, ρ0, α, K,
       CFL, c_fact, ϵ) = dr

    α_dr = zero(eltype(R))
    β    = zero(eltype(R))
    nr0  = zero(eltype(R))

    for it in 1:10_000
        do_∂R∂P = (mod(it, ncheck) == 0) || (it == 1)
        do_∂R∂P && copyto!(R0, R)

        assemble_lithostatic_pressure_matrices_atomix!(
            R, ∂R∂P, PC, T, P, mesh.el2n, geo, mesh.nels,
            element, phases, ρ0, α, K, Tref, g, do_∂R∂P,
            backend, workgroup,
        )

        apply_dirichlet!(R,    Γ_dofs, Γ_zero, backend, workgroup)
        apply_dirichlet!(∂P∂τ, Γ_dofs, Γ_zero, backend, workgroup)

        update_rate_kernel!(backend, workgroup)(∂P∂τ, R, PC, β; ndrange = mesh.nnodes)
        update_variable_kernel!(backend, workgroup)(P, ∂P∂τ, α_dr; ndrange = mesh.nnodes)

        apply_dirichlet!(P, Γ_dofs, Γ_vals, backend, workgroup)

        if do_∂R∂P
            nr = norm(R)
            it == 1 && (nr0 = nr)
            isnan(nr / nr0) && error("NaNs at PT iter $it")

            λmax  = maximum(∂R∂P ./ PC)
            Δτ    = 2 / √(λmax) * CFL
            denom = sum((Δτ .* ∂P∂τ) .^ 2)
            λmin  = (it == 1 || denom == 0) ? zero(eltype(R)) :
                    abs(sum(Δτ .* ∂P∂τ .* ((R .- R0) ./ PC))) / denom
            c    = 2 * √(λmin) * c_fact
            α_dr = 2 * Δτ^2 / (2 + c * Δτ)
            β    = (2 - c * Δτ) / (2 + c * Δτ)
            verbose && @printf("  PT %05d  res = %6.2e\n", it, nr / nr0)
            nr / nr0 < ϵ && break
        end
    end
    return nothing
end
