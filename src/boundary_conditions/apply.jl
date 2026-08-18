"""
    apply_bc!(rhs, ΓD::DirichletBoundaryCondition)
    apply_bc!(A, ΓD::DirichletBoundaryCondition)
    apply_bc!(A, rhs, ΓD::DirichletBoundaryCondition)

Apply a Dirichlet boundary condition to the right-hand-side vector `rhs` in
place by assigning `ΓD.vals` at `ΓD.DoFs`.

For a matrix alone, `apply_bc!(A, ΓD)` zeroes constrained rows and sets the
constrained diagonal entries to one.

For a matrix system, `apply_bc!(A, rhs, ΓD)` first moves constrained-column
contributions to `rhs`, then zeroes constrained rows and columns of `A`, sets
constrained diagonal entries to one, and assigns constrained RHS values.

All methods mutate their arguments in place and return `nothing`.
"""
apply_bc!(rhs::AbstractVector, ΓD::DirichletBoundaryCondition; workgroup = 256) =
    apply_dirichlet!(rhs, ΓD.DoFs, ΓD.vals, KA.get_backend(rhs), workgroup)

function apply_bc!(A::AbstractMatrix, ΓD::DirichletBoundaryCondition)
    for dof in ΓD.DoFs
        @views A[dof, :] .= 0
        A[dof, dof] = 1
    end
    return nothing
end

function apply_bc!(A::AbstractMatrix, rhs::AbstractVector, ΓD::DirichletBoundaryCondition)
    size(A, 1) == size(A, 2) ||
        throw(DimensionMismatch("Dirichlet system application requires a square matrix"))
    length(rhs) == size(A, 1) ||
        throw(DimensionMismatch("RHS length must match matrix size"))

    dofs = ΓD.DoFs
    vals = ΓD.vals

    # Preserve the effect of constrained columns before those columns are
    # eliminated from the system matrix.
    for i in axes(A, 1)
        i in dofs && continue
        for (dof, val) in zip(dofs, vals)
            rhs[i] -= A[i, dof] * val
        end
    end

    # Symmetric elimination: constrained rows and columns are cleared, the
    # diagonal is pinned to one, and the RHS receives the prescribed value.
    for (dof, val) in zip(dofs, vals)
        @views A[dof, :] .= 0
        @views A[:, dof] .= 0
        A[dof, dof] = 1
        rhs[dof] = val
    end

    return nothing
end
