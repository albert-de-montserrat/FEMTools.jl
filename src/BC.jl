struct DirichletBoundaryCondition{T}
    dofs::Vector{Int64}
    values::Vector{T}

    function DirichletBoundaryCondition(bc_dof, bc_val::Vararg{Pair, N}) where {T, N}
        dof_array, value_array = generate_bc_arrays(bc_dof, bc_val...)
        new{eltype(value_array)}(dof_array, value_array)
    end
end

function generate_bc_arrays(bc_dof, bc_val::Vararg{Pair, N}) where {N}
    value = ntuple(Val(N)) do i
        generate_bc_array(bc_dof, bc_val[i])
    end
    value_array = vcat(value...)
    dof_array = reduce(vcat, values(bc_dof))
    return dof_array, value_array
end

function generate_bc_array(bc_dof, bc_val::Pair{Symbol, T}) where {T<:Number}
    key    = bc_val.first
    value  = bc_val.second
    dofs   = bc_dof[key]
    return  fill(value, length(dofs))
end

