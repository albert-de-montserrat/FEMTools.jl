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

set_boundary_condition!(bcs::DirichletBoundaryCondition, args::AbstractArray) = set_boundary_condition!(bcs, (args,))

@generated function set_boundary_condition!(bcs::DirichletBoundaryCondition, args::NTuple{N, AbstractArray}) where N
    quote 
        for i in eachindex(bcs.dofs)
            dofᵢ          = bcs.dofs[i]
            valᵢ          = bcs.values[i]

            for j in axes(bcs.dofs, 2)
                # set diagonal entries to 0
                Base.@nexprs $N I -> begin
                    @inline
                    arg = args[I]
                    if arg isa AbstractMatrix
                        arg[dofᵢ, j] = 0.0
                    end
                end
            end

            Base.@nexprs $N I -> begin
                @inline
                arg = args[I]
                # set diagonal entries to 1
                if arg isa AbstractMatrix
                    arg[dofᵢ, dofᵢ] = 1.0
                
                # set entries of the vector to the prescribed value
                elseif arg isa AbstractVector
                    arg[dofᵢ]       = valᵢ

                else
                    error("Unsupported type in args")
                end
            end
        end
    end
end