"""
    assemble_velocity_mass!(M_V, mesh, geo_v, element_v, backend, workgroup)

Assemble the positive diagonal velocity mass

    M_V[i] = Σₑ ∫_{Ωₑ} N_i² dΩ

on the velocity nodes of `mesh`. Unlike row-sum lumping (`∫N_i dΩ`), the
squared-basis diagonal remains positive for higher-order elements whose shape
functions may change sign or integrate to zero.
"""
function assemble_velocity_mass!(
    M_V,
    mesh::MixedMesh{2},
    geo_v,
    element_v::ReferenceElement,
    backend,
    workgroup,
)
    length(M_V) == mesh.nnodes ||
        throw(DimensionMismatch("M_V length must match the velocity node count"))
    return assemble_velocity_mass!(
        M_V, mesh.el2n, geo_v, mesh.nels, element_v, backend, workgroup,
    )
end

function assemble_velocity_mass!(
    M_V,
    el2n_v,
    geo_v,
    nels,
    element_v::ReferenceElement{TV},
    backend,
    workgroup,
) where {TV <: AbstractElement{2, NV}} where {NV}
    NqV = shape_function_values(element_v)
    fill!(M_V, 0)
    velocity_mass_kernel!(backend, workgroup)(
        M_V, el2n_v, geo_v, NqV, Val(NV);
        ndrange = nels,
    )
    KA.synchronize(backend)
    return M_V
end

@kernel function velocity_mass_kernel!(
    M_V,
    @Const(el2n_v),
    @Const(geo_v),
    NqV,
    ::Val{NV},
) where {NV}
    iel = @index(Global)
    local_nodes_v = local_nodes_of(el2n_v, iel, Val(NV))
    geo_v_el = geo_v[iel]

    for q in eachindex(geo_v_el)
        _, dΩ = geo_v_el[q]
        Nv = NqV[q]
        for a in 1:NV
            inod = local_nodes_v[a]
            Atomix.@atomic :monotonic M_V[inod] += abs2(Nv[a]) * dΩ
        end
    end
end
