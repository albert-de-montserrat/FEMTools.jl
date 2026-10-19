# FEMTools setup boundary. The solver receives only standard containers and
# prepared arrays; it does not call package accessors or shape/geometry methods.
function standalone_dyrel_inputs(dr, mesh, bc)
    cache = mesh.geometry
    backend = KernelAbstractions.get_backend(mesh.coords)
    components(a) = (a.x, a.y)
    state = (; v = components(dr.v), rate = components(dr.∂v∂τ),
        Rv = components(dr.Rv), Rv0 = components(dr.Rv0),
        rowsums = components(dr.∂Rv∂v), PC = components(dr.PC_v),
        dr.P, dr.RP, dr.Pnum, dr.M_P, dr.γP, dr.CFL_v, dr.c_fact)
    mesh_data = (; mesh.el2n, mesh.DoFsP, mesh.nels, mesh.nnodes, mesh.nnodesP,
        cache.geo_v, cache.geo_P)
    tables = (;
        v = FEMTools.quadrature_table(backend, FEMTools.shape_function_values(cache.element_v)),
        P = FEMTools.quadrature_table(backend, FEMTools.shape_function_values(cache.element_P, cache.element_v.integration_points)),
        gradients = FEMTools.quadrature_table(backend, FEMTools.shape_function_gradients(cache.element_v)),
    )
    boundary = map(b -> (; b.DoFs, b.vals), bc)
    return (; state, mesh = mesh_data, tables, bc = boundary)
end
