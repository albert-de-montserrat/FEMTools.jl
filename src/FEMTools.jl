module FEMTools

using ForwardDiff, StaticArrays, SparseArrays, LinearAlgebra, Triangulate
using Atomix

import Base.getindex, Base.length

include("elements.jl")
export AbstractFiniteElement, T1Element, T2Element
export local_SMatrix, local_SVector

include("basis.jl")

include("quadrature.jl")

include("mesh.jl")
export Point, Grid, getelement, getelementcoords, node_x_el

include("DoFs.jl")
export get_dofs

include("BC.jl")
export DirichletBoundaryCondition, set_boundary_condition!

include("sparsity.jl")
export preallocate_sparse_matrix

include("assembly.jl")
export assemble_system!, assemble_sparse_matrix!, assemble_sparse_vector!
export assemble_system_atomics!

include("coloring.jl")
export color_mesh

end # module JustFEM
