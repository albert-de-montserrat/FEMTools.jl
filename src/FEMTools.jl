module FEMTools

using ForwardDiff, StaticArrays, SparseArrays, LinearAlgebra, Triangulate

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
export DirichletBoundaryCondition

include("sparsity.jl")
export preallocate_sparse_matrix

include("assembly.jl")
export assemble_system!, assemble_sparse_matrix!, assemble_sparse_vector!

include("coloring.jl")
export color_mesh

end # module JustFEM
