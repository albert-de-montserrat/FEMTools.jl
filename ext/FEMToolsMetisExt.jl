module FEMToolsMetisExt

using FEMTools
using Metis
using SparseArrays

function FEMTools.partition_elements(mesh::FEMTools.Mesh{nDim}, nparts::Integer) where {nDim}
    nparts > 0 || throw(ArgumentError("nparts must be positive, got $nparts"))
    # Metis cannot split a graph into more parts than it has vertices.
    active = min(nparts, mesh.nels)
    active <= 1 && return ones(Int, mesh.nels)
    adjacency = generate_element_adjacency(Array(mesh.el2n); shared = :face, dimension = nDim)
    rows = reduce(vcat, (fill(e, length(neighbors)) for (e, neighbors) in enumerate(adjacency)))
    columns = reduce(vcat, adjacency)
    graph = sparse(rows, columns, ones(Int, length(rows)), mesh.nels, mesh.nels)
    return Int.(Metis.partition(graph, active))
end

end # module FEMToolsMetisExt
