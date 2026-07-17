module FEMToolsWriteVTKExt

using FEMTools
using FEMTools: Mesh, MixedMesh
using WriteVTK: vtk_grid, MeshCell, VTKPointData, VTKCellData, VTKCellTypes

# Map the legacy VTK cell-type id produced by `FEMTools._vtk_topology` to the
# corresponding WriteVTK cell type.
function _vtu_celltype(id)
    id == 3  && return VTKCellTypes.VTK_LINE
    id == 5  && return VTKCellTypes.VTK_TRIANGLE
    id == 9  && return VTKCellTypes.VTK_QUAD
    id == 10 && return VTKCellTypes.VTK_TETRA
    id == 12 && return VTKCellTypes.VTK_HEXAHEDRON
    throw(ArgumentError("unsupported VTK cell-type id $id"))
end

# Scalar fields pass through; vector fields become a 3×N component matrix, padded
# with a zero z-component to match the padded point coordinates.
function _vtu_field(values)
    (isempty(values) || first(values) isa Number) && return values
    data = zeros(Float64, 3, length(values))
    for (j, v) in enumerate(values)
        length(v) in (2, 3) || throw(DimensionMismatch(
            "vector field must have two or three components"))
        for c in eachindex(v)
            data[c, j] = v[c]
        end
    end
    return data
end

"""
    write_vtu(path, mesh; point_data=(;), cell_data=(;)) -> path

Write a binary VTK XML unstructured-grid (`.vtu`) file for `Mesh` or `MixedMesh`
using WriteVTK. Field handling matches [`FEMTools.write_vtk`](@ref): high-order
elements are linearized to their corner nodes, `point_data` fields carry one
value per coordinate node or per written point, two-component vectors are padded
with a zero z-component, and `cell_data` fields carry one value per element.

Unlike the legacy ASCII `write_vtk`, the `.vtu` XML format has no document-title
header, so no `title` argument is accepted.
"""
function FEMTools.write_vtu(path, mesh::Union{Mesh, MixedMesh};
        point_data = (;), cell_data = (;))
    topo = FEMTools._vtk_topology(mesh)

    points = zeros(Float64, 3, length(topo.nodes))
    for (j, old_i) in enumerate(topo.nodes)
        c = topo.coords[old_i]
        if c isa Number
            points[1, j] = c
        else
            for d in eachindex(c)
                points[d, j] = c[d]
            end
        end
    end

    ncorner = size(topo.el2n, 1)
    celltype = _vtu_celltype(topo.cell_type)
    cells = [
        MeshCell(celltype, Int[topo.node_map[topo.el2n[a, iel]] for a in 1:ncorner])
        for iel in 1:topo.nels
    ]

    vtk_grid(path, points, cells) do vtk
        for (name, field) in pairs(point_data)
            vtk[string(name), VTKPointData()] = _vtu_field(FEMTools._vtk_point_values(field, topo, name))
        end
        for (name, field) in pairs(cell_data)
            vtk[string(name), VTKCellData()] = FEMTools._vtk_cell_values(field, topo, name)
        end
    end
    return path
end

end # module FEMToolsWriteVTKExt
