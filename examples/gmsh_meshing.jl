using Gmsh
using FEMTools
using StaticArrays

function _gmsh_triangles(order)
    order in (1, 2) || throw(ArgumentError("triangle order must be 1 or 2"))
    gmsh.model.mesh.generate(2)
    order == 2 && gmsh.model.mesh.setOrder(2)

    node_tags, xyz, _ = gmsh.model.mesh.getNodes()
    coords = [SVector{2, Float64}(xyz[3i - 2], xyz[3i - 1]) for i in eachindex(node_tags)]

    element_type, nnodes = order == 1 ? (2, 3) : (9, 6)
    element_types, _, element_nodes = gmsh.model.mesh.getElements(2)
    element_index = findfirst(==(element_type), element_types)
    isnothing(element_index) && error("Gmsh returned no order-$order triangles")
    nodes = element_nodes[element_index]
    el2n = FEMTools.renumber_connectivity(node_tags, nodes, nnodes)
    FEMTools.orient_triangle_elements!(coords, el2n)
    return coords, el2n
end

function _mesh_size(max_area, width, height)
    return isnothing(max_area) ? max(width, height) / 32 : sqrt(2 * max_area)
end

function _set_circle_mesh_size!(circles, mesh_size, n_circle)
    curves = Int32[]
    for (cx, cy, r) in circles, (dim, tag) in gmsh.model.getEntities(1)
        xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(dim, tag)
        hypot((xmin + xmax) / 2 - cx, (ymin + ymax) / 2 - cy) < r / 2 && push!(curves, tag)
    end
    unique!(curves)
    isempty(curves) && return

    target = minimum((mesh_size, (2pi * r / n_circle for (_, _, r) in circles)...))
    distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance, "CurvesList", curves)
    threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMin", target)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMax", mesh_size)
    gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(threshold, "DistMax", maximum(last, circles))
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
    return
end

"""Build an order-1 or order-2 triangular Gmsh mesh with circular holes."""
function build_gmsh_hole_mesh(; Lx, Ly, holes, n_circle = 64, max_area = nothing, order = 1)
    mesh_size = _mesh_size(max_area, 2Lx, Ly)
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("heat_diffusion_holes")
        rectangle = gmsh.model.occ.addRectangle(-Lx, -Ly, 0, 2Lx, Ly)
        disks = [(2, gmsh.model.occ.addDisk(cx, cy, 0, r, r)) for (cx, cy, r) in holes]
        gmsh.model.occ.cut([(2, rectangle)], disks)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
        _set_circle_mesh_size!(holes, mesh_size, n_circle)
        coords, el2n = _gmsh_triangles(order)
        outer_nodes = FEMTools.rectangle_boundary_nodes(coords, -Lx, Lx, -Ly, 0.0)
        hole_nodes = [FEMTools.circle_boundary_nodes(coords, hole...) for hole in holes]
        return coords, el2n, outer_nodes, hole_nodes
    finally
        gmsh.isInitialized() == 1 && gmsh.finalize()
    end
end

"""Build a T7 Gmsh mesh with a conforming circular material interface."""
function build_gmsh_t7_circle_inclusion_mesh(;
        x0 = 0.0, y0 = 0.0, Lx, Ly, cx, cy, r, n_circle = 96, max_area = nothing)
    mesh_size = _mesh_size(max_area, Lx, Ly)
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("circular_inclusion")
        rectangle = gmsh.model.occ.addRectangle(x0, y0, 0, Lx, Ly)
        disk = gmsh.model.occ.addDisk(cx, cy, 0, r, r)
        gmsh.model.occ.fragment([(2, rectangle)], [(2, disk)])
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
        _set_circle_mesh_size!([(cx, cy, r)], mesh_size, n_circle)
        coords, el2n = FEMTools.add_t7_bubbles!(_gmsh_triangles(2)...)
        outer_nodes = FEMTools.rectangle_boundary_nodes(coords, x0, x0 + Lx, y0, y0 + Ly)
        interface_nodes = FEMTools.circle_boundary_nodes(coords, cx, cy, r)
        return coords, el2n, outer_nodes, interface_nodes
    finally
        gmsh.isInitialized() == 1 && gmsh.finalize()
    end
end

"""Build a T7 Gmsh mesh with a conforming rectangular material interface."""
function build_gmsh_t7_rectangle_inclusion_mesh(;
        Lx, Ly, cx, cy, half_width, max_area = nothing)
    mesh_size = _mesh_size(max_area, Lx, Ly)
    xlo, xhi = cx - half_width, cx + half_width
    ylo, yhi = cy - half_width, cy + half_width
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("rectangular_inclusion")
        rectangle = gmsh.model.occ.addRectangle(0, -Ly, 0, Lx, Ly)
        inclusion = gmsh.model.occ.addRectangle(xlo, ylo, 0, 2half_width, 2half_width)
        gmsh.model.occ.fragment([(2, rectangle)], [(2, inclusion)])
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
        coords, el2n = FEMTools.add_t7_bubbles!(_gmsh_triangles(2)...)
        outer_nodes = FEMTools.rectangle_boundary_nodes(coords, 0.0, Lx, -Ly, 0.0)
        interface_nodes = FEMTools.rectangle_boundary_nodes(coords, xlo, xhi, ylo, yhi)
        return coords, el2n, outer_nodes, interface_nodes
    finally
        gmsh.isInitialized() == 1 && gmsh.finalize()
    end
end

function _gmsh_meshing_selfcheck()
    hole = [(0.0, -0.5, 0.15)]
    _, t3, outer3, holes3 = build_gmsh_hole_mesh(;
        Lx = 1.0, Ly = 1.0, holes = hole, n_circle = 16, max_area = 0.03)
    _, t6, outer6, holes6 = build_gmsh_hole_mesh(;
        Lx = 1.0, Ly = 1.0, holes = hole, n_circle = 16, max_area = 0.03, order = 2)
    _, circle_t7, circle_outer, circle_interface = build_gmsh_t7_circle_inclusion_mesh(;
        Lx = 1.0, Ly = 1.0, cx = 0.5, cy = 0.5, r = 0.15, n_circle = 16, max_area = 0.03)
    _, rectangle_t7, rectangle_outer, rectangle_interface = build_gmsh_t7_rectangle_inclusion_mesh(;
        Lx = 1.0, Ly = 1.0, cx = 0.5, cy = -0.5, half_width = 0.15, max_area = 0.03)
    @assert size(t3, 1) == 3 && !isempty(outer3) && !isempty(only(holes3))
    @assert size(t6, 1) == 6 && !isempty(outer6) && !isempty(only(holes6))
    @assert size(circle_t7, 1) == 7 && !isempty(circle_outer) && !isempty(circle_interface)
    @assert size(rectangle_t7, 1) == 7 && !isempty(rectangle_outer) && !isempty(rectangle_interface)
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && _gmsh_meshing_selfcheck()
