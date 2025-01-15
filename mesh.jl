using Triangulate

function example_domain_regions()
    triin = Triangulate.TriangulateIO()
    triin.pointlist = Matrix{Cdouble}([
        0.0 0.0; 
        0.5 0.0; 
        1.0 0.0; 
        1.0 1.0; 
        0.6 0.6; 
        0.0 1.0
    ]') # outter domain corners
    triin.segmentlist = Matrix{Cint}([
        1 2; 
        2 3; 
        3 4; 
        4 5; 
        5 6; 
        6 1; 
        2 5
        ]')
    triin.segmentmarkerlist = Vector{Int32}([1, 2, 3, 4, 5, 6, 7])
    triin.regionlist = Matrix{Cdouble}([
        0.2  0.8; # points inside each subdomain
        0.2  0.2; # points inside each subdomain
        1    2;   # domain labels
        0.01 0.05 # max area for each subdomain
    ])
    tri,  = triangulate("paAq$(angle)Qo2", triin)
    tri
end

tri = example_domain_regions()

box = [
    0 0;
    1 0;
    1 1;
    0 1;
]

layers_depth = [
   0.5
   0.25
]

nl = length(layers_depth) + 1

xmin, xmax = extrema(box[i] for i in axes(box, 1))

left_intersections = [
    [xmin, layers_depth[i]] for i in 1:nl-1
]
right_intersections = [
    [xmax, layers_depth[i]] for i in 1:nl-1
]

new_box = box[1:2, :]
new_box = vcat(new_box, reduce(hcat, right_intersections)')
new_box = vcat(new_box, box[3:end, :])
new_box = vcat(new_box, reduce(hcat, left_intersections)')


scatter(new_box[:,1], new_box[:,2])

##################

function example_domain_qcdt_area(; minangle = 20, maxarea = 0.05)
    triin = Triangulate.TriangulateIO()
    triin.pointlist = Matrix{Cdouble}([0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]')
    triin.segmentlist = Matrix{Cint}([1 2; 2 3; 3 4; 4  1]')
    triin.segmentmarkerlist = Vector{Int32}([1, 2, 3, 4])
    triout, _ = triangulate("pa$(maxarea)qo2", triin)
    return triout
end


function nodes_element(mesh::TriangleMesh, el::Int64)
    x = SVector{7,Float64}(mesh.x[mesh.element2node[i, el]] for i in 1:7)
    y = SVector{7,Float64}(mesh.y[mesh.element2node[i, el]] for i in 1:7)
    return x, y
end

function nodes_element(element2node, x, y, el::Int64)
    x = SVector{7,Float64}(x[element2node[i, el]] for i in 1:7)
    y = SVector{7,Float64}(y[element2node[i, el]] for i in 1:7)
    return x, y
end


struct TriangleMesh
    x::Vector{Float64}
    y::Vector{Float64}
    element2node::Matrix{Int32}
    element_coords::Vector{Tuple{SVector{7, Float64}, SVector{7, Float64}}}
    
    function TriangleMesh(tri::TriangulateIO)
        x = tri.pointlist[1, :]
        y = tri.pointlist[2, :]
        element2node = tri.trianglelist

        # add bubble node 
        x = vcat(x, x_bubble)
        y = vcat(y, y_bubble)
        element2node = vcat(element2node, tag_bubble')

        element_coords = [nodes_element(element2node, x, y, i) for i in axes(mesh.element2node, 2)]
        new(x, y, element2node, element_coords)
    end
end

Base.getindex(tri::TriangleMesh, i::Int) = SA[tri.x[i], tri.y[i]]

tri = example_domain_qcdt_area()

mesh = TriangleMesh(tri)

