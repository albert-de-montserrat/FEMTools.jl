using Triangulate
using FEMTools
using StaticArrays
using LinearAlgebra

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
        x_bubble = [(x[element2node[1,i]] + x[element2node[2,i]] + x[element2node[3,i]]) / 3  for i in axes(element2node, 2)]
        y_bubble = [(y[element2node[1,i]] + y[element2node[2,i]] + y[element2node[3,i]]) / 3  for i in axes(element2node, 2)]
        tag_bubble = length(x) .+ (1:(size(element2node, 2)))

        x = vcat(x, x_bubble)
        y = vcat(y, y_bubble)
        element2node = vcat(element2node, tag_bubble')

        element_coords = [nodes_element(element2node, x, y, i) for i in axes(element2node, 2)]
        new(x, y, element2node, element_coords)
    end
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

function get_el_x_coord(mesh::TriangleMesh, j::Int64, el::Element)
    N = length(el.SF)
    SVector{N}(mesh.x[mesh.element2node[i, j]] for i in 1:N)
end

function get_el_y_coord(mesh::TriangleMesh, j::Int64, el::Element)
    N = length(el.SF)
    SVector{N}(mesh.y[mesh.element2node[i, j]] for i in 1:N)
end

function example_domain_qcdt_area(; minangle = 20, maxarea = 0.05)
    triin = Triangulate.TriangulateIO()
    triin.pointlist = Matrix{Cdouble}([0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]')
    triin.segmentlist = Matrix{Cint}([1 2; 2 3; 3 4; 4  1]')
    triin.segmentmarkerlist = Vector{Int32}([1, 2, 3, 4])
    triout, _ = triangulate("pa$(maxarea)qo2", triin)
    return triout
end


Base.getindex(tri::TriangleMesh, i::Int) = SA[tri.x[i], tri.y[i]]
Base.length(tri::TriangleMesh) = size(tri.element2node, 2)

tri = example_domain_qcdt_area()

mesh = TriangleMesh(tri)
nvert = minimum(mesh.element2node[4:end,1])-1 # this needs to be generalized
TempElement = Element(LinearTriangle)

# solution vector
T = zeros(nvert)

# global matrices
K = zeros(nvert, nvert)
M = zeros(nvert, nvert)
F = zeros(nvert)

for iel in 1:nvert
    el_x = get_el_x_coord(mesh, iel, TempElement)
    el_y = get_el_y_coord(mesh, iel, TempElement)
    el_coords = hcat(el_x, el_y)

    # nodal values
    # T_element = SVector{3}(T[mesh.element2node[i, iel]] for i in 1:3)
    ρCp  = 1
    κ    = 1 # conductivity
    H    = 1 # source term 

    K_el = @MMatrix zeros(length(TempElement), length(TempElement)) # local stiffness matrix
    M_el = @MMatrix zeros(length(TempElement), length(TempElement)) # local mass matrix
    F_el = @MVector zeros(length(TempElement)) # local force vector

    # local integration loop
    for ip in 1:length(TempElement)
        # local shape function
        N = TempElement[ip].N
        # derivative of the local shape function
        ∇N = TempElement[ip].∇N

        # compute jacobian (in the particular linear case this is constant for all integration points)
        J    = ∇N * el_coords
        detJ = det(J)
        detJ < 0 && error("Negative Jacobi: mesh too distorted")
        # invert Jacobian
        invJ = 1 \ detJ

        # Derivatives w.r.t. global coords
        dNdx = J \ ∇N

        # Do integration
        ω = TempElement.ip.vals[ip] * detJ

        # reduce over local matrices
        K_el .+= dNdx' * κ * dNdx * detJ * ω
        M_el .+= N * N' * detJ * ω
        @. F_el += N * H * detJ * ω
    end

    # assemble local matrices into their global counterparts
    dofs = SVector{3}(mesh.element2node[i, iel] for i in 1:3)
    @show dofs
    @views K[dofs, dofs] .+= K_el # these two can be fused
    @views M[dofs, dofs] .+= M_el # these two can be fused
    @views F[dofs] .+= F_el 
end

K_sp = sparse(K)
M_sp = sparse(M)

dt = 1

# left-hand-side
G = @. K_sp + M_sp / dt
# right-hand-side
rhs = (M ./ dt) * T .+ F

# apply dirichlet boundary conditions
bc_dofs         = [1, 2, 3]
dirichlet_value = zeros(3)
for (k, j) in enumerate(bc_dofs)
    rhs[j]  = dirichlet_value[k]
    for i in bc_dofs
        G[i, j] = eltype(G)(i == 1)
    end
end

T .= G \ rhs