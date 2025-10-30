using JustFEM, Triangulate
using LinearAlgebra, SparseArrays, TimerOutputs

function example_domain_qcdt_area(; minangle = 20, maxarea = 0.05)
    triin                   = Triangulate.TriangulateIO()
    triin.pointlist         = Matrix{Cdouble}([0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]')
    triin.segmentlist       = Matrix{Cint}( [1   2  ; 2   3  ; 3    4 ; 4   1  ]')
    triin.segmentmarkerlist = Vector{Int32}([1, 2, 3, 4])
    area                    = maxarea
    angle                   = minangle
    triout,                 = triangulate("pa$(area)q$(angle)", triin)
    return triout
end;

tri = example_domain_qcdt_area()

x            = tri.pointlist[1, :]
y            = tri.pointlist[2, :]
element2node = tri.trianglelist

element = T1Element()
grid = Grid(x, y, element2node, element)

# # grid limits
# x1 = 0e0
# x2 = 10e0
# number of elements
nel = size(grid)
# chose element type
# element = LinearElement1D()
element = T1Element()
# create the grid
nnodes = length(grid)  # number of nodes

# physical properties
κ    = 1
H    = 0
ttot = 5
nt   = 100
dt   = ttot / nt

# initial conditions
Tmax = 10
σ    = 1
T    = Tmax * exp.(-(grid.x.^2 .+ grid.y.^2) ./ σ)
# boundary conditions
c1 = grid.y => ==(0) => :bot
c2 = grid.y => ==(1) => :top
c3 = grid.x => ==(0) => :left
c4 = grid.x => ==(1) => :right

bc_dof = get_dofs(c1, c2, c3, c4)
bc_val = :bot => 0e0, :top => 0e0, :left => 0e0, :right => 0e0

bcs    = DirichletBoundaryCondition(bc_dof, bc_val...)

# Preallocate sparse matrix structure
dofs_per_element = 1
KG = preallocate_sparse_matrix(grid, 1)
MG = preallocate_sparse_matrix(grid, 1)
FG = zeros(nnodes)

# to = TimerOutput()
# @timeit to "Assembly" for iel in 1:nel
iel = 1
    KL = local_SMatrix(element)
    ML = local_SMatrix(element)
    FL = local_SVector(element)

    el_coords = getelementcoords(grid, iel)

    # @timeit to "Integration" for ip in eachindex(element.ip)
        ip = 1
        # extract integration point and shape functions data
        ξ, ω, N, ∇N_local = element[ip]
        
        # compute the Jacobian for 1D linear element
        J = ∇N_local' * el_coords'
        # in 1D the determinant of the Jacobian is the Jacobian
        detJ = det(J)
        
        # assemble local stiffness matrix (transform gradients to physical coordinates)
        ∇N_global = J \ ∇N_local'
        KL += (∇N_global * κ * ∇N_global') * detJ * ω

        # assemble local mass matrix
        ML += (N * N') * detJ * ω

        # assemble local force vector
        FL += N * H * detJ * ω
    # end

    # assemgle global matrices
    @timeit to "Assembly" begin
        global_dofs = getelement(grid, iel)
        assemble_sparse_matrix!(KG, KL, global_dofs)
        assemble_sparse_matrix!(MG, ML, global_dofs)
        assemble_sparse_vector!(FG, FL, global_dofs)
    end
# end

# Create system matrix (sparse)
KLG = KG + MG / dt

# Apply boundary conditions to sparse matrix
@views KLG[bcs.dofs,:]         .= 0.0
@views KLG[bcs.dofs, bcs.dofs] .= I(length(bcs.dofs))

t = 0

f,ax, = lines(T, label="T0")

b = similar(T)

@timeit to "Time loop" for _ in 1:nt
    t += dt
    @timeit to "build RHS" b .= MG / dt * T .+ FG

    # Apply boundary conditions
    bcs.value          .= T_analytical.(grid.x[bcs.dofs], Tmax, t, κ, σ) 
    @views b[bcs.dofs] .= bcs.value

    # Solve the linear system
    @timeit to "solve" T .= KLG \ b
end

@show to
sol = T_analytical.(grid.x, Tmax, t, κ, σ) 
lines!(ax, T, label="T FEM")
lines!(ax, sol, label="T analytical")
axislegend(ax)
display(f)

function assemble_sparse_matrix!(global_matrix, local_matrix, global_dofs)
    for j in eachindex(global_dofs), i in eachindex(global_dofs)
        @inbounds global_matrix[global_dofs[i], global_dofs[j]] += local_matrix[i, j]
    end
end
@b assemble_sparse_matrix!($(KG, KL, global_dofs)...)

function assemble_sparse_vector!(global_vector, local_vector, global_dofs)
    @inbounds @simd for i in eachindex(global_dofs)
        global_vector[global_dofs[i]] += local_vector[i]
    end
end

@b assemble_sparse_vector!($(FG, FL, global_dofs)...)

function set_boundary_condition!(A::AbstractMatrix, F::AbstractVector, bcs::DirichletBoundaryCondition)
    (; dofs, values) = bcs

end