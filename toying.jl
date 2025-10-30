using FEMTools, Triangulate
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

tri = example_domain_qcdt_area(; maxarea = 1e-3)

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
Tmax = 2
σ    = 1e-1
T    = Tmax * exp.(-((grid.x.-0.5).^2 .+ (grid.y.-0.5).^2) ./ 2σ^2)

f = mesh(
    [grid.x grid.y], 
    grid.element2node', 
    colormap = Reverse(:romaO),
    color = T, 
    shading = false
)

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

assemble_system!(KG, MG, FG, grid, element, κ, H)

@b assemble_system!($(KG, MG, FG, grid, element, κ, H)...)

to

# # Create system matrix (sparse)
# KLG = KG + MG / dt

# # Apply boundary conditions to sparse matrix
# @views KLG[bcs.dofs,:]         .= 0.0
# @views KLG[bcs.dofs, bcs.dofs] .= I(length(bcs.dofs))

# t = 0

# f,ax, = lines(T, label="T0")

# b = similar(T)

# @timeit to "Time loop" for _ in 1:nt
#     t += dt
#     @timeit to "build RHS" b .= MG / dt * T .+ FG

#     # Apply boundary conditions
#     bcs.value          .= T_analytical.(grid.x[bcs.dofs], Tmax, t, κ, σ) 
#     @views b[bcs.dofs] .= bcs.value

#     # Solve the linear system
#     @timeit to "solve" T .= KLG \ b
# end

# @show to
# sol = T_analytical.(grid.x, Tmax, t, κ, σ) 
# lines!(ax, T, label="T FEM")
# lines!(ax, sol, label="T analytical")
# axislegend(ax)
# display(f)

# function set_boundary_condition!(A::AbstractMatrix, F::AbstractVector, bcs::DirichletBoundaryCondition)
#     (; dofs, values) = bcs
# end
