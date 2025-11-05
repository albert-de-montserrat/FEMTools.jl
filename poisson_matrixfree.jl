using FEMTools, Triangulate
using LinearAlgebra, SparseArrays, TimerOutputs
using GLMakie

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

function main(tri; reorder = true)
    to = TimerOutput()

    x            = tri.pointlist[1, :]
    y            = tri.pointlist[2, :]
    element2node = tri.trianglelist

    element = T1Element()
    grid = Grid(x, y, element2node, element)
    element_colors, color_list = color_mesh(grid)

    # if reorder
    #     A = adj_matrix(grid)
    #     p = colamd(A)
    #     e2n = map(x -> p[x], grid.element2node)
    #     grid.element2node .= e2n 
    # end

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
    dt   = ttot / nt / 10

    # initial conditions
    alp     = 0.1; bet = 0.3; a = 5.1; b = 4.3; c = -6.2; d = 3.4;
    (; x,y) = grid
    T       = exp.(@. alp*sin(a*x + c*y) + bet*cos(b*x + d*y))
    b       = @. T*(-a*alp*cos(a*x + c*y) + b*bet*sin(b*x + d*y))*(a*alp*cos(a*x + c*y) - b*bet*sin(b*x + d*y)) + T*(a^2*alp*sin(a*x + c*y) + b^2*bet*cos(b*x + d*y)) + T*(-alp*c*cos(a*x + c*y) + bet*d*sin(b*x + d*y))*(alp*c*cos(a*x + c*y) - bet*d*sin(b*x + d*y)) + T*(alp*c^2*sin(a*x + c*y) + bet*d^2*cos(b*x + d*y))

    # # boundary conditions
    # c1 = grid.y => ==(0) => :bot
    # c2 = grid.y => ==(1) => :top
    # c3 = grid.x => ==(0) => :left
    # c4 = grid.x => ==(1) => :right

    # bc_dof = get_dofs(c1, c2, c3, c4)
    # bc_val = :bot => 0e0, :top => 0e0, :left => 0e0, :right => 0e0

    # bcs    = DirichletBoundaryCondition(bc_dof, bc_val...)
    bcs    = DirichletBoundaryCondition()

    # Preallocate sparse matrix structure
    R     = zeros(nnodes)
    θ, Δτ =  0.20398980000000003*0.38, 0.23333333333333336/1.44

    update_T!(T, T0, grid, element, ρ, Cp, κ,  dt, θ, Δτ)
    update_T_atomics!(T, T0, grid, element, ρ, Cp, κ,  dt, θ, Δτ)
    update_T_colored!(T, T0, grid, element, ρ, Cp, κ,  dt, θ, Δτ, color_list)
    
    @timeit to "FEM assembly" begin
        @timeit to "classic" update_T!(T, T0, grid, element, ρ, Cp, κ,  dt, θ, Δτ)
        @timeit to "atomics" update_T_atomics!(T, T0, grid, element, ρ, Cp, κ,  dt, θ, Δτ)
        @timeit to "colored" update_T_colored!(T, T0, grid, element, ρ, Cp, κ,  dt, θ, Δτ, color_list)
    end
    # assemble_system!(KG, MG, FG, grid, element, κ, H)
    # assemble_system_color_coded!(KG, MG, FG, grid, element, κ, H, color_list)
    # assemble_system_atomics!(KG, MG, FG, grid, element, κ, H)

    # @timeit to "FEM assembly" begin
    #     @timeit to "classic" assemble_system!(KG, MG, FG, grid, element, κ, H)
    #     @timeit to "colored" assemble_system_color_coded!(KG, MG, FG, grid, element, κ, H, color_list)
    #     @timeit to "atomics" assemble_system_atomics!(KG, MG, FG, grid, element, κ, H)
    # end
    display(to)

    # # Create stiffness system matrix
    # KLG = @. KG + MG / dt

    # # Apply boundary conditions to sparse matrix
    # set_boundary_condition!(bcs, KLG)

    # t = 0
    # for _ in 1:25
    #     t += dt
    #     # update right-hand side
    #     rhs .= (MG * T) ./ dt .+ FG
    #     set_boundary_condition!(bcs, rhs)
    #     # Solve the linear system
    #     T .= KLG \ rhs
    # end

    # f = mesh(
    #     [grid.x grid.y], 
    #     grid.element2node', 
    #     colormap = :romaO,
    #     color = b, 
    #     shading = false
    # )
end

# tri = example_domain_qcdt_area(; maxarea = 1e-4)
main(tri; reorder = false)


# f = mesh(
#     [grid.x grid.y], 
#     grid.element2node', 
#     colormap = :romaO,
#     color = T, 
#     shading = false
# )

# f = mesh(
#     [grid.x grid.y], 
#     grid.element2node', 
#     colormap = :romaO,
#     color = b, 
#     shading = false
# )

κ = ones(nnodes)
update_poisson_R!(R, T, b, κ, grid, element)

f = mesh(
    [grid.x grid.y], 
    grid.element2node', 
    colormap = :romaO,
    color = R, 
    shading = false
)