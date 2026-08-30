# Viscous sinking block in three dimensions: a dense, stiff cube embedded in a
# lighter, weaker matrix descends under gravity. The discretization is Q2
# velocity / P1-discontinuous pressure on Hex27 cells, solved matrix-free.
#
# The solve runs on any KernelAbstractions backend. For CUDA, `using CUDA` and
# call `run_sinking_block_3d(; backend = CUDABackend())`.
const isGPU = false

@static if isGPU
    using CUDA
    const backend = CUDABackend()
else
    using KernelAbstractions: CPU
    const backend = CPU()
end
using FEMTools
using Gmsh
import KernelAbstractions as KA
using StaticArrays

include("mesher.jl")

"""
    run_sinking_block_3d(; backend=CPU(), mesh_size=0.06, nz=12, half_width=0.15,
                         η=(1.0,100.0), ρ=(1.0,2.0),
                         g=(0.0,0.0,-1.0), write_output=true,
                         verbose=true, solver_tol=1e-6, workgroup=128)

Solve the viscous 3-D sinking-block problem on a Gmsh Hex27 mesh with the
matrix-free `solve_stokes_dyrel!` solver. Velocity uses continuous Q2 functions
and pressure four cell-local P1 modes. The two entries of `η` and `ρ` describe
the matrix and centred block; `half_width` is the block half-width in every
coordinate direction. Free-slip conditions constrain the normal velocity on all
six walls.

Mesh, phases, velocity, and pressure live on `backend`; the mesh generation and
the phase and boundary-node searches run on the host and are transferred once.
Load the matching GPU package before passing its backend, e.g. `using CUDA` and
`backend = CUDABackend()`. `workgroup` sets the kernel workgroup size.

Returns the mesh, velocity and pressure, cell phases, constrained nodes, solver
statistics, and material inputs. With `write_output=true`, also writes
`stokes_3D_sinking_block.vtk` beside this script. `solver_tol` is the absolute
combined residual tolerance of the solve.
"""
function run_sinking_block_3d(;
    backend = CPU(), mesh_size = 0.06, nz = 12, half_width = 0.15,
    η = (1.0, 100.0), ρ = (1.0, 2.0), g = (0.0, 0.0, -1.0),
    write_output = true, verbose = true, solver_tol = 1e-6, workgroup = 128,
)
    # Array constructor of the target backend, used to move the host-built
    # phase, boundary-node, and solution arrays onto the device.
    TDev = FEMTools.TA(backend)
    # Gmsh meshes the horizontal x-y plane with recombined quadrilaterals and
    # extrudes it upward into `nz` layers, giving the unit cube [0,1]³. Gravity
    # acts along -z, so z is the vertical axis.
    coords, el2n, _ = build_gmsh_hex27_mesh(; mesh_size, nz)
    # Velocity is continuous Q2 with 27 nodes per cell. Pressure is not carried
    # by the element: it lives in a separate 4 × nels array of cell-local modes.
    element = ReferenceElement(QuadraticElement{3, 27, Float64})
    mesh = Mesh(backend, coords, el2n, element; workgroup)

    # Phase 1 is the matrix and phase 2 the block; both index into `η` and `ρ`.
    # Cells take a phase whole, so the block is a staircase approximation of the
    # cube whose fidelity depends on `mesh_size` and `nz`.
    center = SVector(0.5, 0.5, 0.5)
    cell_phase = Vector{Int}(undef, mesh.nels)
    for cell in axes(el2n, 2)
        nodes = @view el2n[:, cell]
        # The eight corner nodes lead the Hex27 ordering, so they alone give the centroid.
        centroid = sum(coords[nodes[a]] for a in 1:8) / 8
        cell_phase[cell] = all(abs.(centroid - center) .≤ half_width) ? 2 : 1
    end
    any(==(2), cell_phase) || error("mesh does not resolve the sinking block; reduce mesh_size or increase half_width")
    cell_phase = TDev(cell_phase)

    # Free slip: each wall pins only the velocity component normal to it, leaving
    # tangential flow along the wall unconstrained. `fixed_nodes[c]` lists the
    # nodes whose component `c` is held at zero, so a node on an edge or corner
    # appears once per wall it touches. Extruded coordinates land on a wall only
    # to rounding, which is why the test is a tolerance and not equality.
    tol = 32eps(Float64)
    fixed_nodes = ntuple(component -> TDev(Int32[
        node for (node, c) in pairs(coords)
        if (component == 1 && (abs(c[1]) ≤ tol || abs(c[1] - 1) ≤ tol)) ||
           (component == 2 && (abs(c[2]) ≤ tol || abs(c[2] - 1) ≤ tol)) ||
           (component == 3 && (abs(c[3]) ≤ tol || abs(c[3] - 1) ≤ tol))
    ]), 3)
    # The solver updates both in place, starting from this zero initial guess.
    # Pressure holds the four P1 modes (1, ξ, η, ζ) of each cell.
    velocity = ntuple(_ -> KA.zeros(backend, Float64, mesh.nnodes), 3)
    pressure = KA.zeros(backend, Float64, 4, mesh.nels)
    # `ncheck` sets how often the residual norms are recomputed and reported;
    # the iteration budget the 3-D method actually enforces is `total_iterMax`.
    solve_stats = solve_stokes_dyrel!(
        velocity, pressure, mesh, cell_phase, η, ρ, g, fixed_nodes;
        ncheck = 50, ϵ_tol = solver_tol, iterMax = 50_000, total_iterMax = 50_000,
        workgroup, verbose,
    )
    solve_stats.converged || error("3D DYREL solve did not converge: $(solve_stats.err)")
    if write_output
        # `pressure[1, :]` is the cell-constant mode; the three linear modes have
        # no single cell-centre value and are dropped from the output.
        write_vtk(joinpath(@__DIR__, "stokes_3D_sinking_block.vtk"), mesh;
            point_data = (; vx = velocity[1], vy = velocity[2], vz = velocity[3]),
            cell_data = (; pressure = pressure[1, :], phase = cell_phase),
            title = "FEMTools 3D Q2/P1-disc sinking block")
    end
    return (; mesh, velocity, pressure, cell_phase, fixed_nodes, solve_stats, η, ρ, g)
end

result = run_sinking_block_3d(backend = backend)
# @info "3D sinking-block forward solve" result.mesh.nnodes result.mesh.nels result.solve_stats.err
