# Viscous sinking sphere in three dimensions: a dense, stiff sphere embedded in
# a lighter, weaker matrix descends under gravity. The discretization is Q2
# velocity / P1-discontinuous pressure on Hex27 cells, solved matrix-free.
#
# Unlike the sinking-block example, the mesh conforms to the inclusion: element
# faces lie on the sphere, so the phase boundary carries no staircase error.
#
# The solve runs on any KernelAbstractions backend. For CUDA, `using CUDA` and
# call `run_sinking_sphere_3d(; backend = CUDABackend())`.
const isGPU = false

@static if isGPU
    using CUDA
    const backend = CUDABackend()
else
    using KernelAbstractions: CPU
    const backend = CPU()
end
using FEMTools
import KernelAbstractions as KA
using LinearAlgebra: norm
using StaticArrays

"""
    sphere_shell_map(u, ms, radius, halfwidth) -> SVector{3}

Map a logical coordinate `u ∈ [-1,1]³` onto the box of half-width `halfwidth`
centred at the origin, with the shell `‖u‖∞ = ms` sent onto the sphere of
radius `radius`.

Each shell `‖u‖∞ = m` keeps its topology and is reshaped by blending the unit
cube surface `p = u/m` with its radial projection `q = p/‖p‖₂` onto the unit
sphere. The blend is spherical at `m = ms` and cubic at both `m = 0` and
`m = 1`, so the mesh is a plain grid at the centre, follows the sphere at the
interface, and meets the flat walls of the box.
"""
function sphere_shell_map(u::SVector{3}, ms, radius, halfwidth)
    m = maximum(abs, u)
    iszero(m) && return zero(u)
    p = u / m
    q = p / norm(p)
    # Shell size `s` and blend `β` are piecewise linear in `m`, pinned so that
    # s(ms) = radius with a pure sphere, and s(1) = halfwidth with a pure cube.
    # Towards the centre β vanishes as m, leaving x ∝ u and a regular Jacobian.
    s, β = if m ≤ ms
        radius * m / ms, m / ms
    else
        radius + (halfwidth - radius) * (m - ms) / (1 - ms), (1 - m) / (1 - ms)
    end
    return s * ((1 - β) * p + β * q)
end

"""
    build_sphere_fitted_hex27_mesh(; L=1.0, radius=0.2, ncells=16, ncells_inner=8)

Build a Hex27 mesh of the box `[0,L]³` whose element faces conform to the
sphere of `radius` centred in the box.

The mesh is the image of a structured logical grid on `[-1,1]³` under
[`sphere_shell_map`](@ref), with the sphere placed on the shell
`ms = ncells_inner / ncells`. `ncells` sets the cells per direction and
`ncells_inner` how many of them span the core of the sphere; the two must have
equal parity so that the shell falls on a cell boundary. All 27 nodes of a cell
are placed by the map, so the interface is a curved quadratic surface
interpolating the sphere rather than a facetted approximation of it.

Returns `(coords, el2n, cell_phase)`: node coordinates, `27 × nels`
connectivity in FEMTools' Hex27 ordering, and a phase per cell that is 2 inside
the sphere and 1 outside. The phase is exact: no cell straddles the interface.
"""
function build_sphere_fitted_hex27_mesh(; L = 1.0, radius = 0.2, ncells = 16, ncells_inner = 8)
    0 < ncells_inner < ncells ||
        throw(ArgumentError("ncells_inner must lie strictly between 0 and ncells, got $ncells_inner of $ncells"))
    iseven(ncells - ncells_inner) ||
        throw(ArgumentError("ncells and ncells_inner must have equal parity so the sphere lands on a cell boundary, got $ncells and $ncells_inner"))
    halfwidth = L / 2
    radius < halfwidth ||
        throw(ArgumentError("sphere of radius $radius does not fit in a box of half-width $halfwidth"))
    ms = ncells_inner / ncells

    element = ReferenceElement(QuadraticElement{3, 27, Float64})
    el2n = generate_element2node(element, (ncells, ncells, ncells))

    # Quadratic cells carry corner, mid-edge, mid-face, and centre nodes, so the
    # node grid is refined twice per cell. `generate_element2node` numbers it
    # with x varying fastest, then y, then z.
    center = SVector(halfwidth, halfwidth, halfwidth)
    nrefined = 2ncells + 1
    coords = Vector{SVector{3, Float64}}(undef, nrefined^3)
    inode = 1
    for iz in 0:(2ncells), iy in 0:(2ncells), ix in 0:(2ncells)
        u = SVector(ix / ncells - 1, iy / ncells - 1, iz / ncells - 1)
        coords[inode] = center + sphere_shell_map(u, ms, radius, halfwidth)
        inode += 1
    end

    # A cell lies inside the sphere exactly when its logical centre is inside
    # the shell `ms`, matching the element loop of `generate_element2node`.
    cell_phase = Vector{Int}(undef, ncells^3)
    icell = 1
    for ez in 0:(ncells - 1), ey in 0:(ncells - 1), ex in 0:(ncells - 1)
        uc = SVector(
            (2ex + 1) / ncells - 1, (2ey + 1) / ncells - 1, (2ez + 1) / ncells - 1,
        )
        cell_phase[icell] = maximum(abs, uc) < ms ? 2 : 1
        icell += 1
    end

    return coords, el2n, cell_phase
end

"""
    run_sinking_sphere_3d(; backend=CPU(), ncells=12, ncells_inner=6, radius=0.2,
                          η=(1.0,100.0), ρ=(1.0,2.0),
                          g=(0.0,0.0,-1.0), write_output=true, verbose=true,
                          solver_tol=1e-6, max_iterations=100_000, workgroup=128)

Solve the viscous 3-D sinking-sphere problem on a sphere-fitted Hex27 mesh with
the matrix-free `solve_stokes_dyrel!` solver. Velocity uses continuous Q2
functions and pressure four cell-local P1 modes. The two entries of `η` and `ρ`
describe the matrix and the centred sphere of `radius`. Free-slip conditions
constrain the normal velocity on all six walls.

`ncells` and `ncells_inner` set the mesh resolution and the share of it spent
on the core of the sphere; see [`build_sphere_fitted_hex27_mesh`](@ref). The
element faces on the interface follow the sphere, so refinement improves the
solution without also changing the shape of the inclusion.

Dynamic relaxation needs iterations in proportion to the spread of cell sizes,
which the shell map widens as the mesh is refined: `ncells = 12` reaches
`solver_tol = 1e-6` in about 35 000 iterations, `ncells = 16` needs more than
50 000. Raise `max_iterations` alongside `ncells`.

Mesh, phases, velocity, and pressure live on `backend`; the mesh generation and
the phase and boundary-node searches run on the host and are transferred once.
Load the matching GPU package before passing its backend, e.g. `using CUDA` and
`backend = CUDABackend()`. `workgroup` sets the kernel workgroup size.

Returns the mesh, velocity and pressure, cell phases, constrained nodes, solver
statistics, and material inputs. With `write_output=true`, also writes
`stokes_3D_sinking_sphere.vtk` beside this script. `solver_tol` is the absolute
combined residual tolerance of the solve and `max_iterations` its budget;
failing to reach the tolerance within the budget is an error.
"""
function run_sinking_sphere_3d(;
    backend = CPU(), ncells = 12, ncells_inner = 6, radius = 0.2,
    η = (1.0, 100.0), ρ = (1.0, 2.0), g = (0.0, 0.0, -1.0),
    write_output = true, verbose = true, solver_tol = 1e-6,
    max_iterations = 100_000, workgroup = 128,
)
    # Array constructor of the target backend, used to move the host-built
    # phase, boundary-node, and solution arrays onto the device.
    TDev = FEMTools.TA(backend)
    # The mesh fills the unit cube [0,1]³ and conforms to the sphere. Gravity
    # acts along -z, so z is the vertical axis.
    coords, el2n, cell_phase = build_sphere_fitted_hex27_mesh(;
        L = 1.0, radius, ncells, ncells_inner,
    )
    # Velocity is continuous Q2 with 27 nodes per cell. Pressure is not carried
    # by the element: it lives in a separate 4 × nels array of cell-local modes.
    element = ReferenceElement(QuadraticElement{3, 27, Float64})
    mesh = Mesh(backend, coords, el2n, element; workgroup)

    # Phase 1 is the matrix and phase 2 the sphere; both index into `η` and `ρ`.
    cell_phase = TDev(cell_phase)

    # Free slip: each wall pins only the velocity component normal to it, leaving
    # tangential flow along the wall unconstrained. `fixed_nodes[c]` lists the
    # nodes whose component `c` is held at zero, so a node on an edge or corner
    # appears once per wall it touches. The outermost shell of the map is the
    # box itself, so wall nodes carry their wall coordinate to rounding only.
    tol = 32eps(Float64)
    fixed_nodes = ntuple(component -> TDev(Int32[
        node for (node, c) in pairs(coords)
        if (component == 1 && (abs(c[1]) ≤ tol || abs(c[1] - 1) ≤ tol)) ||
           (component == 2 && (abs(c[2]) ≤ tol || abs(c[2] - 1) ≤ tol)) ||
           (component == 3 && (abs(c[3]) ≤ tol || abs(c[3] - 1) ≤ tol))
    ]), 3)
    # The solver updates both in place, starting from this zero initial guess.
    # Pressure holds the four P1 modes (1, ξ, η, ζ) of each cell.
    velocity = FEMTools.VectorField3D(backend, Float64, mesh.nnodes)
    pressure = KA.zeros(backend, Float64, 4, mesh.nels)
    # `ncheck` sets how often the residual norms are recomputed and reported;
    # the iteration budget the 3-D method actually enforces is `total_iterMax`.
    solve_stats = solve_stokes_dyrel!(
        Tuple(velocity), pressure, mesh, cell_phase, η, ρ, g, fixed_nodes;
        ncheck = 50, ϵ_tol = solver_tol,
        iterMax = max_iterations, total_iterMax = max_iterations,
        workgroup, verbose,
    )
    solve_stats.converged || error("3D DYREL solve did not converge: $(solve_stats.err)")
    if write_output
        # `pressure[1, :]` is the cell-constant mode; the three linear modes have
        # no single cell-centre value and are dropped from the output.
        write_vtk(joinpath(@__DIR__, "stokes_3D_sinking_sphere.vtk"), mesh;
            point_data = (; vx = velocity.x, vy = velocity.y, vz = velocity.z),
            cell_data = (; pressure = pressure[1, :], phase = cell_phase),
            title = "FEMTools 3D Q2/P1-disc sinking sphere")
    end
    return (; mesh, velocity, pressure, cell_phase, fixed_nodes, solve_stats, η, ρ, g)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    result = run_sinking_sphere_3d(backend = backend, ncells = 12, ncells_inner = 6)
    @info "3D sinking-sphere forward solve" result.mesh.nnodes result.mesh.nels result.solve_stats.err
end
