# Linear viscous spherical inclusion in three dimensions: a weak sphere embedded
# in a stiffer matrix, driven by a far-field pure-shear velocity. This is the
# 3-D counterpart of `stokes_2D_viscous_inclusion_triangle`, discretized with Q2
# velocity / P1-discontinuous pressure on Hex27 cells and solved matrix-free.
#
# The mesh conforms to the inclusion: element faces lie on the sphere, so the
# viscosity contrast falls on a curved interface and carries no staircase error.
#
# The solve runs on any KernelAbstractions backend. For CUDA, `using CUDA` and
# call `run_viscous_inclusion_3d(; backend = CUDABackend())`.
const isGPU = true

@static if isGPU
    using CUDA
    const backend = CUDABackend()
else
    using KernelAbstractions: CPU
    const backend = CPU()
end
using FEMTools
import KernelAbstractions as KA
using KernelAbstractions: @Const, @index, @kernel
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
    build_sphere_fitted_hex27_mesh(; L=5.0, radius=1.0, ncells=20, ncells_inner=8)

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
function build_sphere_fitted_hex27_mesh(; L = 5.0, radius = 1.0, ncells = 20, ncells_inner = 8)
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
    cell_strain_rate_invariants(velocity, mesh, cell_phase, η; workgroup=256)
        -> (ε̇_II, τ_II)

Volume-averaged second invariants of the deviatoric strain rate and stress, one
value per cell of a Hex27 mesh.

`velocity` is an `NTuple{3}` of nodal components. At each quadrature point the
velocity gradient is `∂vᵢ/∂xⱼ = Σₐ ∂Nₐ/∂xⱼ vᵢ[a]`; its symmetric deviatoric part
`ε̇'` gives `ε̇_II = sqrt(ε̇'ᵢⱼ ε̇'ᵢⱼ / 2)`, which is averaged over the cell with
the quadrature weights. The linear viscous stress is `τ_II = 2 η ε̇_II`.

Both outputs are allocated on the backend of `velocity`, so the whole
postprocess stays on the device the solve ran on.
"""
function cell_strain_rate_invariants(
    velocity::NTuple{3}, mesh::Mesh, cell_phase, η; workgroup = 256,
)
    backend = KA.get_backend(first(velocity))
    εII = similar(first(velocity), mesh.nels)
    τII = similar(εII)
    cell_invariants_kernel!(backend, workgroup)(
        εII, τII, velocity, mesh.el2n, mesh.geometry, cell_phase, η;
        ndrange = mesh.nels,
    )
    KA.synchronize(backend)
    return εII, τII
end

@kernel function cell_invariants_kernel!(
    εII, τII, @Const(v), @Const(el2n), @Const(geometry), @Const(cell_phase), @Const(η),
)
    cell = @index(Global)
    nodes = ntuple(a -> el2n[a, cell], Val(27))
    vloc = ntuple(i -> SVector{27}(ntuple(a -> v[i][nodes[a]], Val(27))), Val(3))
    weighted = zero(eltype(εII))
    volume = zero(eltype(εII))
    for q in axes(geometry, 1)
        ∂N∂x, dΩ = geometry[q, cell]
        # Velocity gradient: `L[i][j]` is ∂vᵢ/∂xⱼ = Σₐ ∂Nₐ/∂xⱼ vᵢ[a], which is
        # the `27 × 3` gradient matrix contracted over the nodes of the cell.
        L = ntuple(i -> transpose(∂N∂x) * vloc[i], Val(3))
        div = (L[1][1] + L[2][2] + L[3][3]) / 3
        ε̇xx, ε̇yy, ε̇zz = L[1][1] - div, L[2][2] - div, L[3][3] - div
        ε̇xy = (L[1][2] + L[2][1]) / 2
        ε̇xz = (L[1][3] + L[3][1]) / 2
        ε̇yz = (L[2][3] + L[3][2]) / 2
        second = sqrt(
            (ε̇xx^2 + ε̇yy^2 + ε̇zz^2) / 2 + ε̇xy^2 + ε̇xz^2 + ε̇yz^2,
        )
        weighted += second * dΩ
        volume += dΩ
    end
    strain_rate = weighted / volume
    εII[cell] = strain_rate
    τII[cell] = 2 * η[Int(cell_phase[cell])] * strain_rate
end

"""
    run_viscous_inclusion_3d(; backend=CPU(), L=5.0, ncells=20, ncells_inner=8,
                             radius=1.0, η=(2.0,1.0), ε̇_bg=1e-3,
                             write_output=true, verbose=true, solver_tol=1e-9,
                             max_iterations=50_000, workgroup=256)

Solve the linear viscous spherical-inclusion problem on a sphere-fitted Hex27
mesh of the cube `[0,L]³` with the matrix-free `solve_stokes_dyrel!` solver.
Velocity uses continuous Q2 functions and pressure four cell-local P1 modes. The
entries of `η` are the matrix and inclusion viscosities; the sphere of `radius`
sits at the centre of the cube. There is no body force, so the pressure is
entirely dynamic.

All six walls carry the far-field pure-shear velocity

    vx = +ε̇_bg (x - L/2),   vy = 0,   vz = -ε̇_bg (z - L/2),

which is also the initial guess in the interior. The velocity is prescribed on
the whole boundary, so the pressure is fixed only up to a constant; the solver
removes its mean.

`ncells` and `ncells_inner` set the mesh resolution and the share of it spent on
the core of the sphere; see [`build_sphere_fitted_hex27_mesh`](@ref). Element
faces on the interface follow the sphere, so refinement improves the solution
without also changing the shape of the inclusion.

Mesh, phases, velocity, and pressure live on `backend`; the mesh generation,
which also assigns the cell phases, and the boundary-node search run on the host
and are transferred once. Load the matching GPU package before passing its
backend, e.g. `using CUDA` and `backend = CUDABackend()`.

`workgroup` sets the kernel workgroup size.

Returns the mesh, velocity and pressure, cell phases, the strain-rate invariant,
solver statistics, and material inputs. With `write_output=true`, also writes
`stokes_3D_viscous_inclusion.vtk` beside this script. `solver_tol` is the
absolute combined residual tolerance of the solve and `max_iterations` its
budget; failing to reach the tolerance within the budget is an error.
"""
function run_viscous_inclusion_3d(;
    backend = CPU(), L = 5.0, ncells = 20, ncells_inner = 8, radius = 1.0,
    η = (2.0, 1.0), ε̇_bg = 1.0e-3,
    write_output = true, verbose = true, solver_tol = 1.0e-9,
    max_iterations = 50_000, workgroup = 256,
)
    # Array constructor of the target backend, used to move the host-built
    # phase, boundary-node, and boundary-value arrays onto the device.
    TDev = FEMTools.TA(backend)
    # The mesh fills the cube [0,L]³ and conforms to the sphere.
    coords, el2n, cell_phase = build_sphere_fitted_hex27_mesh(;
        L, radius, ncells, ncells_inner,
    )
    # Velocity is continuous Q2 with 27 nodes per cell. Pressure is not carried
    # by the element: it lives in a separate 4 × nels array of cell-local modes.
    element = ReferenceElement(QuadraticElement{3, 27, Float64})
    mesh = Mesh(backend, coords, el2n, element; workgroup)

    # Phase 1 is the matrix and phase 2 the inclusion; both index into `η`.
    cell_phase = TDev(cell_phase)

    # Every node on the six walls carries all three components of the far-field
    # pure-shear velocity. The outermost shell of the map is the box itself, so
    # wall nodes carry their wall coordinate to rounding only.
    tol = 32eps(Float64) * L
    on_wall(c) = any(x -> abs(x) ≤ tol || abs(x - L) ≤ tol, c)
    wall_nodes = Int32[node for (node, c) in pairs(coords) if on_wall(c)]
    pure_shear = (c -> ε̇_bg * (c[1] - L / 2), c -> zero(ε̇_bg), c -> -ε̇_bg * (c[3] - L / 2))
    fixed_nodes = ntuple(_ -> TDev(wall_nodes), 3)
    bc_values = ntuple(component -> TDev([pure_shear[component](coords[n]) for n in wall_nodes]), 3)

    # Seeding the interior with the far-field field costs nothing and starts the
    # iteration from a divergence-free state that already satisfies the walls.
    velocity = FEMTools.VectorField3D(backend, Float64, mesh.nnodes)
    for (component, field) in enumerate(Tuple(velocity))
        copyto!(field, [pure_shear[component](c) for c in coords])
    end
    # Pressure holds the four P1 modes (1, ξ, η, ζ) of each cell.
    pressure = KA.zeros(backend, Float64, 4, mesh.nels)
    # `ncheck` sets how often the residual norms are recomputed and reported;
    # the iteration budget the 3-D method actually enforces is `total_iterMax`.
    solve_stats = solve_stokes_dyrel!(
        Tuple(velocity), pressure, mesh, cell_phase, η, (0.0, 0.0), (0.0, 0.0, 0.0),
        fixed_nodes;
        bc_values, ncheck = 100, ϵ_tol = solver_tol,
        iterMax = max_iterations, total_iterMax = max_iterations,
        workgroup, verbose,
    )
    solve_stats.converged || error("3D DYREL solve did not converge: $(solve_stats.err)")

    εII, τII = cell_strain_rate_invariants(Tuple(velocity), mesh, cell_phase, η; workgroup)
    if write_output
        # `pressure[1, :]` is the cell-constant mode; the three linear modes have
        # no single cell-centre value and are dropped from the output.
        write_vtk(joinpath(@__DIR__, "stokes_3D_viscous_inclusion.vtk"), mesh;
            point_data = (; vx = velocity.x, vy = velocity.y, vz = velocity.z),
            cell_data = (; pressure = pressure[1, :], phase = cell_phase, epsII = εII, tauII = τII),
            title = "FEMTools 3D Q2/P1-disc viscous inclusion")
    end
    return (; mesh, velocity, pressure, cell_phase, εII, τII, solve_stats, η, ε̇_bg)
end

@time result = run_viscous_inclusion_3d(backend = backend,  solver_tol = 1.0e-6)
@info "3D viscous-inclusion forward solve" result.mesh.nnodes result.mesh.nels result.solve_stats.err extrema(result.τII)
