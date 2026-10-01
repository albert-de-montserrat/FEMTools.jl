using KernelAbstractions: CPU
using LinearAlgebra: norm

const _SNAPSHOT_EXAMPLES = joinpath(pkgdir(FEMTools), "examples", "reykjanes")

_snapshot_leaves(x::AbstractArray) = Any[x]
_snapshot_leaves(x::Union{Tuple, NamedTuple}) = reduce(vcat, [_snapshot_leaves(y) for y in x]; init = Any[])
_snapshot_leaves(::Nothing) = Any[]

_snapshot_randomise!(a) = copyto!(a, rand(eltype(a), size(a)))

function _snapshot_objects()
    material = StokesMaterial(;
        η = (1.0, 2.0), ηb = (1.0, 2.0), G = (3.0, 4.0), α = (0.0, 0.0), ρ0 = (1.0, 1.0), K = (5.0, 5.0),
    )
    dr = StokesDR(CPU(), 30, 20, material; stress_size = (6, 7))
    thermal = ThermalDiffusionDR(CPU(), 30, (1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (0.0, 0.0), (Inf, Inf))
    history = FEMTools.IntegrationPointPlasticHistory(zeros(6, 7), zeros(6, 7), zeros(6, 7))
    return (; dr, thermal, history, coords = zeros(2, 30))
end

_snapshot_field_arrays(x::AbstractArray) = Any[x]
_snapshot_field_arrays(x) = Any[Tuple(x)...]

# Every array the objects own, solver scratch included.
function _snapshot_all_arrays(objects, scratch)
    scratch_arrays = vcat(
        [_snapshot_field_arrays(getfield(objects.dr, f)) for f in scratch.stokes]...,
        [_snapshot_field_arrays(getfield(objects.thermal, f)) for f in scratch.thermal]...,
    )
    return vcat(_snapshot_leaves(physical_state(objects)), scratch_arrays)
end

@testset "state snapshot classifies every array field" begin
    include(joinpath(_SNAPSHOT_EXAMPLES, "state_snapshot.jl"))
    objects = _snapshot_objects()

    for (object, scratch) in ((objects.dr, STOKES_SCRATCH_FIELDS), (objects.thermal, THERMAL_SCRATCH_FIELDS))
        held = Set(f for f in fieldnames(typeof(object)) if !(getfield(object, f) isa Union{Number, Tuple}))
        captured = Set(keys(physical_state(object)))
        @test isdisjoint(captured, scratch)
        # A new array field of a solver state fails here until it is classified as state or scratch.
        @test held == union(captured, Set(scratch))
    end
end

@testset "state snapshot round trip" begin
    include(joinpath(_SNAPSHOT_EXAMPLES, "state_snapshot.jl"))
    objects = _snapshot_objects()
    scratch = (; stokes = STOKES_SCRATCH_FIELDS, thermal = THERMAL_SCRATCH_FIELDS)
    state_arrays() = _snapshot_leaves(physical_state(objects))
    scratch_arrays() = _snapshot_all_arrays(objects, scratch)[(length(state_arrays()) + 1):end]

    foreach(_snapshot_randomise!, _snapshot_all_arrays(objects, scratch))
    reference = map(copy, state_arrays())
    counters = [1, 2, 3]
    snapshot = capture_state(objects; time = 3.5, counters, arm = :full)

    for round in 1:2
        # Scramble every array, scratch included, then restore: the state comes back exactly and
        # the scratch is left alone. The second round restores from a snapshot that was restored once.
        foreach(_snapshot_randomise!, _snapshot_all_arrays(objects, scratch))
        scrambled_scratch = map(copy, scratch_arrays())
        values = restore_state!(objects, snapshot)
        @test state_arrays() == reference
        @test scratch_arrays() == scrambled_scratch
        @test values == (; time = 3.5, counters = [1, 2, 3], arm = :full)

        # Neither the returned values nor the live arrays alias the snapshot.
        values.counters[1] = 99
        counters[2] = 99
        @test snapshot.values.counters == [1, 2, 3]
    end
    @test !any(Base.mightalias(a, b) for (a, b) in zip(state_arrays(), _snapshot_leaves(snapshot.arrays)))
end

@testset "state snapshot fails loudly when the structure changed" begin
    include(joinpath(_SNAPSHOT_EXAMPLES, "state_snapshot.jl"))
    objects = _snapshot_objects()
    snapshot = capture_state(objects)

    @test_throws DimensionMismatch restore_state!(merge(objects, (; coords = zeros(2, 31))), snapshot)
    @test_throws DimensionMismatch restore_state!(merge(objects, (; coords = zeros(Float32, 2, 30))), snapshot)
    @test_throws ArgumentError restore_state!((; objects.dr, objects.thermal, objects.history), snapshot)
    @test_throws MethodError capture_state((; time = 1.0))
end

# A tiny compressible problem: a soft inclusion in a stiff host, fixed outer boundary, injection through Q.
function _snapshot_replay_model()
    a, b, workgroup = 1.0, 0.2, 64
    coords, el2n, groups = build_triangulate_t7_cavity_mesh(;
        radius = 6a, cavity_radii = (a, b), max_area = 0.64, refinement = 4,
    )
    element_v = ReferenceElement(QuadraticElement{2, 7, Float64})
    element_P = ReferenceElement(LinearElement{2, 3, Float64})
    mesh = MixedMesh(Mesh(CPU(), coords, el2n, element_v; workgroup), element_P; workgroup)
    K = (5 / 3, 1 / 3)
    material = StokesMaterial(;
        η = (4.0e4, 4.0), ηb = K, G = (1.0, 1.0e-3), α = (0.0, 0.0), ρ0 = (1.0, 1.0), K, g = (0.0, 0.0), Tref = 0.0,
    )
    nq = length(element_v.integration_points.ω)
    dr = StokesDR(CPU(), mesh.nnodes, mesh.nnodesP, material; stress_size = (nq, mesh.nels))
    cavity = findall(==(2), groups.phase)
    Q = uniform_pressure_source(0.1 * π * a * b, Array(mesh.DoFsP), Array(mesh.geometry.geo_P), cavity)
    copyto!(dr.Q, Q)
    zero_bc = DirichletBoundaryCondition(nothing, groups.Γnodes, zeros(length(groups.Γnodes)))
    phases_v = repeat(reshape(groups.phase, 1, :), length(element_v), 1)
    phases_P = repeat(reshape(groups.phase, 1, :), length(element_P), 1)
    γP = zeros(mesh.nnodesP)
    FEMTools.assemble_viscosity_weighted_pressure_scaling!(γP, dr, mesh, 100.0, 1.0; workgroup, phases_v)
    τ, τ_old = Tuple(dr.τ), Tuple(dr.τ_old)
    solve!() = begin
        stats = solve_stokes_dyrel!(
            dr, mesh, zero_bc, zero_bc, 1.0, γP;
            phases_v, phases_P, τ_old, workgroup, ϵ_tol = 1.0e-6, verbose = false, verbose_inner = false,
        )
        return (; P = Array(dr.P), vx = Array(dr.v.x), stats.converged)
    end
    # What an accepted step does after its solve: the current stress and pressure become the history.
    commit!() = begin
        update_stokes_current_stress!(dr, mesh, τ, 1.0; phases_v, τ_old, workgroup)
        copyto!(dr.P0, dr.P)
        foreach(copyto!, τ_old, τ)
    end
    return (; dr, Q, solve!, commit!)
end

@testset "state snapshot replay reproduces a clean run after a discarded trial" begin
    include(joinpath(_SNAPSHOT_EXAMPLES, "state_snapshot.jl"))
    include(joinpath(_SNAPSHOT_EXAMPLES, "elliptical_cavity_setup.jl"))
    include(joinpath(_SNAPSHOT_EXAMPLES, "injection_source.jl"))
    reference = _snapshot_replay_model()
    clean = reference.solve!()
    @test clean.converged

    model = _snapshot_replay_model()
    snapshot = capture_state((; model.dr))
    copyto!(model.dr.Q, 2 .* model.Q)
    trial = model.solve!()
    @test trial.converged
    @test norm(trial.P - clean.P) > 0.1 * norm(clean.P)
    # An accepted step commits its history, which a discarded trial must not leave behind.
    model.commit!()
    @test norm(Tuple(model.dr.τ_old)[1]) > 0

    # Without a restore the trial leaks: it kept the doubled source and the committed history.
    @test norm(model.solve!().P - clean.P) > 0.1 * norm(clean.P)

    restore_state!((; model.dr), snapshot)
    replay = model.solve!()
    @test replay.converged
    @test replay.P ≈ clean.P rtol = 1.0e-8
    @test replay.vx ≈ clean.vx rtol = 1.0e-8
end
