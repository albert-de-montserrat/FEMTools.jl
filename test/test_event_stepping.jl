using KernelAbstractions: CPU
using LinearAlgebra: norm

const _STEPPING_EXAMPLES = joinpath(pkgdir(FEMTools), "examples", "reykjanes")

@testset "transactional step accepts and reports" begin
    include(joinpath(_STEPPING_EXAMPLES, "event_stepping.jl"))
    objects = (; a = zeros(3))

    report = attempt_step!(objects; Δτ = 2.0, values = (; nevents = 0)) do Δτ
        objects.a .= Δτ
        return (; converged = true, err = 1.0e-7)
    end

    @test report.outcome == :accepted
    @test report.accepted
    @test (report.attempts, report.Δτ, report.limit) == (1, 2.0, :none)
    @test isempty(report.failures)
    @test report.stats.err == 1.0e-7
    @test report.values == (; nevents = 0)
    # An accepted step keeps what it wrote.
    @test objects.a == fill(2.0, 3)
end

@testset "transactional step retries with a smaller step" begin
    include(joinpath(_STEPPING_EXAMPLES, "event_stepping.jl"))
    objects = (; a = zeros(3))
    rebuilds = Ref(0)

    # Converges only once the step is small enough, as an elastic-dominated step does.
    report = attempt_step!(objects; Δτ = 1.0, max_attempts = 3, rebuild! = () -> (rebuilds[] += 1)) do Δτ
        objects.a .= Δτ
        return (; converged = Δτ <= 0.3, err = Δτ, iter = 10)
    end

    @test report.accepted
    @test (report.attempts, report.Δτ, report.limit) == (3, 0.25, :none)
    @test objects.a == fill(0.25, 3)
    @test rebuilds[] == 2
    @test [f.Δτ for f in report.failures] == [1.0, 0.5]
    @test all(f -> (f.check, f.outcome) == (:solver_converged, :solver_failed), report.failures)
    @test occursin("iter = 10", report.failures[1].detail)
end

@testset "a rejected step leaves the state untouched" begin
    include(joinpath(_STEPPING_EXAMPLES, "event_stepping.jl"))
    objects = (; a = fill(7.0, 3))

    exhausted = attempt_step!(objects; Δτ = 1.0, max_attempts = 2) do Δτ
        objects.a .= Δτ
        return (; converged = false, err = 1.0, iter = 50)
    end
    @test !exhausted.accepted
    @test (exhausted.outcome, exhausted.limit, exhausted.attempts) == (:solver_failed, :attempts, 2)
    @test exhausted.stats === nothing
    @test objects.a == fill(7.0, 3)

    # The retries stop at the smallest step instead of shrinking indefinitely.
    floored = attempt_step!(objects; Δτ = 1.0, max_attempts = 10, Δτ_min = 0.3) do Δτ
        objects.a .= Δτ
        return (; converged = false)
    end
    @test (floored.limit, floored.attempts, floored.Δτ) == (:step_size, 2, 0.5)
    @test objects.a == fill(7.0, 3)

    # A state that is not finite is rejected even when the solver claims to have converged.
    nonfinite = attempt_step!(objects; Δτ = 1.0) do Δτ
        objects.a[2] = NaN
        return (; converged = true)
    end
    @test nonfinite.outcome == :invalid_state
    @test nonfinite.failures[1].check == :finite_state
    @test occursin("a holds a non-finite value", nonfinite.failures[1].detail)
    @test objects.a == fill(7.0, 3)

    # A step rejects its own trial for what only it can see, such as an inverted element.
    rejected = attempt_step!(objects; Δτ = 1.0) do Δτ
        objects.a .= Δτ
        throw(StepRejection(:invalid_geometry, "mesh advection inverted an element"))
    end
    @test rejected.outcome == :invalid_geometry
    @test rejected.failures[1].check == :step_rejected
    @test objects.a == fill(7.0, 3)
    @test occursin("invalid_geometry", step_failure_message(rejected))
    @test occursin("inverted an element", step_failure_message(rejected))
end

@testset "transactional step argument validation" begin
    include(joinpath(_STEPPING_EXAMPLES, "event_stepping.jl"))
    objects = (; a = zeros(3))
    ok(_) = (; converged = true)

    @test_throws ArgumentError attempt_step!(ok, objects; Δτ = 0.0)
    @test_throws ArgumentError attempt_step!(ok, objects; Δτ = 1.0, Δτ_min = 2.0)
    @test_throws ArgumentError attempt_step!(ok, objects; Δτ = 1.0, max_attempts = 0)
    @test_throws ArgumentError attempt_step!(ok, objects; Δτ = 1.0, max_attempts = 2, shrink = 1.0)
    @test_throws ArgumentError StepRejection(:not_an_outcome, "")
    @test_throws ArgumentError StepCheck(:c, :not_an_outcome, (objects, stats) -> nothing)

    # A step that fails for a reason that is not a physical rejection is a bug, so it propagates.
    @test_throws DomainError attempt_step!(objects; Δτ = 1.0) do Δτ
        throw(DomainError(Δτ, "not a rejection"))
    end
end

# The tiny compressible cavity of `test_state_snapshot.jl`: a soft inclusion in a stiff host with a
# fixed outer boundary, injected through `Q`. `step!` is one accepted step: solve, then commit the
# stress and pressure history. `Q_factor` scales the source so a trial can differ from the clean run.
function _stepping_replay_model()
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
    zero_bc = DirichletBoundaryCondition(nothing, groups.Γnodes, zeros(length(groups.Γnodes)))
    phases_v = repeat(reshape(groups.phase, 1, :), length(element_v), 1)
    phases_P = repeat(reshape(groups.phase, 1, :), length(element_P), 1)
    γP = zeros(mesh.nnodesP)
    τ, τ_old = Tuple(dr.τ), Tuple(dr.τ_old)

    function step!(Δτ; Q_factor = 1)
        copyto!(dr.Q, Q_factor .* Q)
        # `γP` and `dr.M_P` depend on the step, so a trial at another step size rebuilds them or the
        # relaxation crawls: at Δτ = 0.5 with the scaling of Δτ = 1 it does not converge at all.
        FEMTools.assemble_viscosity_weighted_pressure_scaling!(γP, dr, mesh, 100.0, Δτ; workgroup, phases_v)
        stats = solve_stokes_dyrel!(
            dr, mesh, zero_bc, zero_bc, Δτ, γP;
            phases_v, phases_P, τ_old, workgroup, ϵ_tol = 1.0e-6, verbose = false, verbose_inner = false,
        )
        update_stokes_current_stress!(dr, mesh, τ, Δτ; phases_v, τ_old, workgroup)
        copyto!(dr.P0, dr.P)
        foreach(copyto!, τ_old, τ)
        return stats
    end
    return (; dr, step!)
end

@testset "a forced failure inside a trial does not change the next accepted step" begin
    include(joinpath(_STEPPING_EXAMPLES, "event_stepping.jl"))
    include(joinpath(_STEPPING_EXAMPLES, "elliptical_cavity_setup.jl"))
    include(joinpath(_STEPPING_EXAMPLES, "injection_source.jl"))

    reference = _stepping_replay_model()
    clean = map(1:2) do _
        report = attempt_step!(Δτ -> reference.step!(Δτ), (; reference.dr); Δτ = 1.0)
        @test report.accepted
        return Array(reference.dr.P)
    end

    model = _stepping_replay_model()
    trial_P = Float64[]
    # A trial that solves and commits a doubled source, and is only then found to have failed.
    failed = attempt_step!((; model.dr); Δτ = 1.0, max_attempts = 1) do Δτ
        stats = model.step!(Δτ; Q_factor = 2)
        append!(trial_P, Array(model.dr.P))
        return merge(stats, (; converged = false))
    end
    @test !failed.accepted
    @test failed.outcome == :solver_failed
    # The trial did reach a different state, so the rollback below has something to undo.
    @test norm(trial_P - clean[1]) > 0.1 * norm(clean[1])

    replay = map(1:2) do _
        report = attempt_step!(Δτ -> model.step!(Δτ), (; model.dr); Δτ = 1.0)
        @test report.accepted
        return Array(model.dr.P)
    end
    @test replay[1] ≈ clean[1] rtol = 1.0e-8
    @test replay[2] ≈ clean[2] rtol = 1.0e-8
end

# A trial whose only physics is that it advances the state by its own step size, so that the search
# can be checked against a crossing whose position is known exactly.
_stepping_ramp(objects) = Δτ -> (objects.state[1] = Δτ; (; converged = true))
_stepping_at(threshold) = (objects, _) -> objects.state[1] >= threshold

@testset "the crossing search brackets the earliest crossing" begin
    include(joinpath(_STEPPING_EXAMPLES, "event_stepping.jl"))
    objects = (; state = zeros(1))
    trials = capture_trials(objects)

    report = bracket_crossing!(
        _stepping_at(0.37), trials, _stepping_ramp(objects);
        Δτ = 1.0, samples = 4, tolerance = 0.01, commit = false,
    )
    @test report.outcome == :crossing_bracketed
    @test report.toggles == 1
    @test report.lo < 0.37 <= report.hi
    @test report.hi - report.lo <= report.tolerance
    # Every trial ran from the same left state, which is the state left behind without a commit.
    @test objects.state == [0.0]
    @test all(p -> p.tripped == (p.Δτ >= 0.37), report.probes)

    committed = bracket_crossing!(
        _stepping_at(0.37), trials, _stepping_ramp(objects);
        Δτ = 1.0, samples = 4, tolerance = 0.01,
    )
    @test committed.committed.accepted
    @test objects.state == [committed.hi]
end

@testset "the crossing search reports what it cannot resolve" begin
    include(joinpath(_STEPPING_EXAMPLES, "event_stepping.jl"))
    objects = (; state = zeros(1))
    trials = capture_trials(objects)
    ramp = _stepping_ramp(objects)

    # A detector that switches back off: bisection alone would return whichever crossing it fell into.
    intermittent = (objects, _) -> 0.2 <= objects.state[1] <= 0.4 || objects.state[1] >= 0.9
    report = bracket_crossing!(intermittent, trials, ramp; Δτ = 1.0, samples = 4, tolerance = 0.01)
    @test report.outcome == :unresolved_crossing
    @test report.toggles == 3
    # The earliest crossing is still resolved and reported.
    @test report.lo < 0.2 <= report.hi
    @test objects.state == [0.0]

    quiet = bracket_crossing!((objects, _) -> false, trials, ramp; Δτ = 1.0, samples = 3)
    @test quiet.outcome == :no_crossing
    @test length(quiet.probes) == 3

    # The trial budget stops the bisection before the tolerance is reached.
    starved = bracket_crossing!(
        _stepping_at(0.37), trials, ramp; Δτ = 1.0, samples = 4, tolerance = 1.0e-6, max_trials = 6,
    )
    @test starved.outcome == :unresolved_crossing
    @test length(starved.probes) == 6
    @test starved.hi - starved.lo > starved.tolerance

    failing = bracket_crossing!(
        _stepping_at(0.37), trials, Δτ -> (objects.state[1] = Δτ; (; converged = false));
        Δτ = 1.0, samples = 4,
    )
    @test failing.outcome == :solver_failed
    @test length(failing.probes) == 1
    @test objects.state == [0.0]

    @test_throws ArgumentError bracket_crossing!(_stepping_at(0.37), trials, ramp; Δτ = 0.0)
    @test_throws ArgumentError bracket_crossing!(_stepping_at(0.37), trials, ramp; Δτ = 1.0, samples = 0)
    @test_throws ArgumentError bracket_crossing!(_stepping_at(0.37), trials, ramp; Δτ = 1.0, tolerance = 2.0)
    @test_throws ArgumentError bracket_crossing!(
        _stepping_at(0.37), trials, ramp; Δτ = 1.0, samples = 4, max_trials = 3,
    )
end

@testset "the crossing search leaves the state a plain step would leave" begin
    include(joinpath(_STEPPING_EXAMPLES, "event_stepping.jl"))
    include(joinpath(_STEPPING_EXAMPLES, "elliptical_cavity_setup.jl"))
    include(joinpath(_STEPPING_EXAMPLES, "injection_source.jl"))

    # The injected volume, and with it the cavity pressure, grows with the step, so a pressure
    # threshold between the pressures of a half and a full step is crossed inside the step.
    reference = _stepping_replay_model()
    reference_trials = capture_trials((; reference.dr))
    pressure = (objects, _) -> sum(objects.dr.P) / length(objects.dr.P)
    half = run_trial!(pressure, reference_trials, reference.step!; Δτ = 0.5)
    full = run_trial!(pressure, reference_trials, reference.step!; Δτ = 1.0)
    threshold = (half.observation + full.observation) / 2
    @test half.observation < threshold < full.observation

    model = _stepping_replay_model()
    trials = capture_trials((; model.dr))
    report = bracket_crossing!(
        (objects, stats) -> pressure(objects, stats) >= threshold, trials, model.step!;
        Δτ = 1.0, samples = 2, tolerance = 0.15,
    )
    @test report.outcome == :crossing_bracketed
    @test 0.5 <= report.lo < report.hi <= 1.0
    @test report.hi - report.lo <= report.tolerance

    # The committed state is the one a single step of `hi` reaches from the left state: the trials
    # in between left nothing behind.
    direct = _stepping_replay_model()
    direct.step!(report.hi)
    @test Array(model.dr.P) ≈ Array(direct.dr.P) rtol = 1.0e-10
    @test Array(model.dr.v.x) ≈ Array(direct.dr.v.x) rtol = 1.0e-10
end

# A trial whose observed quantity is a known, increasing function of the amplitude it is given.
_stepping_amplitude_step(objects, response) =
    (amplitude, Δτ) -> (objects.state[1] = response(amplitude) * Δτ; (; converged = true))

@testset "the amplitude solve brackets the smallest amplitude that reaches the target" begin
    include(joinpath(_STEPPING_EXAMPLES, "event_stepping.jl"))
    objects = (; state = zeros(1))
    trials = capture_trials(objects)
    value = (objects, _) -> objects.state[1]

    report = solve_amplitude!(
        value, trials, _stepping_amplitude_step(objects, a -> 2a + 1);
        Δτ = 1.0, target = 8.0, amplitude = 1.0, rtol = 1.0e-4,
    )
    @test report.outcome == :amplitude_bracketed
    @test isapprox(report.amplitude, 3.5; rtol = 1.0e-3)      # 2a + 1 = 8
    @test report.observation >= 8.0                            # the committed end reaches the target
    @test report.lo < 3.5 <= report.hi
    @test report.hi - report.lo <= report.rtol * report.hi
    @test objects.state[1] == report.observation               # committed, not restored
    @test first(report.probes).amplitude == 0.0                # the zero trial safeguards the bracket
    @test [p.amplitude for p in report.probes[2:3]] == [1.0, 2.0]   # then doubling until the target

    # A target already met at rest is not an intrusion.
    at_rest = solve_amplitude!(
        (objects, _) -> 10.0, trials, _stepping_amplitude_step(objects, identity);
        Δτ = 1.0, target = 8.0, amplitude = 1.0,
    )
    @test at_rest.outcome == :no_intrusion_needed
    @test at_rest.amplitude == 0.0
    @test length(at_rest.probes) == 1
end

@testset "the amplitude solve reports a target it cannot reach" begin
    include(joinpath(_STEPPING_EXAMPLES, "event_stepping.jl"))
    objects = (; state = zeros(1))
    trials = capture_trials(objects)
    value = (objects, _) -> objects.state[1]

    # A response that saturates below the target: doubling must stop, not run away.
    missing_root = solve_amplitude!(
        value, trials, _stepping_amplitude_step(objects, a -> 5 * a / (1 + a));
        Δτ = 1.0, target = 8.0, amplitude = 1.0, amplitude_max = 64.0,
    )
    @test missing_root.outcome == :no_arrest_bracket
    @test missing_root.hi == 64.0
    @test last(missing_root.probes).amplitude == 64.0          # `amplitude_max` itself is tried
    @test objects.state == [0.0]

    starved = solve_amplitude!(
        value, trials, _stepping_amplitude_step(objects, a -> 2a + 1);
        Δτ = 1.0, target = 8.0, amplitude = 1.0, rtol = 1.0e-9, max_trials = 6,
    )
    @test starved.outcome == :unresolved_amplitude
    @test starved.lo < 3.5 <= starved.hi
    @test length(starved.probes) == 6
    @test objects.state == [0.0]

    failing = solve_amplitude!(
        value, trials, (amplitude, Δτ) -> (; converged = false);
        Δτ = 1.0, target = 8.0, amplitude = 1.0,
    )
    @test failing.outcome == :solver_failed

    ok = _stepping_amplitude_step(objects, identity)
    @test_throws ArgumentError solve_amplitude!(value, trials, ok; Δτ = 1.0, target = 1.0, amplitude = 0.0)
    @test_throws ArgumentError solve_amplitude!(
        value, trials, ok; Δτ = 1.0, target = 1.0, amplitude = 2.0, amplitude_max = 1.0,
    )
    @test_throws ArgumentError solve_amplitude!(
        value, trials, ok; Δτ = 1.0, target = 1.0, amplitude = 1.0, rtol = 1.0,
    )
    @test_throws ArgumentError solve_amplitude!(
        value, trials, ok; Δτ = 1.0, target = 1.0, amplitude = 1.0, max_trials = 1,
    )
end

@testset "the amplitude solve recovers the injection an elastic solve needs" begin
    include(joinpath(_STEPPING_EXAMPLES, "event_stepping.jl"))
    include(joinpath(_STEPPING_EXAMPLES, "elliptical_cavity_setup.jl"))
    include(joinpath(_STEPPING_EXAMPLES, "injection_source.jl"))

    model = _stepping_replay_model()
    trials = capture_trials((; model.dr))
    pressure = (objects, _) -> sum(objects.dr.P) / length(objects.dr.P)

    # One elastic step is linear in the injected volume, so the amplitude that reaches a target
    # pressure is known from a single trial: this is the oracle the search must land on.
    unit = run_trial!(pressure, trials, Δτ -> model.step!(Δτ); Δτ = 1.0)
    @test unit.observation > 0
    target = 1.5 * unit.observation

    # The bracket is deliberately loose: each trial is a solve, and the oracle is exact, so the
    # question is whether the search lands on it, not how many digits a bisection can add.
    report = solve_amplitude!(
        pressure, trials, (amplitude, Δτ) -> model.step!(Δτ; Q_factor = amplitude);
        Δτ = 1.0, target, amplitude = 1.0, rtol = 2.0e-2, check_zero = false,
    )
    @test report.outcome == :amplitude_bracketed
    @test isapprox(report.amplitude, 1.5; rtol = 3.0e-2)
    # The committed end reaches the target and overshoots it by at most the bracket.
    @test target <= report.observation <= 1.03target
    # The committed step is the one the reported amplitude describes.
    @test isapprox(sum(model.dr.P) / length(model.dr.P), report.observation; rtol = 1.0e-12)
end
