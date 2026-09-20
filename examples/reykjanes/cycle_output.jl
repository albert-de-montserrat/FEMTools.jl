# VTK output for the recharge-and-intrusion cycles: what the numbers in the log are actually doing.
#
# `run_cycles` reports scalars — a reservoir pressure, a count of failed elements, a corridor fill, a
# threshold — and until now wrote nothing, so a run could be tabulated but not looked at. This writes
# the fields those scalars are read from, at every accepted step and on both sides of every event, so
# that a threshold can be inspected rather than only believed.
#
# The detector's own quantities are written beside the stress, because the event is a statement about
# them: which elements failed and how much of each one's quadrature did, the hydraulic margin that
# decides it, the fixed corridor and how full each of its bins is, the band the intrusion opens, and
# the connected component the adjacency rule would have used. Reading a VTK file should make it
# obvious why a given step was or was not an event.

using Printf

"""
    build_cycle_writer(model, detector, protocol; out_dir, prefix = "reykjanes_cycle") -> write(tag, state; band, time)

Return the function that writes one VTK snapshot of the current state of `model`, numbered
sequentially, and appends a line to `index.txt` in `out_dir` saying what that number was.

`state` is a [`build_event_detector`](@ref) result, so the snapshot carries the same failure field the
protocol judged, not a recomputation of it. `band` is the element set the intrusion opens, empty
except at an event.

Point data is pressure and velocity in SI; cell data is the strain rate, the deviatoric stress, the
phase and temperature, and the detector fields:

- `failed`, one where the element failed under the protocol's criterion, and `failed_fraction`, the
  fraction of that element's quadrature that did — the weight the corridor actually integrates, and
  the field to look at when a bin sits near its fill threshold;
- `hydraulic_margin_MPa` (`P_f − s₃ − T₀`, the tensile criterion itself), `s3_MPa` and `F_MPa`;
- `corridor_bin`, zero outside the fixed strip and the bin index inside it, with
  `corridor_bin_fill`, that bin's failed-area fraction copied onto its members, so the column rule
  can be read off the picture;
- `in_band`, the elements the dike opens, and `in_component`, the face-connected failed region
  reachable from the reservoir, kept as a diagnostic so the two detectors can be compared by eye.
"""
function build_cycle_writer(model, detector, protocol; out_dir, prefix = "reykjanes_cycle")
    (; mesh_stokes, el2n_v_cpu, el2nP_cpu, DoFsP_cpu, coords_cpu, cell_phase) = model
    (; L_c, t_c, σ_c, geo_v, element_v, dr, τ, thermal) = model
    nels = mesh_stokes.nels
    mkpath(out_dir)

    manifest = joinpath(out_dir, "index.txt")
    open(manifest, "w") do io
        println(io, "# VTK snapshots of run_cycles. Corridor: $(length(detector.corridor.cells)) ",
            "elements in $(length(detector.corridor.counts)) bins, counts $(detector.corridor.counts).")
        @printf(io, "# %5s  %-12s %10s %11s %8s %9s %6s  %s\n",
            "index", "tag", "time_kyr", "P_res_MPa", "failed", "corridor", "band", "file")
    end

    index = Ref(0)
    corridor_bin = detector.corridor.bin

    return function write_cycle_vtk(tag, state; band = Int32[], time = 0.0)
        index[] += 1
        P_cpu = Array(dr.P)
        vx_cpu = Array(dr.v.x)
        vy_cpu = Array(dr.v.y)
        post = compute_strain_rate_stress_postprocess(
            vx_cpu, vy_cpu, el2n_v_cpu, Array(geo_v), map(Array, τ), element_v,
        )
        T_cpu = Array(thermal.T)
        el_T = [mean(T_cpu[el2n_v_cpu[1:3, iel]]) for iel in 1:nels]

        # The same integration-point flags the detector reduced, so the picture cannot disagree with
        # the verdict that was recorded.
        points = failed_points(state.diagnostics, protocol.criterion)
        failed_fraction = [corridor_failed_weight(points, iel) for iel in 1:nels]
        element_mean(A) = [mean(@view A[:, iel]) for iel in 1:nels]

        in_component = zeros(Int, nels)
        for iel in state.component
            in_component[iel] = 1
        end
        in_band = zeros(Int, nels)
        for iel in band
            in_band[iel] = 1
        end
        corridor_bin_fill = [k == 0 ? 0.0 : state.fractions[k] for k in corridor_bin]

        vtk_path = joinpath(out_dir, @sprintf("%s_%04d.vtk", prefix, index[]))
        write_stokes_vtk(
            vtk_path, mesh_stokes, coords_cpu .* L_c, el2nP_cpu, DoFsP_cpu,
            P_cpu .* σ_c, vx_cpu .* (L_c / t_c), vy_cpu .* (L_c / t_c),
            # Strain-rate fields scale with 1/t_c, stress fields with σ_c.
            merge(
                map(f -> f .* σ_c, post),
                map(f -> f ./ t_c, post[(:εxx, :εyy, :εzz, :εxy, :εII)]),
            );
            title = "reykjanes recharge-and-intrusion cycles",
            cell_data = (;
                phase = cell_phase,
                T = el_T,
                failed = Int.(state.failed),
                failed_fraction,
                hydraulic_margin_MPa = element_mean(state.diagnostics.hydraulic_margin) ./ 1.0e6,
                s3_MPa = element_mean(state.diagnostics.s3) ./ 1.0e6,
                F_MPa = element_mean(state.diagnostics.F) ./ 1.0e6,
                corridor_bin,
                corridor_bin_fill,
                in_band,
                in_component,
            ),
        )
        open(manifest, "a") do io
            @printf(
                io, "  %5d  %-12s %10.4f %11.4f %8d %9.3f %6d  %s\n",
                index[], tag, time / kyr, state.P_reservoir / 1.0e6, count(state.failed),
                state.fraction, length(band), basename(vtk_path),
            )
        end
        return vtk_path
    end
end
