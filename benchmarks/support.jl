# Host-side helpers shared by the exact-field benchmarks: quadrature sampling,
# quadrature-weighted errors, convergence-history archives, and comparison
# figures. Device fields are copied to the host once per call; nothing here
# runs inside a solver iteration.

using FEMTools
using JLD2
import GLMakie
using GLMakie: Figure, Axis, Colorbar, poly!, Point2f, DataAspect

"""
    quadrature_samples(mesh, geometry, element) -> (; points, weights)

Physical quadrature points and weights `|det J| w`, indexed `[q, e]`.
"""
function quadrature_samples(mesh, geometry, element)
    coords, connectivity = Array(mesh.coords), Array(mesh.el2n)
    Nq = shape_function_values(element)
    gradients = shape_function_gradients(element)
    geo = Array(geometry)
    points = [sum(Nq[q][a] * coords[connectivity[a, e]] for a in axes(connectivity, 1))
              for q in eachindex(Nq), e in axes(connectivity, 2)]
    weights = [FEMTools.element_geometry(geo, e, gradients)[q][2]
               for q in eachindex(Nq), e in axes(connectivity, 2)]
    return (; points, weights)
end

"""
    sample_field(field, connectivity, shapes) -> Matrix

Interpolate a nodal field to the quadrature points, indexed `[q, e]`.
"""
function sample_field(field, connectivity, shapes)
    values = Array(field)
    field_nodes = Array(connectivity)
    return [sum(shapes[q][a] * values[field_nodes[a, e]] for a in axes(field_nodes, 1))
            for q in eachindex(shapes), e in axes(field_nodes, 2)]
end

"""
    nodal_load(f, connectivity, shapes, samples, nnodes) -> Vector

Assemble `∫ Nᵢ f dΩ` from quadrature samples into a host vector of length `nnodes`.
"""
function nodal_load(f, connectivity, shapes, samples, nnodes)
    load = zeros(nnodes)
    for e in axes(connectivity, 2), q in eachindex(shapes), a in axes(connectivity, 1)
        load[connectivity[a, e]] += shapes[q][a] * f(samples.points[q, e]) * samples.weights[q, e]
    end
    return load
end

"""
    field_error(numerical, exact, weights) -> (; absolute, relative)

Quadrature-weighted L² error. Tuples of component arrays give the error of the
vector field.
"""
field_error(numerical, exact, weights) =
    _weighted_error(abs2.(numerical .- exact), abs2.(exact), weights)
field_error(numerical::Tuple, exact::Tuple, weights) =
    _weighted_error(sum(map((n, x) -> abs2.(n .- x), numerical, exact)),
                    sum(map(x -> abs2.(x), exact)), weights)

function _weighted_error(squared_error, squared_reference, weights)
    absolute = sqrt(sum(weights .* squared_error))
    reference = sqrt(sum(weights .* squared_reference))
    return (; absolute, relative = iszero(reference) ? NaN : absolute / reference)
end

"""
    remove_mean!(field, weights) -> field

Subtract the quadrature-weighted mean, fixing the pressure gauge.
"""
remove_mean!(field, weights) = field .-= sum(weights .* field) / sum(weights)

cell_average(field, weights) = vec(sum(weights .* field; dims = 1) ./ sum(weights; dims = 1))

"""
    save_convergence_history(path; convergence_history, metadata) -> path

Write the history and metadata to a JLD2 archive, creating its directory.
"""
function save_convergence_history(path; convergence_history, metadata)
    mkpath(dirname(path))
    jldsave(path; convergence_history, metadata)
    return path
end

"""
    comparison_figure(coords, connectivity, nvertices, weights, fields; size) -> Figure

One row per `(name, numerical, analytical)` entry of `fields`: numerical,
analytical, and absolute-error cell averages on filled elements, which keep
material interfaces sharp. `nvertices` is the number of corner nodes per cell.
"""
function comparison_figure(coords, connectivity, nvertices, weights, fields; size)
    pts = [Point2f(c) for c in coords]
    polys = [[pts[connectivity[a, e]] for a in 1:nvertices] for e in axes(connectivity, 2)]
    fig = Figure(; size)
    for (row, (name, numerical, analytical)) in enumerate(fields)
        el_num = cell_average(numerical, weights)
        el_anal = cell_average(analytical, weights)
        el_error = cell_average(abs.(numerical .- analytical), weights)
        clims = extrema(vcat(el_num, el_anal))
        for (i, (label, field)) in enumerate((("FEMTools", el_num),
                                             ("analytics", el_anal),
                                             ("Absolute error", el_error)))
            ax = Axis(fig[row, 2i - 1]; aspect = DataAspect(),
                      title = "$name ($label)", xlabel = "x", ylabel = "y")
            colorrange = i == 3 ? extrema(field) : clims
            plot = poly!(ax, polys; color = field, colormap = :vik, colorrange, strokewidth = 0)
            Colorbar(fig[row, 2i], plot)
        end
    end
    return fig
end

"""
    convergence_figure(iterations, series; title, xlabel) -> Figure

Residual histories on a log axis; `series` holds `label => residuals` pairs.
"""
function convergence_figure(iterations, series; title, xlabel)
    fig = Figure(size = (900, 500))
    ax = Axis(fig[1, 1]; title, xlabel, ylabel = "Residual", yscale = log10)
    for (label, residuals) in series
        # Display exact zeros at the floating-point precision floor on log axes.
        GLMakie.lines!(ax, iterations, max.(residuals, eps(Float64)); label)
    end
    GLMakie.axislegend(ax)
    return fig
end

"""
    show_and_save(fig, path; show_plot, write_output)

Display `fig` in its own window and/or save it to `path`.
"""
function show_and_save(fig, path; show_plot, write_output)
    show_plot && display(GLMakie.Screen(), fig)
    write_output && GLMakie.save(path, fig)
    return nothing
end
