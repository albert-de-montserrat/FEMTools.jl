using GLMakie, DelimitedFiles, Interpolations

include(joinpath(@__DIR__, "mesher_FEM.jl"))

@views function read_data(dat_file::String; resol::Int=128, visu_chk::Bool=false)
    print("Reading the data... ")
    data = readdlm(dat_file, Float64)
    data = data[3:end-2, :]
    xv_d = data[:, 1]
    bed_d = data[:, 2]
    surf_d = data[:, 3]
    xv = LinRange(xv_d[1], xv_d[end], resol)
    itp1 = interpolate((xv_d,), bed_d[:, 1], Gridded(Linear()))
    itp2 = interpolate((xv_d,), surf_d[:, 1], Gridded(Linear()))
    bed = itp1.(xv)[2:end-1]
    surf = itp2.(xv)[2:end-1]
    xv = xv[2:end-1]
    dat_len = length(xv)
    @assert dat_len == size(bed)[1] == size(surf)[1]

    xp = zeros(2 * dat_len)
    yp = zeros(2 * dat_len)
    tp = ones(Int64, 2 * dat_len)
    xp[1:dat_len] .= xv
    xp[dat_len+1:end] .= reverse(xv)
    yp[1:dat_len] .= bed
    yp[dat_len+1:end] .= reverse(surf)
    tp[1:dat_len] .= 1
    tp[dat_len+2:end-1] .= 2

    if visu_chk
        fig = Figure()
        ax = Axis(fig[1, 1]; aspect=DataAspect(), xlabel="x [km]", ylabel="y [km]")
        scatter!(ax, xp[tp.==1] ./ 1e3, yp[tp.==1] ./ 1e3; marker=:cross, label="Base")
        scatter!(ax, xp[tp.==2] ./ 1e3, yp[tp.==2] ./ 1e3; marker=:cross, label="Surface")
        axislegend(ax)
        display(fig)
    end

    println("done.")
    return xp, yp, tp
end

function plot_boundary_points(xp, yp, tp)
    fig = Figure()
    ax = Axis(fig[1, 1]; aspect=DataAspect(), xlabel="x/Lc", ylabel="y/Lc", title="Input boundary markers")
    scatter!(ax, xp[tp.==1], yp[tp.==1]; marker=:cross, label="Base")
    scatter!(ax, xp[tp.==2], yp[tp.==2]; marker=:cross, label="Surface")
    scatter!(ax, xp[tp.==4], yp[tp.==4]; marker=:cross, label="Free slip")
    axislegend(ax)
    display(fig)
    return fig
end

function plot_fem_mesh(mesh)
    xs = Vector{Float64}(undef, 6 * mesh.nel)
    ys = similar(xs)

    for iel in 1:mesh.nel
        i = 6 * (iel - 1)
        v1, v2, v3 = mesh.e2n[iel, :]
        xs[i+1] = mesh.xn[v1]; ys[i+1] = mesh.yn[v1]
        xs[i+2] = mesh.xn[v2]; ys[i+2] = mesh.yn[v2]
        xs[i+3] = mesh.xn[v2]; ys[i+3] = mesh.yn[v2]
        xs[i+4] = mesh.xn[v3]; ys[i+4] = mesh.yn[v3]
        xs[i+5] = mesh.xn[v3]; ys[i+5] = mesh.yn[v3]
        xs[i+6] = mesh.xn[v1]; ys[i+6] = mesh.yn[v1]
    end

    fig = Figure()
    ax = Axis(fig[1, 1]; aspect=DataAspect(), xlabel="x/Lc", ylabel="y/Lc", title="FEM mesh")
    linesegments!(ax, xs, ys; color=:black, linewidth=0.5)
    scatter!(ax, mesh.xn, mesh.yn; markersize=3, color=:dodgerblue, label="Nodes")

    if !isempty(mesh.boundary_edges)
        colors = [:red, :orange, :green, :purple, :brown, :deeppink, :cyan]
        for (i, marker) in enumerate(sort(collect(keys(mesh.boundary_edges))))
            edges = mesh.boundary_edges[marker]
            bx = Vector{Float64}(undef, 2 * length(edges))
            by = similar(bx)
            for (j, (v1, v2)) in enumerate(edges)
                k = 2 * (j - 1)
                bx[k+1] = mesh.xn[v1]; by[k+1] = mesh.yn[v1]
                bx[k+2] = mesh.xn[v2]; by[k+2] = mesh.yn[v2]
            end

            color = colors[mod1(i, length(colors))]
            linesegments!(ax, bx, by; color=color, linewidth=3, label="Boundary $marker")
            if haskey(mesh.boundary_nodes, marker)
                nodes = mesh.boundary_nodes[marker]
                scatter!(ax, mesh.xn[nodes], mesh.yn[nodes]; markersize=6, color=color)
            end
        end
        axislegend(ax)
    end

    display(fig)
    return fig
end

@views function MainArollaFEM()
    println("\n******** FEM STOKES ********")

    xp, yp, tp = read_data(joinpath(@__DIR__, "arolla51.txt"); resol=100, visu_chk=false)
    tp[(xp .> 2200.0) .& (xp .< 2500.0) .& (yp .< 2700.0)] .= 4

    Lc = maximum(xp) - minimum(xp)
    xp ./= Lc
    yp ./= Lc
    area = 8 / Lc^2

    plot_boundary_points(xp, yp, tp)

    mesh = MakeTriangleFEMMesh(
        1,
        1,
        minimum(xp),
        maximum(xp),
        minimum(yp),
        maximum(yp),
        -1,
        0.0,
        [1; 1; 1; 1],
        area;
        xp_in=xp,
        yp_in=yp,
        tp_in=tp,
    )

    println("Number of elements: ", mesh.nel)
    println("Number of nodes: ", mesh.nn)
    for marker in sort(collect(keys(mesh.boundary_nodes)))
        println("Boundary $marker nodes: ", length(mesh.boundary_nodes[marker]))
    end

    plot_fem_mesh(mesh)
end

# MainArollaFEM()
