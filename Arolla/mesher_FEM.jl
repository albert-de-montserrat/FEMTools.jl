using Printf
import Triangulate

Base.@kwdef struct FEM_Mesh
    type::String = ""
    nel::Int64 = 0
    nn::Int64 = 0
    nnel::Int64 = 3
    order::Int64 = 1
    xn::Vector{Float64} = Float64[]
    yn::Vector{Float64} = Float64[]
    e2n::Matrix{Int64} = Matrix{Int64}(undef, 0, 0)
    Ω::Vector{Float64} = Float64[]
    xc::Vector{Float64} = Float64[]
    yc::Vector{Float64} = Float64[]
    bcn::Vector{Int64} = Int64[]
    hull_nodes::Vector{Int64} = Int64[]
    boundary_nodes::Dict{Int64, Vector{Int64}} = Dict{Int64, Vector{Int64}}()
    boundary_edges::Dict{Int64, Vector{NTuple{2, Int64}}} = Dict{Int64, Vector{NTuple{2, Int64}}}()
    phase::Vector{Float64} = Float64[]
    ke::Vector{Float64} = Float64[]
end

function build_triangle_input(xmin, xmax, ymin, ymax, inclusion, R, BC, no_pts_incl; xp_in=0, yp_in=0, tp_in=0)
    regions = Array{Float64}(undef, 4, 0)

    if inclusion == -1
        px = collect(xp_in)
        py = collect(yp_in)
        sx = collect(1:length(px))
        sy = sx .+ 1
        sy[end] = 1
        st = collect(tp_in)
        h1 = [xmin + 1e-13; ymin + 1e-13; 1.0; 0.0]
    else
        px = [xmin; xmax; xmax; xmin]
        py = [ymin; ymin; ymax; ymax]
        sx = [1; 2; 3; 4]
        sy = [2; 3; 4; 1]
        st = collect(BC)
        h1 = [xmin + 1e-13; ymin + 1e-13; 1.0; 0.0]
    end

    if inclusion == 1
        pts_l = length(px) + 1
        theta = LinRange(0.0, 2.0 * pi, no_pts_incl + 1)[1:end-1]
        center_x = (xmax + xmin) / 2.0
        center_y = (ymax + ymin) / 2.0
        X = center_x .+ R .* cos.(theta)
        Y = center_y .+ R .* sin.(theta)
        sx_incl = collect(pts_l:pts_l+no_pts_incl-1)
        sy_incl = sx_incl .+ 1
        sy_incl[end] = pts_l

        append!(px, X)
        append!(py, Y)
        append!(sx, sx_incl)
        append!(sy, sy_incl)
        append!(st, -ones(Int64, no_pts_incl))
        h2 = [0.0; 0.0; 2.0; 0.0]
        regions = hcat(h1, h2)
    else
        regions = hcat(h1)
    end

    return px, py, sx, sy, st, regions
end

function collect_boundary_groups(seglist, edgemarkerlist)
    boundary_nodes = Dict{Int64, Vector{Int64}}()
    boundary_edges = Dict{Int64, Vector{NTuple{2, Int64}}}()

    for ifac in axes(seglist, 2)
        marker = Int64(edgemarkerlist[ifac])
        marker == 0 && continue
        v1 = seglist[1, ifac]
        v2 = seglist[2, ifac]
        push!(get!(boundary_nodes, marker, Int64[]), v1, v2)
        push!(get!(boundary_edges, marker, NTuple{2, Int64}[]), (v1, v2))
    end

    for marker in keys(boundary_nodes)
        boundary_nodes[marker] = sort!(unique(boundary_nodes[marker]))
    end

    return boundary_nodes, boundary_edges
end

function MakeTriangleFEMMesh(
    nx,
    ny,
    xmin,
    xmax,
    ymin,
    ymax,
    inclusion,
    R,
    BC=[1; 1; 1; 1],
    area=((xmax - xmin) / nx) * ((ymax - ymin) / ny),
    no_pts_incl=Int64(floor(1.0 * pi * R / sqrt(((xmax - xmin) / nx)^2 + ((ymax - ymin) / ny)^2)));
    xp_in=0,
    yp_in=0,
    tp_in=0,
)
    px, py, sx, sy, st, regions = build_triangle_input(
        xmin,
        xmax,
        ymin,
        ymax,
        inclusion,
        R,
        BC,
        no_pts_incl;
        xp_in=xp_in,
        yp_in=yp_in,
        tp_in=tp_in,
    )

    triin = Triangulate.TriangulateIO()
    triin.pointlist = Matrix{Cdouble}(vcat(px', py'))
    triin.segmentlist = Matrix{Cint}(vcat(sx', sy'))
    triin.segmentmarkerlist = Vector{Int32}(st)
    triin.regionlist = Matrix{Cdouble}(regions)

    astring = @sprintf("%0.10lf", area)
    trimesh, vorout = Triangulate.triangulate("vQDpenq33o2IAa$(astring)", triin)

    nel = size(trimesh.trianglelist, 2)
    e2n = Matrix{Int64}(trimesh.trianglelist[1:3, :]')
    nn = maximum(e2n)
    xn = Vector{Float64}(trimesh.pointlist[1, 1:nn])
    yn = Vector{Float64}(trimesh.pointlist[2, 1:nn])
    bcn = Vector{Int64}(trimesh.pointmarkerlist[1:nn])
    phase = Vector{Float64}(trimesh.triangleattributelist[:])
    ke = ones(Float64, nel)

    Ω = zeros(Float64, nel)
    xc = zeros(Float64, nel)
    yc = zeros(Float64, nel)
    for iel in 1:nel
        v1, v2, v3 = e2n[iel, :]
        x1, y1 = xn[v1], yn[v1]
        x2, y2 = xn[v2], yn[v2]
        x3, y3 = xn[v3], yn[v3]
        Ω[iel] = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        xc[iel] = (x1 + x2 + x3) / 3.0
        yc[iel] = (y1 + y2 + y3) / 3.0
    end

    vor_edges = Matrix{Int64}(vorout.edgelist)
    seglist = Matrix{Int64}(trimesh.edgelist)
    hull_edges = (vor_edges[1, :] .<= 0) .| (vor_edges[2, :] .<= 0)
    hull_nodes = sort!(unique(vec(seglist[:, hull_edges])))
    boundary_nodes, boundary_edges = collect_boundary_groups(seglist, trimesh.edgemarkerlist)

    return FEM_Mesh(;
        type="P1Triangles",
        nel=nel,
        nn=nn,
        xn=xn,
        yn=yn,
        e2n=e2n,
        Ω=Ω,
        xc=xc,
        yc=yc,
        bcn=bcn,
        hull_nodes=hull_nodes,
        boundary_nodes=boundary_nodes,
        boundary_edges=boundary_edges,
        phase=phase,
        ke=ke,
    )
end

function MakeTriangleMesh(
    nx,
    ny,
    xmin,
    xmax,
    ymin,
    ymax,
    τr,
    inclusion,
    R,
    BC=[1; 1; 1; 1],
    area=((xmax - xmin) / nx) * ((ymax - ymin) / ny),
    no_pts_incl=Int64(floor(1.0 * pi * R / sqrt(((xmax - xmin) / nx)^2 + ((ymax - ymin) / ny)^2)));
    xp_in=0,
    yp_in=0,
    tp_in=0,
)
    return MakeTriangleFEMMesh(
        nx,
        ny,
        xmin,
        xmax,
        ymin,
        ymax,
        inclusion,
        R,
        BC,
        area,
        no_pts_incl;
        xp_in=xp_in,
        yp_in=yp_in,
        tp_in=tp_in,
    )
end
