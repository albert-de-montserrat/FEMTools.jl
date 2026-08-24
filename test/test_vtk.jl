using Test

using FEMTools
using StaticArrays

function _read_temp_vtk(f)
    path, io = mktemp()
    close(io)
    try
        f(path)
        return read(path, String)
    finally
        rm(path; force = true)
    end
end

@testset "write_vtk writes legacy ASCII unstructured grid" begin
    coords = SVector{2, Float64}[
        SVector(0.0, 0.0),
        SVector(1.0, 0.0),
        SVector(0.0, 1.0),
    ]
    el2n = reshape(Int32[1, 2, 3], 3, 1)
    mesh = Mesh(coords, el2n)

    text = _read_temp_vtk() do path
        write_vtk(path, mesh; point_data = (; T = [1.0, 2.0, 3.0]), cell_data = (; Q = [4.0]), title = "tiny")
    end

    @test occursin("# vtk DataFile Version 3.0", text)
    @test occursin("tiny", text)
    @test occursin("DATASET UNSTRUCTURED_GRID", text)
    @test occursin("POINTS 3 float", text)
    @test occursin("CELLS 1 4", text)
    @test occursin("CELL_TYPES 1\n5", text)
    @test occursin("POINT_DATA 3", text)
    @test occursin("SCALARS T float 1", text)
    @test occursin("CELL_DATA 1", text)
    @test occursin("SCALARS Q float 1", text)
end

@testset "write_vtk linearizes high-order triangle corners" begin
    coords = SVector{2, Float64}[
        SVector(0.0, 0.0),
        SVector(1.0, 0.0),
        SVector(0.0, 1.0),
        SVector(0.5, 0.0),
        SVector(0.5, 0.5),
        SVector(0.0, 0.5),
    ]
    el2n = reshape(Int32[1, 2, 3, 4, 5, 6], 6, 1)
    mesh = Mesh(coords, el2n; order = 2)

    text = _read_temp_vtk() do path
        write_vtk(path, mesh; point_data = (; marker = collect(10.0:15.0)))
    end

    @test occursin("POINTS 3 float", text)
    @test occursin("CELLS 1 4\n3 0 1 2", text)
    @test occursin("SCALARS marker float 1\nLOOKUP_TABLE default\n10.0\n11.0\n12.0", text)
end

@testset "write_vtk validates field lengths" begin
    coords = SVector{2, Float64}[
        SVector(0.0, 0.0),
        SVector(1.0, 0.0),
        SVector(0.0, 1.0),
    ]
    el2n = reshape(Int32[1, 2, 3], 3, 1)
    mesh = Mesh(coords, el2n)

    path, io = mktemp()
    close(io)
    try
        @test_throws DimensionMismatch write_vtk(path, mesh; point_data = (; bad = [1.0, 2.0]))
        @test_throws DimensionMismatch write_vtk(path, mesh; cell_data = (; bad = [1.0, 2.0]))
    finally
        rm(path; force = true)
    end
end

@testset "write_stokes_vtk delegates through generic VTK writer" begin
    velocity_element = ReferenceElement(QuadraticElement{2, 6, Float64})
    pressure_element = ReferenceElement(LinearElement{2, 3, Float64})
    coords = SVector{2, Float64}[
        SVector(0.0, 0.0),
        SVector(1.0, 0.0),
        SVector(0.0, 1.0),
        SVector(0.5, 0.0),
        SVector(0.5, 0.5),
        SVector(0.0, 0.5),
    ]
    DoFs = Int32.(1:6)
    el2n = reshape(Int32[1, 2, 3, 4, 5, 6], 6, 1)
    DoFsP = reshape(Int32[1, 2, 3], 3, 1)
    el2nP = reshape(Int32[1, 2, 3], 3, 1)
    mesh = MixedMesh(velocity_element, pressure_element, coords, DoFs, el2n, DoFsP, el2nP)
    post = (;
        εxx = [1.0],
        εyy = [2.0],
        εzz = [3.0],
        εxy = [4.0],
        εII = [5.0],
        τxx = [6.0],
        τyy = [7.0],
        τzz = [8.0],
        τxy = [9.0],
        tauII = [10.0],
    )

    text = _read_temp_vtk() do path
        write_stokes_vtk(path, mesh, coords, el2nP, DoFsP, [10.0, 20.0, 30.0], 1.0:6.0, 2.0:7.0, post)
    end

    @test occursin("POINT_DATA 3", text)
    @test occursin("SCALARS P float 1", text)
    @test occursin("SCALARS Vx float 1", text)
    @test occursin("SCALARS V float 1", text)
    @test occursin("CELL_DATA 1", text)
    @test occursin("SCALARS strain_xx float 1", text)
    @test occursin("SCALARS tau_II float 1", text)
end

# Parse a legacy ASCII VTK unstructured grid back into arrays, so a round trip
# is checked on values rather than on exact formatting.
function _parse_vtk(text)
    lines = filter(!isempty, strip.(split(text, '\n')))
    points = Vector{Float64}[]
    cells = Vector{Int}[]
    cell_types = Int[]
    point_data = Dict{String, Vector{Float64}}()
    cell_data = Dict{String, Vector{Float64}}()
    npoints = ncells = 0
    section = nothing
    i = 1
    while i ≤ length(lines)
        fields = split(lines[i])
        if fields[1] == "POINTS"
            npoints = parse(Int, fields[2])
            for k in 1:npoints
                push!(points, parse.(Float64, split(lines[i + k])))
            end
            i += npoints
        elseif fields[1] == "CELLS"
            ncells = parse(Int, fields[2])
            for k in 1:ncells
                entry = parse.(Int, split(lines[i + k]))
                push!(cells, entry[2:end])
            end
            i += ncells
        elseif fields[1] == "CELL_TYPES"
            n = parse(Int, fields[2])
            append!(cell_types, parse(Int, lines[i + k]) for k in 1:n)
            i += n
        elseif fields[1] == "POINT_DATA"
            section = point_data
        elseif fields[1] == "CELL_DATA"
            section = cell_data
        elseif fields[1] == "SCALARS"
            n = section === point_data ? npoints : ncells
            # The LOOKUP_TABLE line sits between the header and the values.
            section[fields[2]] = [parse(Float64, lines[i + 1 + k]) for k in 1:n]
            i += 1 + n
        end
        i += 1
    end
    return (; points, cells, cell_types, point_data, cell_data)
end

@testset "write_vtk round-trips a 2-D triangle mesh" begin
    coords = SVector{2, Float64}[
        SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(0.0, 1.0), SVector(1.0, 1.0),
    ]
    el2n = Int32[1 2; 2 4; 3 3]
    mesh = Mesh(coords, el2n)
    T = [1.0, 2.0, 4.0, 8.0]
    Q = [7.0, 9.0]

    vtk = _parse_vtk(
        _read_temp_vtk() do path
            write_vtk(path, mesh; point_data = (; T), cell_data = (; Q), title = "pair")
        end
    )

    @test length(vtk.points) == 4
    @test [p[1:2] for p in vtk.points] == [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    @test all(p -> p[3] == 0.0, vtk.points)      # padded to three components
    @test vtk.cells == [[0, 1, 2], [1, 3, 2]]    # zero-based corner indices
    @test vtk.cell_types == [5, 5]
    @test vtk.point_data["T"] == T
    @test vtk.cell_data["Q"] == Q
end

@testset "write_vtk rejects unsupported cell layouts" begin
    @test_throws "cannot write VTK line cells with 4 local nodes" FEMTools._vtk_corner_rows(Val(1), 4)
    @test_throws "cannot write VTK 2D cells with 5 local nodes" FEMTools._vtk_corner_rows(Val(2), 5)
    @test_throws "cannot write VTK 3D cells with 5 local nodes" FEMTools._vtk_corner_rows(Val(3), 5)

    # The same message must reach a caller of the public writer.
    velocity_element = ReferenceElement(QuadraticElement{2, 6, Float64})
    pressure_element = ReferenceElement(LinearElement{2, 3, Float64})
    coords = SVector{2, Float64}[
        SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(0.0, 1.0),
        SVector(0.5, 0.0), SVector(0.5, 0.5), SVector(0.0, 0.5),
    ]
    el2n = reshape(Int32[1, 2, 3, 4, 5, 6], 6, 1)
    mesh = MixedMesh(
        velocity_element, pressure_element, coords, Int32.(1:6), el2n,
        reshape(Int32[1, 2, 3, 4, 5], 5, 1), reshape(Int32[1, 2, 3, 4, 5], 5, 1),
    )
    path, io = mktemp()
    close(io)
    try
        @test_throws "cannot write VTK 2D cells with 5 local nodes" write_vtk(path, mesh)
    finally
        rm(path; force = true)
    end
end
