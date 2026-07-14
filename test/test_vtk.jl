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

@testset "write_vtk writes vector point fields" begin
    coords = SVector{2, Float64}[SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(0.0, 1.0)]
    mesh = Mesh(coords, reshape(Int32[1, 2, 3], 3, 1))
    velocity = SVector{2, Float64}[SVector(1.0, 2.0), SVector(3.0, 4.0), SVector(5.0, 6.0)]

    text = _read_temp_vtk() do path
        write_vtk(path, mesh; point_data = (; velocity))
    end

    @test occursin("VECTORS velocity float\n1.0 2.0 0.0\n3.0 4.0 0.0", text)
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
