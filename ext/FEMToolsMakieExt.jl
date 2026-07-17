module FEMToolsMakieExt

using FEMTools
using Makie: Figure, Axis, Colorbar, poly!, lines!, Point2f, DataAspect, axislegend

const helpers_path = joinpath(@__DIR__, "..", "src", "visu", "helpers.jl")
Base.include_dependency(helpers_path)
include(helpers_path)

end # module FEMToolsMakieExt
