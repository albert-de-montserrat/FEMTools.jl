using Documenter
using FEMTools

DocMeta.setdocmeta!(FEMTools, :DocTestSetup, :(using FEMTools); recursive=true)

makedocs(;
    modules=[FEMTools],
    authors="Albert de Montserrat",
    sitename="FEMTools.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://albert-de-montserrat.github.io/FEMTools.jl",
        edit_link="main",
    ),
    checkdocs=:exports,
    pages=[
        "Home" => "index.md",
        "Elements" => "elements.md",
        "Mesh" => "mesh.md",
        "Boundary Conditions" => "boundary_conditions.md",
        "Heat Diffusion" => "heat_diffusion.md",
        "Lithostatic Pressure" => "lithostatic_pressure.md",
        "Stokes" => [
            "Overview" => "stokes.md",
            "Sinking block" => "sinking_block.md",
            "Compressible inclusion" => "compressible_inclusion.md",
        ],
        "API" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/albert-de-montserrat/FEMTools.jl",
    devbranch="main",
)
