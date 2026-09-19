using Test

const _EXAMPLE_REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
_example_has_parse_error(x) = x isa Expr &&
    (x.head in (:error, :incomplete) || any(_example_has_parse_error, x.args))

@testset "maintained example scripts parse" begin
    repo_root = _EXAMPLE_REPO_ROOT
    scripts = [
        "examples/miniapps/thermal/1D_diffusion_FEMTools/1D_diffusion_FEMTools.jl",
        "examples/miniapps/thermal/1D_diffusion_FEMTools_color/1D_diffusion_FEMTools_color.jl",
        "examples/miniapps/thermal/2D_diffusion_FEMTools/2D_diffusion_FEMTools.jl",
        "examples/miniapps/thermal/3D_diffusion_FEMTools/3D_diffusion_FEMTools.jl",
        "examples/miniapps/thermal/1D_Poisson/1D_Poisson.jl",
        "examples/miniapps/thermal/1D_Poisson_AD/1D_Poisson_AD.jl",
        "examples/miniapps/thermal/1D_Poisson_Q2/1D_Poisson_Q2.jl",
        "examples/miniapps/thermal/2D_Poisson/2D_Poisson.jl",
        "examples/miniapps/thermal/2D_Poisson_AD/2D_Poisson_AD.jl",
        "examples/miniapps/thermal/2D_Poisson_AD_KA/2D_Poisson_AD_KA.jl",
        "examples/miniapps/thermal/3D_Poisson/3D_Poisson.jl",
        "examples/miniapps/thermal/3D_Poisson_AD/3D_Poisson_AD.jl",
        "examples/miniapps/thermal/3D_Poisson_AD_KA/3D_Poisson_AD_KA.jl",
        "examples/miniapps/stokes/2D_Elasticity_stress_postprocess/2D_Elasticity_stress_postprocess.jl",
        "examples/miniapps/stokes/2D_Elasticiy_DR_KA/2D_Elasticiy_DR_KA.jl",
        "examples/miniapps/stokes/2D_Elasticiy_Direct_KA/2D_Elasticiy_Direct_KA.jl",
        "examples/miniapps/thermal/2D_heat_diffusion/2D_heat_diffusion.jl",
        "examples/miniapps/thermal/2D_heat_diffusion_triangles/2D_heat_diffusion_triangles.jl",
        "examples/miniapps/thermal/2D_heat_diffusion_unstructured/2D_heat_diffusion_unstructured.jl",
        "examples/miniapps/thermal/2D_heat_diffusion_unstructured_T6/2D_heat_diffusion_unstructured_T6.jl",
        "examples/miniapps/thermal/3D_heat_diffusion_unstructured_hex/3D_heat_diffusion_unstructured_hex.jl",
        "examples/miniapps/stokes/stokes_2D_elastic_buildup/stokes_2D_elastic_buildup.jl",
        "examples/miniapps/stokes/stokes_2D_elastic_buildup_hole/stokes_2D_elastic_buildup_hole.jl",
        "examples/miniapps/stokes/sinking_block/sinking_block.jl",
        "examples/miniapps/stokes/stokes_2D_pure_shear/stokes_2D_pure_shear.jl",
        "examples/miniapps/stokes/stokes_2D_pure_shear_triangle/stokes_2D_pure_shear_triangle.jl",
        "examples/miniapps/stokes/stokes_2D_pure_shear_triangle_adv/stokes_2D_pure_shear_triangle_adv.jl",
        "examples/miniapps/stokes/stokes_2D_pure_shear_triangle_hole/stokes_2D_pure_shear_triangle_hole.jl",
        "examples/miniapps/stokes/ice_bridge_2D/ice_bridge_2D.jl",
        "examples/gmsh_meshing.jl",
        "examples/stokes/volcano/volcano_mesh_3D.jl",
        "examples/stokes/volcano/volcano_mesh_topo_3D.jl",
        "examples/stokes/volcano/volcano_thermal_stokes.jl",
        "examples/stokes/volcano/volcano_thermal_stokes_3D.jl",
        "examples/stokes/volcano/volcano_thermal_stokes_topo_3D.jl",
        "examples/miniapps/stokes/vevp/stokes_2D_pure_shear_triangle_adj.jl",
        "examples/miniapps/stokes/vevp/stokes_2D_shear_bands_triangle.jl",
        "examples/miniapps/stokes/solvi2D/Solvi2D_triangle.jl",
        "examples/miniapps/stokes/sinking_block/sinking_block_3D_setup.jl",
        "examples/miniapps/stokes/sinking_block_3D/sinking_block_3D.jl",
        "examples/miniapps/stokes/sinking_block_3D_adj/sinking_block_3D_adj.jl",
    ]
    for script in scripts
        @testset "$script" begin
            path = joinpath(repo_root, script)
            @test isfile(path)
            source = read(path, String)
            parsed = Meta.parseall(source; filename = path)
            @test !_example_has_parse_error(parsed)
            for m in eachmatch(r"include\(joinpath\(@__DIR__, ((?:\"[^\"]+\", )*)\"gmsh_meshing.jl\"\)\)", source)
                parents = [x.captures[1] for x in eachmatch(r"\"([^\"]+)\"", m.captures[1])]
                @test isfile(joinpath(dirname(path), parents..., "gmsh_meshing.jl"))
            end
        end
    end
end

@testset "documented example paths exist" begin
    docs_dir = joinpath(_EXAMPLE_REPO_ROOT, "docs", "src")
    for (dir, _, files) in walkdir(docs_dir), file in files
        endswith(file, ".md") || continue
        path = joinpath(dir, file)
        for m in eachmatch(r"examples/[A-Za-z0-9_./-]+\.jl", read(path, String))
            @test isfile(joinpath(_EXAMPLE_REPO_ROOT, m.match))
        end
    end
end

@testset "3D drivers include the shared setup" begin
    setup = joinpath(_EXAMPLE_REPO_ROOT, "examples", "miniapps", "stokes",
        "sinking_block", "sinking_block_3D_setup.jl")
    for name in ("sinking_block_3D", "sinking_block_3D_adj")
        path = joinpath(dirname(dirname(setup)), name, name * ".jl")
        source = read(path, String)
        m = only(collect(eachmatch(r"include\(joinpath\(@__DIR__, ((?:\"[^\"]+\", )*)\"sinking_block_3D_setup.jl\"\)\)", source)))
        parents = [x.captures[1] for x in eachmatch(r"\"([^\"]+)\"", m.captures[1])]
        @test normpath(joinpath(dirname(path), parents..., "sinking_block_3D_setup.jl")) == setup
    end
end
