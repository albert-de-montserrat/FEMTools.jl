using Aqua

@testset "Aqua.jl" begin
    Aqua.test_all(
        FEMTools;
        unbound_args = false,
        deps_compat = (check_weakdeps = false,),
    )
end
