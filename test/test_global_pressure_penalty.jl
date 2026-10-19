using Test, FEMTools, StaticArrays, KernelAbstractions, DomainSets
using DomainSets: ×

@testset "Global pressure penalty uses mesh-volume mean viscosity" begin
    for FP in (Float32, Float64), dim in (2, 3)
        if dim == 2
            ev = ReferenceElement(QuadraticElement{2, 7, FP})
            ep = ReferenceElement(LinearElement{2, 3, FP})
            coords = SVector{2, FP}[]
            for (a, b, c) in ((SVector(0,0), SVector(1,0), SVector(0,1)),
                    (SVector(2,0), SVector(5,0), SVector(2,1)))
                append!(coords, [a,b,c,(a+b)/2,(b+c)/2,(c+a)/2,(a+b+c)/3])
            end
            mv = Mesh(CPU(), coords, reshape(Int32.(1:14), 7, 2), ev)
        else
            ev = ReferenceElement(QuadraticElement{3, 27, FP})
            ep = ReferenceElement(LinearElement{3, 4, FP})
            domain = (FP(0)..FP(1)) × (FP(0)..FP(1)) × (FP(0)..FP(1))
            mv = Mesh(CPU(), domain, ev, (2, 1, 1))
            mv.coords .= [SVector(c[1] <= FP(.5) ? c[1] : FP(.5)+3*(c[1]-FP(.5)), c[2], c[3]) for c in mv.coords]
            update_geometry!(mv)
        end
        mesh = MixedMesh(mv, ep)
        phases = repeat(reshape(Int32[1,2], 1, :), length(ev), 1)
        dr = StokesDR(CPU(), mesh.nnodes, mesh.nnodesP,
            StokesMaterial(; η=FP.((2,10,999)), ηb=FP.((4,8,999)),
                G=FP.((Inf,Inf,Inf)), K=FP.((4,8,999)),
                α=FP.((0,0,0)), ρ0=FP.((1,1,1)),
                g=ntuple(_ -> zero(FP), dim), Tref=zero(FP)))
        γ = zeros(FP, mesh.nnodesP)
        assemble_viscosity_weighted_pressure_scaling!(γ, dr, mesh, FP(2), one(FP); phases_v=phases)
        local_γ, mass = copy(γ), copy(dr.M_P)
        assemble_viscosity_weighted_pressure_scaling!(γ, dr, mesh, FP(2), one(FP);
            phases_v=phases, penalty=:local)
        @test γ == local_γ
        assemble_viscosity_weighted_pressure_scaling!(γ, dr, mesh, FP(2), one(FP);
            phases_v=phases, penalty=:global)
        # Element volumes are 1:3; eta_mean=(2+3*10)/4=8, not the phase mean.
        for i in 1:2
            cap = FP((4,8)[i])
            @test γ[mesh.DoFsP[:,i]] ≈ fill(FP(16)*cap/(FP(16)+cap), size(mesh.DoFsP,1)) rtol=FP(2e-5)
        end
        @test dr.M_P == mass
        @test eltype(γ) == FP
        @test dr.η == FP.((2,10,999))
        assemble_viscosity_weighted_pressure_scaling!(dr.M_P, γ,
            mesh.el2n, mesh.DoFsP, mesh.geometry.geo_P, mesh.nels, ev, ep,
            phases, dr.η, FP(2), CPU(), 128; penalty=:global)
        @test γ ≈ fill(FP(8), length(γ)) rtol=FP(2e-5)
        @test_throws ArgumentError assemble_viscosity_weighted_pressure_scaling!(
            γ, dr, mesh, FP(2), one(FP); phases_v=phases, penalty=:invalid)
    end
end
