# Heat diffusion on a rectangle with circular holes — 6-node quadratic triangles.
# Mesh generated with Triangulate.jl (`o2` flag produces T6 elements).
using Pkg; Pkg.activate("examples")

using Triangulate
using StaticArrays
using KernelAbstractions
using Printf
using WriteVTK
using GLMakie
using GeometryBasics
using TimerOutputs
using FEMTools
using LinearAlgebra

const backend   = CPU()
const workgroup = 128

# ---------------------------------------------------------------------------
# Geometry precomputation
# ---------------------------------------------------------------------------

function precompute_geometry(coords, el2n, nels, element::ReferenceElement{T}) where T <: AbstractElement{2, N} where N
    ip    = element.integration_points
    NQ    = length(ip.ω)
    ξq    = ntuple(q -> SVector(ip.ξ[q], ip.η[q]), NQ)
    ∂N∂ξq = ntuple(q -> eval_shape_function_jacobian(element, ξq[q]), NQ)
    geo   = KernelAbstractions.allocate(backend, NTuple{NQ, Tuple{SMatrix{N, 2, Float64, 2N}, Float64}}, nels)
    FEMTools.precompute_geometry_kernel!(backend, workgroup)(geo, coords, el2n, ∂N∂ξq, ip.ω, Val(N); ndrange = nels)
    KernelAbstractions.synchronize(backend)
    return geo
end

@kernel function precompute_mixed_geometry_kernel!(
    geo,
    @Const(coords),
    @Const(el2n_geometry),
    ∂N_geometry∂ξq,
    ∂N_field∂ξq,
    ω,
    ::Val{Ngeometry},
) where Ngeometry
    iel = @index(Global)
    local_nodes = FEMTools.local_nodes_of(el2n_geometry, iel, Val(Ngeometry))
    c = FEMTools.element_coordinate_matrix(coords, local_nodes)
    geo[iel] = ntuple(Val(length(ω))) do q
        J = c' * ∂N_geometry∂ξq[q]
        (∂N_field∂ξq[q] * inv(J), abs(det(J)) * ω[q])
    end
end

function precompute_mixed_geometry(
    coords,
    el2n_geometry,
    nels,
    geometry_element::ReferenceElement{TG},
    field_element::ReferenceElement{TF},
) where {TG <: AbstractElement{2, NG}, TF <: AbstractElement{2, NF}} where {NG, NF}
    ip    = geometry_element.integration_points
    NQ    = length(ip.ω)
    ξq    = ntuple(q -> SVector(ip.ξ[q], ip.η[q]), NQ)
    ∂Ng∂ξq = ntuple(q -> eval_shape_function_jacobian(geometry_element, ξq[q]), NQ)
    ∂Nf∂ξq = ntuple(q -> eval_shape_function_jacobian(field_element, ξq[q]), NQ)
    geo   = KernelAbstractions.allocate(backend, NTuple{NQ, Tuple{SMatrix{NF, 2, Float64, 2NF}, Float64}}, nels)
    precompute_mixed_geometry_kernel!(backend, workgroup)(
        geo,
        coords,
        el2n_geometry,
        ∂Ng∂ξq,
        ∂Nf∂ξq,
        ip.ω,
        Val(NG);
        ndrange = nels,
    )
    KernelAbstractions.synchronize(backend)
    return geo
end

function precompute_discontinuous_geometry(
    mesh::MixedMesh,
    geometry_element::ReferenceElement,
    field_element::ReferenceElement,
)
    return precompute_mixed_geometry(mesh.coords, mesh.el2n, mesh.nels, geometry_element, field_element)
end

@kernel function precompute_stokes_geometry_kernel!(
    geo,
    @Const(coords),
    @Const(el2n),
    ∂NV∂ξq,
    ∂NP∂ξq,
    ω,
    ::Val{NV},
) where NV
    iel = @index(Global)
    local_nodes = FEMTools.local_nodes_of(el2n, iel, Val(NV))
    c = FEMTools.element_coordinate_matrix(coords, local_nodes)
    geo[iel] = ntuple(Val(length(ω))) do q
        J    = c' * ∂NV∂ξq[q]
        Jinv = inv(J)
        (∂NV∂ξq[q] * Jinv, ∂NP∂ξq[q] * Jinv, abs(det(J)) * ω[q])
    end
end

function precompute_geometry_stokes(
    coords,
    el2n,
    nels,
    element_V::ReferenceElement{TV},
    element_P::ReferenceElement{TP},
) where {TV <: AbstractElement{2, NV}, TP <: AbstractElement{2, NP}} where {NV, NP}
    ip_V = element_V.integration_points
    ip_P = element_P.integration_points
    NQV  = length(ip_V.ω)
    NQP  = length(ip_P.ω)

    ξq_V = ntuple(q -> SVector(ip_V.ξ[q], ip_V.η[q]), NQV)
    ξq_P = ntuple(q -> SVector(ip_P.ξ[q], ip_P.η[q]), NQP)

    ∂NV∂ξ_at_Vq = ntuple(q -> eval_shape_function_jacobian(element_V, ξq_V[q]), NQV)
    ∂NP∂ξ_at_Vq = ntuple(q -> eval_shape_function_jacobian(element_P, ξq_V[q]), NQV)
    ∂NV∂ξ_at_Pq = ntuple(q -> eval_shape_function_jacobian(element_V, ξq_P[q]), NQP)
    ∂NP∂ξ_at_Pq = ntuple(q -> eval_shape_function_jacobian(element_P, ξq_P[q]), NQP)

    TV_ip = NTuple{NQV, Tuple{SMatrix{NV, 2, Float64, 2NV}, SMatrix{NP, 2, Float64, 2NP}, Float64}}
    TP_ip = NTuple{NQP, Tuple{SMatrix{NV, 2, Float64, 2NV}, SMatrix{NP, 2, Float64, 2NP}, Float64}}

    geo_at_V_ips = KernelAbstractions.allocate(backend, TV_ip, nels)
    geo_at_P_ips = KernelAbstractions.allocate(backend, TP_ip, nels)

    precompute_stokes_geometry_kernel!(backend, workgroup)(
        geo_at_V_ips, coords, el2n, ∂NV∂ξ_at_Vq, ∂NP∂ξ_at_Vq, ip_V.ω, Val(NV);
        ndrange = nels,
    )
    precompute_stokes_geometry_kernel!(backend, workgroup)(
        geo_at_P_ips, coords, el2n, ∂NV∂ξ_at_Pq, ∂NP∂ξ_at_Pq, ip_P.ω, Val(NV);
        ndrange = nels,
    )
    KernelAbstractions.synchronize(backend)

    return (v_ips = geo_at_V_ips, p_ips = geo_at_P_ips)
end

"""
    build_mesh_T6(; Lx, Ly, holes, n_circle=64, max_area=nothing)

Triangulate the rectangle [-Lx,Lx]×[-Ly,0] with circular holes and produce
6-node quadratic triangles using Triangulate.jl's `o2` flag.

`holes` is a vector of `(cx, cy, r)` tuples. Returns
`(coords, el2n, outer_nodes, hole_nodes_per_hole)` where `el2n` is 6×nels
and nodes 1–3 of each column are corners, nodes 4–6 are edge midpoints
(matching `QuadraticElement{2,6}` ordering).
"""
function build_mesh_T6(; Lx, Ly, holes, n_circle=64, max_area=nothing)
    # ---- outer rectangle (counter-clockwise) ----
    rect_pts  = Cdouble[-Lx  Lx  Lx -Lx;
                        -Ly -Ly  0.0  0.0]
    rect_segs = Cint[1 2; 2 3; 3 4; 4 1]'

    all_pts  = Matrix{Cdouble}(rect_pts)
    all_segs = Matrix{Cint}(rect_segs)
    hole_xy  = Matrix{Cdouble}(undef, 2, length(holes))

    for (h, (cx, cy, r)) in enumerate(holes)
        θ        = range(0, 2π; length = n_circle + 1)[1:end-1]
        circ_pts = Matrix{Cdouble}(hcat(cx .+ r .* cos.(θ), cy .+ r .* sin.(θ))')
        n_so_far = size(all_pts, 2)
        circ_segs = Matrix{Cint}(hcat([
            [n_so_far + i; n_so_far + mod1(i + 1, n_circle)] for i in 1:n_circle
        ]...))
        all_pts  = hcat(all_pts,  circ_pts)
        all_segs = hcat(all_segs, circ_segs)
        hole_xy[:, h] = [cx; cy]
    end

    tio = TriangulateIO()
    tio.pointlist   = all_pts
    tio.segmentlist = all_segs
    tio.holelist    = hole_xy

    # o2 = generate second-order (6-node) elements
    flags = isnothing(max_area) ? "pqo2Q" : "pq30o2a$(max_area)Q"
    result, _ = triangulate(flags, tio)

    pts  = result.pointlist    # 2 × nnodes
    tris = result.trianglelist # 6 × nels  (1-based, o2 ordering matches QuadraticElement{2,6})

    nnodes = size(pts, 2)
    coords = [SVector{2, Float64}(pts[1, i], pts[2, i]) for i in 1:nnodes]
    el2n   = Matrix{Int32}(tris)

    tol_lin = max(Lx, Ly) * 1e-8
    outer_nodes = Int32[i for i in 1:nnodes if
        abs(coords[i][2] + Ly) < tol_lin ||
        abs(coords[i][2])      < tol_lin ||
        abs(coords[i][1] + Lx) < tol_lin ||
        abs(coords[i][1] - Lx) < tol_lin]

    hole_nodes_per_hole = map(holes) do (cx, cy, r)
        tol_circ = r * 1e-4
        Int32[i for i in 1:nnodes if
            abs(sqrt((coords[i][1] - cx)^2 + (coords[i][2] - cy)^2) - r) < tol_circ]
    end

    return coords, el2n, outer_nodes, hole_nodes_per_hole
end

####

@generated function compute_velocity_divergence(v::NTuple{N, SVector{M, T}}, ∂N∂x_v) where {N, M, T}
    quote
        ## Effectively doing the following
        # ∂Vx∂x = ∂N∂x_v[:,1]' * vx
        # ∂Vy∂y = ∂N∂x_v[:,2]' * vy
        # div_el = (∂Vx∂x + ∂Vy∂y)
        @inline 
        ∇V = zero(T)
        # a bit slower than the lines above, but should put less pressure on the registers
        Base.@nexprs $N j-> begin # j corresponds to the x,y,z-dimension
            v_j = v[j]
            Base.@nexprs $M i-> begin
                ∇V += ∂N∂x_v[i,j] * v_j[i]
            end
        end
        return ∇V
    end
end

@inline function integrate_PH_pressure_residual(v::NTuple, Ploc::SVector{N},  P0loc, Tloc, T0loc, geo_v_el, geo_P_el, phase_loc, α, ηb, ξ, Δt, Nq) where N
    RP_e = zero(Ploc)
    for q in eachindex(geo_P_el)
        ∂N∂x_v, = geo_v_el[q] # velocity NOTE: this should be ∂N∂x_v evaluated at linear 3 ips
        _, dΩ   = geo_P_el[q] # pressure
        Nv      = Nq[q]

        # project parameters to integration point
        ηbq = FEMTools.interp2ip_phase(Nv, ηb, ξ, phase_loc)
        αq  = FEMTools.interp2ip_phase(Nv, α, phase_loc)
        # project ∂P∂t to integration points
        ∂P∂t = FEMTools.interp2ip(
            Nv,
            (P, P0) ->  (P - P0) / ηbq, 
            (Ploc, P0loc)
        )
        # project ∂T∂t to integration point
        ∂T∂t = FEMTools.interp2ip(
            Nv,
            (T, T0) ->  αq * (T - T0) / Δt, 
            (Tloc, T0loc)
        )
        # project divergence to integration point
        ∇V = compute_velocity_divergence(v, ∂N∂x_v)
        # compute pressure residual 
        RP_e += SVector{N}(ntuple(
                    i -> Nv[i] * (-∇V -  (∂P∂t + ∂T∂t)) * dΩ,
                    Val(N),
                ))
                
    end
    return RP_e
end

@inline function integrate_momentum_residual(
    v::NTuple{2, SVector{N, T}},
    Ploc::SVector{NP},
    geo_v_el,
    phase_loc, η,
    Nq,   # velocity shape-fn values at velocity IPs
    NqP,  # pressure shape-fn values at velocity IPs
) where {N, T, NP}
    vxloc, vyloc = v
    Rv_x = zero(SVector{N, T})
    Rv_y = zero(SVector{N, T})
    for q in eachindex(geo_v_el)
        ∂N∂x, dΩ = geo_v_el[q]
        Nv = Nq[q]
        # Strain rates
        ∇vx = ∂N∂x' * vxloc   # [∂vx/∂x, ∂vx/∂y]
        ∇vy = ∂N∂x' * vyloc   # [∂vy/∂x, ∂vy/∂y]
        εxx = ∇vx[1]
        εyy = ∇vy[2]
        εxy = (∇vx[2] + ∇vy[1]) / 2
        # Deviatoric stress (plane strain, εzz = 0)
        tr  = (εxx + εyy) / 3
        ηq  = FEMTools.interp2ip_phase(Nv, η, phase_loc)
        τxx = 2 * ηq * (εxx - tr)
        τyy = 2 * ηq * (εyy - tr)
        τxy = 2 * ηq * εxy
        # Pressure at this quadrature point
        Pq  = dot(NqP[q], Ploc)
        # x-momentum: ∫ (∂Nᵢ/∂x·(τxx−P) + ∂Nᵢ/∂y·τxy) dΩ
        Rv_x += (∂N∂x[:, 1] * (τxx - Pq) + ∂N∂x[:, 2] * τxy) * dΩ
        # y-momentum: ∫ (∂Nᵢ/∂y·(τyy−P) + ∂Nᵢ/∂x·τxy) dΩ
        Rv_y += (∂N∂x[:, 2] * (τyy - Pq) + ∂N∂x[:, 1] * τxy) * dΩ
    end
    return Rv_x, Rv_y
end

# NqP: pressure shape-function values evaluated at velocity quadrature points,
# needed to interpolate P to velocity IPs for the pressure-gradient term.
@inline function integrate_momentum_x_residual(
    v::NTuple{2, SVector{N, T}},
    Ploc::SVector{NP},
    geo_v_el,
    phase_loc, η,
    Nq,   # velocity shape-fn values at velocity IPs
    NqP,  # pressure shape-fn values at velocity IPs
) where {N, T, NP}
    vxloc, vyloc = v
    Rv_x = zero(SVector{N, T})
    for q in eachindex(geo_v_el)
        ∂N∂x, dΩ = geo_v_el[q]
        Nv = Nq[q]
        # Strain rates
        ∇vx = ∂N∂x' * vxloc   # [∂vx/∂x, ∂vx/∂y]
        ∇vy = ∂N∂x' * vyloc   # [∂vy/∂x, ∂vy/∂y]
        εxx = ∇vx[1]
        εyy = ∇vy[2]
        εxy = (∇vx[2] + ∇vy[1]) / 2
        # Deviatoric stress (plane strain, εzz = 0)
        tr  = (εxx + εyy) / 3
        ηq  = FEMTools.interp2ip_phase(Nv, η, phase_loc)
        τxx = 2 * ηq * (εxx - tr)
        τxy = 2 * ηq * εxy
        # Pressure at this quadrature point
        Pq  = dot(NqP[q], Ploc)
        # x-momentum: ∫ (∂Nᵢ/∂x·(τxx−P) + ∂Nᵢ/∂y·τxy) dΩ
        Rv_x += (∂N∂x[:, 1] * (τxx - Pq) + ∂N∂x[:, 2] * τxy) * dΩ
    end
    return Rv_x
end

@inline function integrate_momentum_y_residual(
    v::NTuple{2, SVector{N, T}},
    Ploc::SVector{NP},
    geo_v_el,
    phase_loc, η,
    Nq,   # velocity shape-fn values at velocity IPs
    NqP,  # pressure shape-fn values at velocity IPs
) where {N, T, NP}
    vxloc, vyloc = v
    Rv_y = zero(SVector{N, T})
    for q in eachindex(geo_v_el)
        ∂N∂x, dΩ = geo_v_el[q]
        Nv = Nq[q]
        # Strain rates
        ∇vx = ∂N∂x' * vxloc
        ∇vy = ∂N∂x' * vyloc
        εxx = ∇vx[1]
        εyy = ∇vy[2]
        εxy = (∇vx[2] + ∇vy[1]) / 2
        # Deviatoric stress (plane strain, εzz = 0)
        tr  = (εxx + εyy) / 3
        ηq  = FEMTools.interp2ip_phase(Nv, η, phase_loc)
        τyy = 2 * ηq * (εyy - tr)
        τxy = 2 * ηq * εxy
        # Pressure at this quadrature point
        Pq  = dot(NqP[q], Ploc)
        # y-momentum: ∫ (∂Nᵢ/∂y·(τyy−P) + ∂Nᵢ/∂x·τxy) dΩ
        Rv_y += (∂N∂x[:, 2] * (τyy - Pq) + ∂N∂x[:, 1] * τxy) * dΩ
    end
    return Rv_y
end


###


FP   = Float64
TDev = FEMTools.TA(backend)

max_area = 1e5

# Domain geometry
Lx, Ly = 15e3, 30e3
holes  = [(-7e3, -10e3, 2.5e3),   # (cx, cy, r), T = 1173 K
            (  0e0, -11.5e3, 2e3)] # (cx, cy, r), T = 1273 K
T_holes = FP[873, 1173]

# Generate mesh
coords_cpu, el2n_cpu, outer_nodes, hole_nodes_per_hole = build_mesh_T6(;
    Lx, Ly, holes,
    n_circle = 64,
    max_area = max_area,
)
mesh_thermal = FEMTools.Mesh(backend, coords_cpu, el2n_cpu)
@printf("mesh: %d nodes, %d elements\n", mesh_thermal.nnodes, mesh_thermal.nels)

el2nP, DoFsP, p_coords = FEMTools.build_discontinuous_linear_mesh(coords_cpu, el2n_cpu);

element_vel = ReferenceElement(QuadraticElement{2, 6, FP})
element_P = ReferenceElement(LinearElement{2, 3, FP})

mesh_stokes = MixedMesh(
    element_vel,
    element_P,
    coords_cpu,
    collect(1:length(coords_cpu)),
    el2n_cpu,
    DoFsP,
    el2nP,
) 

geo_v = precompute_geometry(mesh_stokes.coords, mesh_stokes.el2n, mesh_stokes.nels, element_vel)
geo_P = precompute_geometry(mesh_stokes.coords, mesh_stokes.el2nP, mesh_stokes.nels, element_P)
# geo_P   = precompute_discontinuous_geometry(mesh_stokes, element_vel, element_P)

Nvq = shape_function_values(element_vel, element_P.integration_points)
NP = shape_function_values(element_P)

iel=q=1

geo_v_el = geo_v[iel]
geo_P_el = geo_P[iel]
∂N∂x_v, dΩ = geo_v_el[iel][q]
∂N∂x_P, = geo_P[iel][q]

vxloc = @SVector rand(6)
vyloc = @SVector rand(6)
Tloc = @SVector rand(3)
T0loc = @SVector rand(3)
Ploc = @SVector rand(3)
P0loc = @SVector rand(3)
ηbloc = @SVector rand(3)

# @inline function compute_velocity_divergence(vx, vy, ∂N∂x_v)
#     ## Effectively doing the following
#     # ∂Vx∂x = ∂N∂x_v[:,1]' * vx
#     # ∂Vy∂y = ∂N∂x_v[:,2]' * vy
#     # div_el = (∂Vx∂x + ∂Vy∂y)

#     ∂Vx∂x = 0e0
#     ∂Vy∂y = 0e0
#     ∂Vx∂x = Base.@nexprs 2 i-> ∂Vx∂x += ∂N∂x_v[i,1] * vx[i]
#     ∂Vy∂y = Base.@nexprs 2 i-> ∂Vy∂y += ∂N∂x_v[i,2] * vy[i]
#     ∇V    = (∂Vx∂x + ∂Vy∂y)
# end


phase_loc = [1, 2, 1]
ηb = rand(), rand()
η  = rand(), rand()
α  = rand(), rand()
Δt = rand()

integrate_PH_pressure_residual((vxloc, vyloc), Ploc,  P0loc, Tloc, T0loc, geo_v_el, geo_P_el, phase_loc, α, ηb, ξ, Δt, NP)

# velocity residuals
∂N∂x, dΩ = geo_v_el[q] # velocity
Nv      = Nvq[q]
        
# Rx = ∂τxx∂x + ∂τxy∂y - ∂P∂x + ρgxy
# εxx = ∂vx∂x
# εyy = ∂vy∂y
# εxy = (∂vx∂y + ∂vy∂x) / 2

# Compute strain rates
∇vx = ∂N∂x' * vxloc # [∂vx∂x, ∂vx∂y]
∇vy = ∂N∂x' * vyloc # [∂vy∂x, ∂vy∂y]
εxx = ∇vx[1]
εyy = ∇vy[2]
εxy = (∇vx[2] + ∇vy[1]) / 2
# Compute deviatoric stress
tr  = (εxx + εyy) / 3
ηq = FEMTools.interp2ip_phase(Nv, η, phase_loc)

τxx = 2 * η * (εxx + tr)
τyy = 2 * η * (εyy + tr)
τxy = 2 * η * εxy
# Compute deviatoric stress derivatives
∂τxx∂x = ∂N∂x[:, 1] * τxx
∂τyy∂y = ∂N∂x[:, 2] * τyy
∂τxy∂x = ∂N∂x[:, 2] * τxy
∂τxy∂y = ∂N∂x[:, 1] * τxy

# Project pressure gradient to integration point
∂NP∂x = ∂N∂x[1:3,:] # this is WRONG, TO BE FIXED
∇P = ∂NP∂x' * Ploc # [∂vx∂x, ∂vx∂y]
Rx = ∂τxx∂x + ∂τxy∂y - ∇P[1] #+ ρgx
Ry = ∂τyy∂y + ∂τxy∂x - ∇P[2] #+ ρgy

