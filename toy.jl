using FEMTools
using DomainSets
using SparseArrays

function generate_sparsity_pattern(mesh)
    rows = Int[]
    cols = Int[]

    for iel in axes(mesh.el2n, 2)
        local_nodes = @view mesh.el2n[:, iel]
        for j in local_nodes
            for i in local_nodes
                push!(rows, i)
                push!(cols, j)
            end
        end
    end

    pattern = sparse(rows, cols, trues(length(rows)), mesh.nnodes, mesh.nnodes)
    return pattern
end

function color_mesh_greedy(mesh)
    colors = zeros(Int, mesh.nels)

    for iel in 1:mesh.nels
        used_colors = Set{Int}()
        for node in @view mesh.el2n[:, iel]
            for jel in mesh.n2el[node]
                color = colors[jel]
                if color != 0
                    push!(used_colors, color)
                end
            end
        end

        color = 1
        while color in used_colors
            color += 1
        end
        colors[iel] = color
    end

    return colors
end

function color_mesh(mesh::Mesh{1})
    return [mod1(iel, 2) for iel in 1:mesh.nels]
end


Lx   = 10
Ω    = 0..Lx
nel  = 20
mesh = FEMTools.Mesh(Ω, nel)
K_pattern = generate_sparsity_pattern(mesh)
M_pattern = copy(K_pattern)
greedy_colors = color_mesh_greedy(mesh)
colors = color_mesh(mesh)

abstract type AbstractBoundaryCondition end

struct DirichletBoundaryCondition{T, D, V} <: AbstractBoundaryCondition
    Γ::T
    DoFs::D
    vals::V
    function DirichletBoundaryCondition(Γ::T, DoFs::D, vals::V) where {T, D, V}
        new{T, D, V}(Γ, DoFs, vals)
    end
end

Γ_dirichlet = boundary(Ω)
Γ_nodes = in.(mesh.coords, Γ_dirichlet)
Γ_DoFs = mesh.DoFs[Γ_nodes]
Γ_vals = zeros(length(Γ_DoFs))

ΓD = DirichletBoundaryCondition(Γ_dirichlet, Γ_DoFs, Γ_vals)
