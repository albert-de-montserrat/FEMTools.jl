using LinearAlgebra

function p1_basis(ξ)
    N = zeros(2)
    dN_dξ = zeros(2)

    N[1] = 0.5 * (1 - ξ)
    N[2] = 0.5 * (1 + ξ)

    dN_dξ[1] = -0.5
    dN_dξ[2] =  0.5

    return N, dN_dξ
end

function build_connectivity_1d(nel)
    conn = Matrix{Int}(undef, nel, 2)

    for e in 1:nel
        conn[e, 1] = e
        conn[e, 2] = e + 1
    end

    return conn
end

function build_partial_data_1d(xnodes, conn; kfun = x -> 1.0)
    nel = size(conn, 1)

    ξq = [-1 / sqrt(3), 1 / sqrt(3)]
    wq = [1.0, 1.0]
    nq = length(ξq)

    D = zeros(nel, nq)
    dNdx = zeros(nel, nq, 2)

    for e in 1:nel
        nodes = conn[e, :]
        x1 = xnodes[nodes[1]]
        x2 = xnodes[nodes[2]]

        J = (x2 - x1) / 2

        for q in 1:nq
            ξ = ξq[q]
            N, dN_dξ = p1_basis(ξ)

            xq = N[1] * x1 + N[2] * x2
            kq = kfun(xq)

            D[e, q] = kq * wq[q] * J

            for a in 1:2
                dNdx[e, q, a] = dN_dξ[a] / J
            end
        end
    end

    return D, dNdx
end

function apply_diffusion_1d!(r, u, conn, D, dNdx)
    fill!(r, 0.0)

    nel = size(conn, 1)
    nq = size(D, 2)

    for e in 1:nel
        n1 = conn[e, 1]
        n2 = conn[e, 2]

        ue1 = u[n1]
        ue2 = u[n2]

        re1 = 0.0
        re2 = 0.0

        for q in 1:nq
            dudx =
                dNdx[e, q, 1] * ue1 +
                dNdx[e, q, 2] * ue2

            flux = D[e, q] * dudx

            re1 += dNdx[e, q, 1] * flux
            re2 += dNdx[e, q, 2] * flux
        end

        r[n1] += re1
        r[n2] += re2
    end

    return r
end

# ------------------------------------------------------------
# Example problem
# ------------------------------------------------------------

nel = 8
nnodes = nel + 1

xnodes = range(0.0, 1.0, length = nnodes) |> collect
conn = build_connectivity_1d(nel)

# Variable diffusivity
kfun(x) = 1.0 + x

D, dNdx = build_partial_data_1d(xnodes, conn; kfun)

# Some test field
u = sin.(π .* xnodes)

r = similar(u)
apply_diffusion_1d!(r, u, conn, D, dNdx)

println("xnodes = ")
println(xnodes)

println("\nu = ")
println(u)

println("\nr = K*u without assembling K = ")
println(r)