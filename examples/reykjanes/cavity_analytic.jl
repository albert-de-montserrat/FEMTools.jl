"""
    elliptical_cavity_area_compliance(a, b, G, ν) -> Float64

Area change per unit pressure of an elliptical hole with semi-axes `a` and `b` in an
infinite elastic plane under plane strain, `ΔA/p = π ((a + b)² + κ (a − b)²) / (4G)` with
`κ = 3 − 4ν`. It reduces to `π a² / G` for a circle and to the Griffith crack value
`π (1 − ν) a² / G` when `b → 0`.
"""
function elliptical_cavity_area_compliance(a, b, G, ν)
    κ = 3 - 4ν
    return π * ((a + b)^2 + κ * (a - b)^2) / (4G)
end

"""
    elliptical_cavity_displacement(x, y; a, b, p, G, ν) -> (ux, uy)

Plane-strain displacement at `(x, y)` outside an elliptical hole with semi-axes `a` (along
`x`) and `b` (along `y`) centred on the origin, loaded by a uniform internal pressure `p`
with zero stress at infinity (Muskhelishvili's complex-potential solution).

The exterior of the ellipse maps onto `|ζ| ≥ 1` through `z = R (ζ + m / ζ)` with
`R = (a + b) / 2` and `m = (a − b) / (a + b)`. The potentials are
`φ = −p R m / ζ` and `ψ = −p R (1 / ζ + m (1 + m ζ²) / (ζ (ζ² − m)))`, and the displacement is
`2G (ux + i uy) = κ φ − z conj(φ′) / conj(ω′) − conj(ψ)` with `κ = 3 − 4ν`. Points on the
ellipse itself are valid.
"""
function elliptical_cavity_displacement(x, y; a, b, p, G, ν)
    κ = 3 - 4ν
    R = (a + b) / 2
    m = (a - b) / (a + b)
    z = complex(x, y)
    root = sqrt(z^2 - 4m * R^2)
    ζ₁, ζ₂ = (z + root) / 2R, (z - root) / 2R
    ζ = abs(ζ₁) >= abs(ζ₂) ? ζ₁ : ζ₂
    dω = R * (1 - m / ζ^2)
    φ = -p * R * m / ζ
    dφ = p * R * m / ζ^2
    ψ = -p * R * (1 / ζ + m * (1 + m * ζ^2) / (ζ * (ζ^2 - m)))
    w = (κ * φ - z * conj(dφ) / conj(dω) - conj(ψ)) / 2G
    return real(w), imag(w)
end
