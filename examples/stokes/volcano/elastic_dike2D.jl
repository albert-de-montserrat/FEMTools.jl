# Plane-strain surface displacements of a uniform elastic half-space for the
# standard analytical volcano sources: a line center of dilatation (2D Mogi)
# and a uniform tensile dislocation of any dip (Okada 1985 with infinite
# strike). x is horizontal, depths are positive downward, uz is positive up.
# Both are linear in the source strength, so passing a rate gives a rate.

"""
    line_source_surface(x; area_change, depth, x0 = 0, ν = 0.25) -> (ux, uz)

Surface displacement of a line center of dilatation at `(x0, depth)` whose
cavity area changes by `area_change` [m²] (volume per unit length along
strike). The infinite-strike integral of the Mogi point source.
"""
function line_source_surface(x; area_change, depth, x0 = 0.0, ν = 0.25)
    depth > 0 || throw(ArgumentError("depth must be positive, got $depth"))
    r = x - x0
    scale = 2 * (1 - ν) * area_change / (π * (r^2 + depth^2))
    return scale * r, scale * depth
end

"""
    tensile_dislocation_surface(x; opening, center_depth, length, dip, x0 = 0) -> (ux, uz)

Surface displacement of a plane-strain tensile dislocation with uniform
`opening` [m], centered at `(x0, center_depth)`, `length` long along dip.
`dip` is from the horizontal in [0, 180): 90 is a vertical dike, 0 a sill,
and a smaller angle descends toward `+x`, as in `dike_injection2D_adj_coverage.jl`.
Independent of the elastic moduli in 2D.
"""
function tensile_dislocation_surface(x; opening, center_depth, length, dip, x0 = 0.0)
    0 <= dip < 180 || throw(ArgumentError("dip must lie in [0, 180), got $dip"))
    if dip > 90
        ux, uz = tensile_dislocation_surface(-(x - x0); opening, center_depth, length, dip = 180 - dip)
        return -ux, uz
    end
    sδ, cδ = sind(dip), cosd(dip)
    half = length / 2
    # Okada frame: origin above the bottom edge, y pointing up-dip (toward −x).
    bottom_depth = center_depth + half * sδ
    top_depth = center_depth - half * sδ
    top_depth > 0 || throw(ArgumentError("the dislocation must lie below the surface"))
    y = x0 + half * cδ - x
    q = y * sδ - bottom_depth * cδ
    # Infinite-strike limits of Okada's (1985) Chinnery terms at down-dip coordinate η.
    function terms(η)
        ỹ = η * cδ + q * sδ
        d̃ = η * sδ - q * cδ
        r² = η^2 + q^2
        # Both edges jump by the same π where q changes sign, so the
        # difference of the two edges stays continuous.
        θ = atan(η / q)
        return 2 * d̃ * q / r² + 2 * sδ * θ, -2 * ỹ * q / r² - 2 * cδ * θ
    end
    p = y * cδ + bottom_depth * sδ
    uy_bottom, uz_bottom = terms(p)
    uy_top, uz_top = terms(p - length)
    scale = opening / (2π)
    return -scale * (uy_bottom - uy_top), scale * (uz_bottom - uz_top)
end

"""
    dike_segment_greens(x; center_depth, length, dip, n_segments, x0 = 0) -> (Gx, Gz)

Surface displacement at stations `x` (rows) per unit opening of each of
`n_segments` equal segments (columns) of the dislocation, numbered from the
bottom (1) to the top, or from the +x end of a sill, as the FEM segments.
"""
function dike_segment_greens(x; center_depth, length, dip, n_segments, x0 = 0.0)
    rotation = deg2rad(90 - dip)
    along = (-sin(rotation), cos(rotation))
    Gx = zeros(Base.length(x), n_segments)
    Gz = zeros(Base.length(x), n_segments)
    for j in 1:n_segments
        s = ((j - 0.5) / n_segments - 0.5) * length
        xj, zj = x0 + s * along[1], center_depth - s * along[2]
        for (i, xi) in pairs(x)
            Gx[i, j], Gz[i, j] = tensile_dislocation_surface(xi;
                opening = 1.0, center_depth = zj, length = length / n_segments, dip, x0 = xj)
        end
    end
    return Gx, Gz
end

if abspath(PROGRAM_FILE) == @__FILE__
let
    # Vertical dike: closed form uz = (U/π)[x²/(x²+d₁²) − x²/(x²+d₂²)], ux odd and outward.
    for xi in (-7.0e3, 0.0, 1.0e3, 4.0e3)
        ux, uz = tensile_dislocation_surface(xi; opening = 1.0, center_depth = 5.0e3, length = 6.0e3, dip = 90)
        @assert isapprox(uz, (xi^2 / (xi^2 + 2.0e3^2) - xi^2 / (xi^2 + 8.0e3^2)) / π; atol = 1.0e-12)
        @assert sign(ux) == sign(xi)
    end
    # Wide shallow sill: uplift equals the opening above its center.
    _, uz = tensile_dislocation_surface(0.0; opening = 1.0, center_depth = 1.0e3, length = 1.0e6, dip = 0)
    @assert isapprox(uz, 1.0; rtol = 1.0e-3)
    # Continuous across vertical, mirror-symmetric about it.
    a = tensile_dislocation_surface(1.5e3; opening = 1.0, center_depth = 5.0e3, length = 6.0e3, dip = 90 - 1.0e-7)
    b = tensile_dislocation_surface(1.5e3; opening = 1.0, center_depth = 5.0e3, length = 6.0e3, dip = 90 + 1.0e-7)
    @assert all(isapprox.(a, b; rtol = 1.0e-6))
    c = tensile_dislocation_surface(-2.0e3; opening = 1.0, center_depth = 5.0e3, length = 6.0e3, dip = 120)
    d = tensile_dislocation_surface(2.0e3; opening = 1.0, center_depth = 5.0e3, length = 6.0e3, dip = 60)
    @assert c[1] ≈ -d[1] && c[2] ≈ d[2]
    # Segments superpose to the whole dislocation.
    Gx, Gz = dike_segment_greens([3.0e3]; center_depth = 5.0e3, length = 6.0e3, dip = 60, n_segments = 6)
    whole = tensile_dislocation_surface(3.0e3; opening = 1.0, center_depth = 5.0e3, length = 6.0e3, dip = 60)
    @assert sum(Gx) ≈ whole[1] && sum(Gz) ≈ whole[2]
    # Line source: the strike integral of the Mogi point source (1−ν)ΔV d / (π R³).
    xi, d, ν = 3.0e3, 4.0e3, 0.25
    ys = range(-2.0e6, 2.0e6; length = 2_000_001)
    mogi_uz = sum(y -> (1 - ν) * d / (π * (xi^2 + y^2 + d^2)^1.5), ys) * step(ys)
    @assert isapprox(line_source_surface(xi; area_change = 1.0, depth = d, ν)[2], mogi_uz; rtol = 1.0e-5)
    println("elastic_dike2D self-checks passed")
end
end
