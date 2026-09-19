# Level 0 placeholder law of the plan (Eqs. l0 and l0state):
#   threshold_n = dP0 (1 - a D_n + b R_n),
#   D_{n+1} = (D_n + 1) exp(-dt_n / t_h),   R_{n+1} = (R_n + 1) exp(-dt_n / t_M),   D_1 = R_1 = 0.
# The interval dt_n depends on the threshold it leads to, so it is solved by fixed-point iteration.
# Time is in units of dP0 / rate. Illustrative only: the parameters are not fitted to data.
function level0(; a, b, th, tm, n = 12, dP0 = 1.0, rate = 1.0)
    D = 0.0; R = 0.0
    thr = dP0
    dPc = [thr]; tev = [thr / rate]
    for _ in 1:n-1
        Δ = thr / rate
        Dn = D; Rn = R; thr_next = thr
        for _ in 1:500
            Dn = (D + 1) * exp(-Δ / th)
            Rn = (R + 1) * exp(-Δ / tm)
            thr_next = dP0 * (1 - a * Dn + b * Rn)
            Δnew = thr_next / rate
            abs(Δnew - Δ) < 1e-12 && break
            Δ = Δnew
        end
        D, R, thr = Dn, Rn, thr_next
        push!(dPc, thr); push!(tev, tev[end] + Δ)
    end
    (; dPc, tev, M = dPc ./ dPc[1] .- 1)
end

# The four scenarios used in the sketches
scenario_h0    = level0(a = 0.0,  b = 0.0,  th = 6.0, tm = 6.0)
scenario_weak  = level0(a = 0.07, b = 0.0,  th = 6.0, tm = 6.0)
scenario_clamp = level0(a = 0.0,  b = 0.05, th = 6.0, tm = 6.0)
scenario_rev   = level0(a = 0.20, b = 0.05, th = 1.5, tm = 12.0)
