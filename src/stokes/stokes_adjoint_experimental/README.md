# Coupled Stokes adjoint experiment

This directory contains an alternative to the production split adjoint solver.
It advances `λvx`, `λvy`, and `λP` in one dynamic-relaxation loop.

The frozen forward state defines the augmented residual

```text
Raug(u) = [ Rvx(vx, vy, P, γP RP / MP)
            Rvy(vx, vy, P, γP RP / MP)
            RP(vx, vy, P) ] .
```

The adjoint operator is `A = (∂Raug/∂u)'`. For each field, the maximum
eigenvalue heuristic is the maximum mass/diagonal-preconditioned absolute row
sum of `A`. The implementation obtains those rows as absolute column sums of
the forward element Jacobian and includes all nine block couplings. In
particular, the pressure estimate includes the transposed momentum-pressure
operators `Gx'` and `Gy'`, even when the pressure self-block vanishes in the
incompressible limit.

The minimum eigenvalue heuristic is updated from consecutive residuals using
the same secant/Rayleigh estimate as `src/stokes/solvers/DR.jl`, independently
for the three fields.

This remains an experiment: a saddle-point operator may be nonsymmetric and
have complex eigenvalues, whereas the scalar Chebyshev recurrence assumes a
positive real spectral interval. The solver therefore reports all three
spectral estimates and returns `converged = false` instead of replacing the
production split method when the simultaneous recurrence does not converge.
