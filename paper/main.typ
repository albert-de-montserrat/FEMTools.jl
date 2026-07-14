#set document(title: "Automatically Tuned Pseudo-Transient Finite-Element Solvers for Heat Diffusion and Stokes Flow", author: ("Albert de Montserrat",))
#set page(paper: "a4", margin: (x: 24mm, y: 22mm), numbering: "1")
#set text(font: "Libertinus Serif", size: 10.5pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")

#align(center)[
  #text(size: 18pt, weight: "bold")[Automatically Tuned Pseudo-Transient Finite-Element Solvers for Heat Diffusion and Stokes Flow]
  #v(0.7em)
  #text(size: 11pt)[Albert de Montserrat]
  #v(0.3em)
  #text(size: 9pt, style: "italic")[Draft manuscript generated from the FEMTools.jl implementation]
]

#v(1em)

#block(inset: (x: 12mm, y: 3mm))[
  *Abstract.* We present a finite-element framework for transient heat diffusion and incompressible or weakly compressible Stokes flow, solved without assembling a global sparse matrix for direct factorisation. Element residuals are evaluated by quadrature and scattered through backend-portable kernels. The resulting nonlinear algebraic systems are advanced to steady state in a fictitious pseudo-time using diagonally preconditioned dynamic relaxation with Chebyshev-type damping. Solver parameters are retuned during the iteration. Automatic differentiation supplies element Jacobian blocks, from which absolute row sums and diagonal entries are assembled. Their ratio provides a conservative estimate of the largest eigenvalue of the preconditioned operator, while a Rayleigh-like residual-difference quotient estimates the smallest eigenvalue. The Stokes saddle-point problem is handled by an outer Powell--Hestenes/Arrow--Hurwicz pressure iteration wrapped around an inner velocity relaxation. Because the element residual is the single source of truth for the discretisation, the same automatically tuned iteration also solves the discrete adjoint problem: reverse-mode automatic differentiation supplies the exact transpose of the discrete forward Jacobian, so the resulting parameter gradients are consistent with the discrete objective to machine precision. Contracting the converged adjoint state against differentiated element residuals yields per-element density and viscosity sensitivities. The formulation is designed for shared CPU/GPU implementation through KernelAbstractions.jl.
]

#v(0.5em)
*Keywords:* finite elements; pseudo-transient continuation; dynamic relaxation; automatic differentiation; discrete adjoint; sensitivity analysis; Stokes flow; heat diffusion; matrix-free methods

= Introduction

Diffusion and creeping-flow problems produce elliptic algebraic systems whose solution often dominates a simulation. Direct sparse factorisation is robust but memory intensive, while general Krylov methods require an effective preconditioner. Pseudo-transient (PT) methods instead augment the steady equations with fictitious-time derivatives and march toward the steady solution. Their main practical weakness is parameter selection: the stable pseudo-time step and optimal damping depend on the spectrum of the preconditioned discrete operator. The finite-element notation follows the standard Galerkin construction described by @hughes1987fem.

This work describes the formulation implemented in *FEMTools.jl*. The same computational pattern is used for transient heat diffusion and mixed velocity--pressure Stokes flow:

+ finite-element residuals are integrated element by element;
+ automatic differentiation (AD) produces local Jacobian blocks;
+ diagonal and absolute-row-sum diagnostics are assembled without storing a global sparse Jacobian;
+ spectral estimates tune a Chebyshev-accelerated dynamic-relaxation iteration;
+ element kernels are expressed through KernelAbstractions.jl for execution on CPUs and accelerators; and
+ reverse-mode AD transposes the same element residuals to solve the discrete adjoint problem and assemble parameter sensitivities.

The aim of this draft is to document the mathematical correspondence between the governing equations, their finite-element residuals, the implemented automatically tuned solver, and its discrete adjoint. The adjoint solver and material sensitivities are a first-class part of the implementation; only quantitative validation experiments and performance measurements are left as explicit future work.

= Governing equations

== Heat diffusion

Let $Omega subset RR^d$ be a spatial domain, $T(x,t)$ temperature, $k$ thermal conductivity, $rho$ density, $C_p$ specific heat, and $s$ a volumetric source. The transient heat equation is

#math.equation($
  rho C_p partial_t T - nabla dot (k nabla T) = s quad "in" Omega.
$)

The boundary is split into Dirichlet and Neumann portions, $partial Omega = Gamma_D union Gamma_N$, with

#math.equation($
  T = T_D quad "on" Gamma_D,
  quad -k nabla T dot n = q_N quad "on" Gamma_N.
$)

For multiphase thermo-mechanical calculations, material properties are interpolated from phase data at quadrature points. The implemented linearised equation of state is

#math.equation($
  rho = rho_0 [1 - alpha (T-T_"ref") + P/K],
$)

where $alpha$ is thermal expansivity, $P$ pressure, and $K$ bulk modulus. Setting $alpha=0$ and $K=infinity$ recovers constant density.

== Stokes flow

At negligible inertia, velocity $v$, pressure $P$, deviatoric stress $tau$, density $rho$, and gravity $g$ satisfy

#math.equation($
  nabla dot tau - nabla P + rho g = 0 quad "in" Omega,
$)

#math.equation($
  nabla dot v + 1/eta_b partial_t P - alpha partial_t T = 0 quad "in" Omega.
$)

Here $eta_b$ is a pressure-storage or bulk-viscosity parameter. The incompressible limit is obtained as $eta_b -> infinity$ with steady temperature. The density is evaluated using the same equation of state as in the thermal problem.

For a Newtonian viscous material,

#math.equation($
  tau = 2 eta epsilon^"dev"(v),
  quad epsilon(v) = 1/2 (nabla v + nabla v^T).
$)

The implementation also supports Maxwell viscoelasticity. With time step $Delta t$, shear modulus $G$, old stress $tau^n$, and effective viscosity

#math.equation($
  eta_"ve" = [eta^(-1) + (G Delta t)^(-1)]^(-1),
$)

the stress update at a quadrature point is written schematically as

#math.equation($
  tau^(n+1) = 2 eta_"ve" [epsilon^"dev"(v^(n+1)) + tau^n/(2 G Delta t)].
$)

Optional Drucker--Prager return mapping can modify this trial stress, but the solver construction below is unchanged.

= Finite-element discretisation

== Approximation spaces and geometry

Let a mesh partition $Omega$ into elements $Omega_e$. Scalar temperature, velocity, and pressure fields are approximated by

#math.equation($
  T_h = sum_i N_i T_i,
  quad v_h = sum_i N_i^v v_i,
  quad P_h = sum_a N_a^p P_a.
$)

The Stokes implementation uses a mixed mesh, allowing different velocity and pressure spaces; examples include quadratic triangular velocity elements with discontinuous linear pressure. Reference-element gradients are mapped to physical coordinates by

#math.equation($
  nabla_x N = J^(-T) nabla_xi N,
  quad d Omega = abs(det J) d hat(Omega).
$)

All element integrals are evaluated using quadrature points $q$ with physical weights $w_q abs(det J_q)$.

== Heat residual

Backward Euler discretisation of the heat equation gives

#math.equation($
  rho C_p (T^(n+1)-T^n)/Delta t - nabla dot(k nabla T^(n+1)) - s = 0.
$)

The conventional Galerkin weak residual for test function $N_i$ is

#math.equation($
  R_i^T = integral_Omega N_i rho C_p (T_h-T_h^n)/Delta t d Omega
  + integral_Omega nabla N_i dot k nabla T_h d Omega
  - integral_Omega N_i s d Omega
  - integral_(Gamma_N) N_i bar(q) d Gamma.
$)

The current implementation stores an algebraically scaled residual equivalent to a pseudo-time update of the backward-Euler equation. At element level it evaluates

#math.equation($
  R_(i,e)^T = sum_q [(-T_i+T_i^n + Delta t s_i/(rho_q C_(p,q))) N_i
  - Delta t/(rho_q C_(p,q)) nabla N_i dot k_q nabla T_h]
  w_q abs(det J_q).
$)

Element vectors are scattered into the global residual with atomic additions. Dirichlet rows are constrained after assembly and before the pseudo-time update.

== Momentum and continuity residuals

Testing momentum balance with velocity basis function $N_i^v$ and integrating the stress divergence by parts gives, omitting prescribed traction terms,

#math.equation($
  R_i^v = integral_Omega [nabla N_i^v : tau - (nabla N_i^v) P
  - N_i^v rho g] d Omega.
$)

In two dimensions the implemented component residuals are

#math.equation($
  R_i^x = integral_Omega [N_(i,x) (tau_(x x)-P-P_"num")
  + N_(i,y) tau_(x y) - N_i rho g_x] d Omega,
$)

#math.equation($
  R_i^y = integral_Omega [N_(i,y) (tau_(y y)-P-P_"num")
  + N_(i,x) tau_(x y) - N_i rho g_y] d Omega.
$)

The pressure test space produces

#math.equation($
  R_a^P = integral_Omega N_a^p [-nabla dot v_h
  - (P_h-P_h^n)/(eta_b Delta t)
  + alpha (T_h-T_h^n)/Delta t] d Omega.
$)

Because $R^P$ is a weak, integrated residual, a lumped pressure mass

#math.equation($
  M_a^P = integral_Omega N_a^p d Omega
$)

converts $R_a^P/M_a^P$ to the pointwise scaling expected by the pressure iteration.

= Automatically tuned pseudo-transient iteration

== Dynamic relaxation

Consider an algebraic residual $R(u)=0$ and a positive diagonal preconditioner $D$. The implementation advances a rate $r$ and solution $u$ in fictitious time according to

#math.equation($
  r^(k+1) = D^(-1) R(u^k) + beta_k r^k,
$)

#math.equation($
  u^(k+1) = u^k + alpha_k r^(k+1).
$)

For the Stokes momentum equation the sign of $alpha_k$ is chosen consistently with the residual convention. The parameters are obtained from estimates of the smallest and largest eigenvalues of the preconditioned Jacobian. Define

#math.equation($
  Delta tau = 2 "CFL" / sqrt(lambda_"max"),
  quad c = 2 c_f sqrt(lambda_"min").
$)

The Chebyshev/dynamic-relaxation coefficients are

#math.equation($
  alpha = (2 Delta tau^2)/(2+c Delta tau),
  quad beta = (2-c Delta tau)/(2+c Delta tau).
$)

When $lambda_"min"=0$, the first update is undamped. As the iteration progresses, the estimated low-frequency stiffness increases the damping and suppresses oscillatory error modes.

== Residual-difference estimate of the smallest eigenvalue

At tuning intervals, the code estimates the smallest eigenvalue from the change in residual caused by the latest pseudo-time increment. Let $delta u = alpha r$ and $delta R=R^k-R^(k-1)$. The implemented quotient is

#math.equation($
  lambda_"min" approx abs((delta u^T D^(-1) delta R)/(delta u^T delta u)).
$)

This is a Rayleigh-like estimate along the actual update direction. It costs only vector operations and reuses successive residuals.

== AD-based estimate of the largest eigenvalue

The largest-eigenvalue procedure has two distinct stages. First, forward-mode AD differentiates the element residual with respect to its element degrees of freedom:

#math.equation($
  J_e = partial R_e / partial u_e.
$)

For heat diffusion, this is the full scalar element Jacobian. For Stokes momentum, separate same-component and cross-component blocks are differentiated so that velocity coupling is retained. In the augmented Stokes operator, the differentiated residual also contains the chain

#math.equation($
  v -> R^P(v) -> P_"num"(v) -> R^v(v),
  quad P_"num" = gamma_P R^P/M^P.
$)

Thus AD captures the grad--div-like Powell--Hestenes coupling without a manually derived tangent.

Second, the code reduces each differentiated row to an absolute row sum and diagonal magnitude,

#math.equation($
  s_i = sum_j abs(J_(i j)),
  quad d_i = abs(J_(i i)).
$)

Element contributions are assembled into global vectors $s$ and $d$. The diagonal vector is used as the preconditioner $D$, and the implemented estimate is

#math.equation($
  lambda_"max"^"est" = max_i s_i/d_i.
$)

This quantity is best interpreted as a conservative Gershgorin-style bound or proxy for the spectral radius of the diagonally preconditioned operator, not as an AD derivative of an eigenvalue. AD supplies exact local Jacobian entries; the row-sum reduction supplies the spectral estimate. Invalid, non-finite, or non-positive estimates are rejected before an update is taken.

== Heat-solver algorithm

For every physical heat time step, the PT loop performs the following operations:

1. Assemble $R^T$; at the first iteration and every $n_"check"$ iterations, also use AD to assemble $s$ and $d$.
2. Enforce homogeneous residual and rate values on Dirichlet degrees of freedom.
3. Evaluate $lambda_"max"^"est"$, update the pseudo-time rate, and advance temperature.
4. Reapply prescribed temperature values.
5. At tuning intervals, compute the residual norm, estimate $lambda_"min"$, and refresh $alpha$ and $beta$.
6. Stop when the relative residual falls below the requested tolerance.

== Stokes pressure--velocity algorithm

The mixed saddle-point problem uses two nested iterations. An outer Powell--Hestenes/Arrow--Hurwicz loop updates pressure, while the inner loop relaxes velocity:

1. Assemble the continuity residual $R^P$ and physical momentum residual $R^v$.
2. Set the inner target to a prescribed reduction of the current outer residual.
3. During the inner loop, form $P_"num"=gamma_P R^P/M^P$, assemble the augmented momentum residual, and update both velocity components by dynamic relaxation.
4. At tuning intervals, use AD on the augmented element residual to refresh $lambda_"max,x"$, $lambda_"max,y"$, and the component-wise Chebyshev parameters.
5. After the inner target is reached, update pressure by

#math.equation($
  P^(m+1) = P^m + gamma_P R^P/M^P.
$)

6. Continue until both momentum and continuity residuals satisfy the absolute or relative stopping criterion.

The pressure scale $gamma_P$ is assembled from viscosity information, while division by the lumped pressure mass removes mesh-volume scaling from the weak continuity residual.

= Discrete adjoint problem

The same element residuals that define the forward discretisation also define its discrete adjoint. Because AD already supplies exact local Jacobian blocks, the transpose operator needed for gradient computation is available without deriving a separate tangent by hand. This section develops the adjoint for the mixed velocity--pressure Stokes problem; the heat equation follows the same construction with a single field.

== Objective and Lagrangian

Collect the discrete state in $u=(v,P)$ and the converged forward equations in
$R(u;m)=0$, where $m$ denotes material parameters --- here per-phase or per-element density $rho$ and viscosity $eta$. A scalar objective $J(u)$ measures a quantity of interest. The reference example uses an observation-box velocity integral

#math.equation($
  J(v) = -integral_(Omega_"obs") v_y d Omega,
$)

whose consistently assembled finite-element load is

#math.equation($
  (partial J)/(partial V_(y,i)) = -integral_(Omega_"obs") N_i^v d Omega.
$)

Evaluating this as an element load vector, rather than assigning a constant to each selected node, keeps the objective discretisation consistent with the momentum residual: on elements cut by the observation boundary, quadrature approximates the fraction of the element inside $Omega_"obs"$.

Introduce a Lagrangian with multiplier $lambda$,

#math.equation($
  cal(L)(u,m,lambda) = J(u) + lambda^T R(u;m).
$)

Stationarity with respect to $u$ at the converged forward state gives the *discrete adjoint equation*

#math.equation($
  (partial R/partial u)^T lambda = -partial J/partial u,
$)

after which the total derivative of the objective with respect to the parameters reduces to a contraction that no longer contains $partial u\/partial m$:

#math.equation($
  (d J)/(d m) = lambda^T (partial R)/(partial m).
$)

The transpose Jacobian is that of the *discrete* forward operator, not a rediscretisation of a continuous adjoint PDE. With this choice $lambda^T partial R\/partial m$ is the exact gradient of the discrete objective, and it matches a finite-difference check of that objective to machine precision. A mismatched or coarser adjoint discretisation would make the gradient inconsistent and degrade any optimisation built on it.

== Transpose of the augmented saddle-point operator

The forward Jacobian $partial R\/partial u$ has the block structure of a Stokes saddle point coupling velocity and pressure, augmented by the Powell--Hestenes term. Its transpose is assembled block by block through reverse-mode AD @moses2020enzyme of the same element functions used in the forward solve. Each reverse pass seeds an adjoint field and returns the action of a transposed block, which is scattered with atomic additions exactly as the forward residual is.

The velocity block contributes $(partial R^v/partial v)^T lambda^v$ and the mixed block $(partial R^v/partial P)^T lambda^v$. The saddle-point coupling from the continuity residual returns $(partial R^P/partial v)^T lambda^P$ into the velocity adjoint and the pressure self-coupling $(partial R^P/partial P)^T lambda^P$ into the pressure adjoint. The latter is nonzero only for finite bulk modulus, where $R^P$ stores pressure elastically through the $-(P-P^n)\/(eta_b Delta t)$ term; omitting it makes the adjoint --- and any gradient built from it --- wrong by $cal(O)(1\/(eta_b Delta t))$, exact only in the incompressible limit.

The augmented coupling requires transposing the numerical-pressure chain. In the forward solve $P_"num"=gamma_P R^P\/M^P$ enters the momentum residual, so its reverse pass closes as

#math.equation($
  (partial R^P/partial u)^T [gamma_P/M^P (partial R^v/partial P_"num")^T lambda^v].
$)

The velocity part of this chain feeds the velocity adjoint and the pressure part feeds the pressure adjoint, reproducing the grad--div-like Powell--Hestenes coupling in transposed form.

Dirichlet conditions transpose as well. The forward momentum operator is boundary-conditioned as $B(R^v)$, with $B$ overwriting constrained residual entries by prescribed values. Differentiating this overwrite gives $B^T lambda$: constrained state seeds are projected to zero before the interior transpose is applied, and the derivative with respect to the prescribed boundary values is retained as a separate cotangent. This makes boundary data usable as a control: its gradient is a byproduct of the same pullback that enforces homogeneous adjoint conditions on the free degrees of freedom.

== Adjoint dynamic-relaxation solver

The discrete adjoint equation is itself a linear Stokes saddle-point system, and it is solved by the same automatically tuned pseudo-transient iteration used for the forward problem, now driven by the transpose operator. The forward state is frozen at convergence: the transpose Jacobian and its diagonal preconditioner are evaluated once at that state and held fixed, since the adjoint is linear.

The spectral estimates carry over with one change. The diagonal preconditioner $D$ is shared with the forward operator, but the conservative largest-eigenvalue bound now requires row sums of the transpose, equivalently column sums of the forward Jacobian:

#math.equation($
  lambda_"max"^"est" = max_i (sum_j abs(J_(j i)))/abs(J_(i i)).
$)

The smallest-eigenvalue estimate reuses the residual-difference quotient of the forward solver, evaluated along the adjoint update direction, and refreshes the Chebyshev pair $(alpha,beta)$ at tuning intervals while $Delta tau$ and $lambda_"max"$ stay fixed.

Because the adjoint retains the saddle-point structure, it also retains the nested iteration. The inner loop relaxes the velocity adjoint $(lambda^(v_x),lambda^(v_y))$ by dynamic relaxation against the transposed, preconditioned momentum residual; the outer loop updates the pressure adjoint by an Arrow--Hurwicz step

#math.equation($
  lambda^(P,m+1) = lambda^(P,m) + gamma_P R^(lambda P)/M^P,
$)

with $R^(lambda P)$ the assembled adjoint pressure residual. Inf--sup stability is inherited from the forward pair: the same quadratic-velocity bubble that keeps the forward Stokes problem LBB-stable keeps the adjoint free of checkerboard pressure modes. The adjoint is cheaper than the forward solve only through solver effort --- a single linear solve at the frozen state reusing the forward Jacobian and preconditioner --- never through a coarser discretisation.

The adjoint algorithm mirrors the forward Stokes algorithm:

1. Assemble the transpose velocity block once at the converged forward state and take its column-sum $lambda_"max"$ bound and diagonal preconditioner.
2. Seed the current velocity and pressure adjoints; apply the Dirichlet pullback; assemble the transposed momentum, continuity, and Powell--Hestenes couplings into the adjoint residuals, adding the objective load to the velocity part.
3. Relax the velocity adjoint by dynamic relaxation, enforcing homogeneous Dirichlet conditions on the adjoint velocity and its rate.
4. At tuning intervals, re-estimate $lambda_"min"$ and refresh the Chebyshev parameters.
5. Once the inner velocity target is met, update the pressure adjoint by the Arrow--Hurwicz step.
6. Continue until the adjoint velocity and pressure residuals satisfy the stopping criterion.

== Material sensitivities

With $lambda$ solving the discrete adjoint equation, the gradient of the objective with respect to a material parameter $m$ follows from $d J\/d m=lambda^T partial R\/partial m$. Restricting to one element $e$ and its momentum residual $R_e$ gives the per-element sensitivity

#math.equation($
  s^e(m) = lambda_e^T (partial R_e)/(partial m).
$)

Each entry is the contribution of a single element. It is formed by reverse-mode AD: the element momentum residual is evaluated with single-entry material tuples seeded by that element's own $eta$ and $rho$, contracted against the element adjoint, and differentiated with respect to those seeds. Summing the entries belonging to one phase gives the derivative with respect to that phase's single global material value,

#math.equation($
  (d J)/(d m_p) = sum_(e in "phase" p) s^e(m_p),
$)

while keeping the contributions element by element gives spatial density and viscosity sensitivity maps.

The raw $s^e$ are un-normalised element integrals, approximately area times a sensitivity density. Dividing by the element area recovers a mesh-independent sensitivity density suitable for plotting on irregular meshes, whereas the phase-wise gradients keep summing the raw integrals.

= Implementation architecture

Element functions gather local vectors into statically sized arrays, evaluate quadrature loops, and return element residuals. KernelAbstractions.jl kernels launch one work item per element; Atomix.jl additions safely scatter local contributions when neighbouring elements share nodes. The same element functions are passed to ForwardDiff.jl @revels2016forwarddiff to generate local tangents used for tuning. Consequently, the residual is the single source of truth for both the nonlinear equation and its Jacobian diagnostics. The implementation is written in Julia @bezanson2017julia and maintained as part of FEMTools.jl @femtools2026.

No global sparse Jacobian is required by the PT solver. Storage is dominated by solution, residual, rate, preconditioner, and geometry vectors. For Stokes flow, separate buffers are retained for the two velocity components, pressure, the pressure mass, and the augmented numerical pressure.

The forward and adjoint paths use complementary modes of automatic differentiation on the same element functions. Forward-mode differentiation @revels2016forwarddiff supplies the element Jacobian blocks whose row sums tune the iteration, while reverse-mode differentiation @moses2020enzyme supplies the transposed block actions of the adjoint solve and the parameter contractions of the sensitivity assembly. Each reverse pass is element-local and scatters its result with the same atomic additions as the forward residual, so the adjoint and sensitivity kernels inherit the backend portability of the forward kernels. The element residual therefore remains the single source of truth for four distinct outputs: the nonlinear equation, its Jacobian diagnostics, the transpose operator, and the material gradient.

= Discussion

The approach combines three useful properties. First, it avoids a global matrix factorisation and maps naturally to accelerator hardware. Second, AD keeps the tuning operator consistent with multiphase interpolation, viscoelasticity, and the augmented pressure coupling. Third, periodic retuning adapts the smoother as nonlinear rheology or coefficients evolve.

The current spectral estimate is deliberately inexpensive. Absolute row sums are conservative, but they may overestimate the true spectral radius and therefore reduce the pseudo-time step. The diagonal preconditioner is also insufficient for strongly anisotropic meshes or extreme material contrasts. More sophisticated block, element, or multilevel preconditioners could retain the same AD-generated local Jacobians.

The distinction between physical and pseudo-time must remain explicit. For heat diffusion, backward Euler advances physical time, while PT iterations solve the implicit algebraic problem at each physical step. For Stokes flow, the inner fictitious time has no physical meaning; only the converged velocity and pressure are retained.

= Verification and planned experiments

A complete validation study should include:

+ manufactured-solution convergence for heat diffusion on structured and unstructured meshes;
+ analytic Stokes benchmarks, including pure shear and buoyant inclusion problems;
+ comparison of the row-sum estimate with explicitly computed eigenvalues on small meshes;
+ iteration counts as functions of mesh spacing, viscosity contrast, element order, and $n_"check"$;
+ finite-difference verification of the adjoint gradient, confirming that $lambda^T partial R\/partial m$ matches the derivative of the discrete objective to machine precision;
+ adjoint iteration counts relative to the forward solve under the same mesh and contrast sweeps;
+ CPU/GPU timing and memory scaling; and
+ comparison against sparse direct and Krylov solvers using identical finite-element residuals.

These experiments are not fabricated in this draft. Their tables and figures should be generated from versioned scripts before submission.

= Conclusion

We described a backend-portable finite-element implementation of heat diffusion and Stokes flow solved by automatically tuned pseudo-transient dynamic relaxation. AD differentiates the same element residuals used in the physical discretisation. Assembled diagonal and row-sum diagnostics estimate the high end of the preconditioned spectrum, while a residual-difference quotient estimates its low end. These estimates determine the stable pseudo-time step and Chebyshev damping. For Stokes flow, the construction extends to a nested pressure--velocity iteration whose differentiated augmented residual includes the numerical pressure coupling. Reverse-mode differentiation of the same element residuals transposes the discrete forward Jacobian, so the identical tuned iteration solves the discrete adjoint and yields parameter gradients consistent with the discrete objective to machine precision; contracting the adjoint state against differentiated element residuals produces per-element density and viscosity sensitivities. The result is a compact matrix-free solver architecture that is suitable for heterogeneous multiphysics problems, gradient-based inversion, and accelerator execution.

= References

#bibliography("references.bib", style: "ieee", title: none)
