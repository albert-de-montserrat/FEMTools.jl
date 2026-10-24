```math
\begin{aligned}
\frac{\partial R_x^{i,j}}{\partial V_x^c} =
&\ \frac{\partial R_x^{i,j}}{\partial \tau_{xx}^{i+1,j}}
   \frac{\partial \tau_{xx}^{i+1,j}}{\partial \varepsilon_{xx}^{i+1,j}}
   \frac{\partial \varepsilon_{xx}^{i+1,j}}{\partial V_x^c} \\
&+ \frac{\partial R_x^{i,j}}{\partial \tau_{xx}^{i,j}}
   \frac{\partial \tau_{xx}^{i,j}}{\partial \varepsilon_{xx}^{i,j}}
   \frac{\partial \varepsilon_{xx}^{i,j}}{\partial V_x^c} \\
&+ \frac{\partial R_x^{i,j}}{\partial \tau_{xy}^{i+1,j+1}}
   \frac{\partial \tau_{xy}^{i+1,j+1}}{\partial \varepsilon_{xy}^{i+1,j+1}}
   \frac{\partial \varepsilon_{xy}^{i+1,j+1}}{\partial V_x^c} \\
&+ \frac{\partial R_x^{i,j}}{\partial \tau_{xy}^{i+1,j}}
   \frac{\partial \tau_{xy}^{i+1,j}}{\partial \varepsilon_{xy}^{i+1,j}}
   \frac{\partial \varepsilon_{xy}^{i+1,j}}{\partial V_x^c} \\
&- \frac{\partial R_x^{i,j}}{\partial P_{\mathrm{num}}^{i+1,j}}
   \gamma_{\mathrm{eff}}^{i+1,j}
   \frac{\partial \nabla V^{i+1,j}}{\partial V_x^c} \\
&- \frac{\partial R_x^{i,j}}{\partial P_{\mathrm{num}}^{i,j}}
   \gamma_{\mathrm{eff}}^{i,j}
   \frac{\partial \nabla V^{i,j}}{\partial V_x^c} .
\end{aligned}
```

```math
\begin{aligned}
\frac{\partial R_x^{i,j}}{\partial V_x^\ell} =
&\ \frac{\partial R_x^{i,j}}{\partial \tau_{xx}^{i,j}}
   \frac{\partial \tau_{xx}^{i,j}}{\partial \varepsilon_{xx}^{i,j}}
   \frac{\partial \varepsilon_{xx}^{i,j}}{\partial V_x^\ell} \\
&- \frac{\partial R_x^{i,j}}{\partial P_{\mathrm{num}}^{i,j}}
   \gamma_{\mathrm{eff}}^{i,j}
   \frac{\partial \nabla V^{i,j}}{\partial V_x^\ell},
\\[1em]
\frac{\partial R_x^{i,j}}{\partial V_x^r} =
&\ \frac{\partial R_x^{i,j}}{\partial \tau_{xx}^{i+1,j}}
   \frac{\partial \tau_{xx}^{i+1,j}}{\partial \varepsilon_{xx}^{i+1,j}}
   \frac{\partial \varepsilon_{xx}^{i+1,j}}{\partial V_x^r} \\
&- \frac{\partial R_x^{i,j}}{\partial P_{\mathrm{num}}^{i+1,j}}
   \gamma_{\mathrm{eff}}^{i+1,j}
   \frac{\partial \nabla V^{i+1,j}}{\partial V_x^r},
\\[1em]
\frac{\partial R_x^{i,j}}{\partial V_x^n} =
&\ \frac{\partial R_x^{i,j}}{\partial \tau_{xy}^{i+1,j+1}}
   \frac{\partial \tau_{xy}^{i+1,j+1}}{\partial \varepsilon_{xy}^{i+1,j+1}}
   \frac{\partial \varepsilon_{xy}^{i+1,j+1}}{\partial V_x^n},
\\[1em]
\frac{\partial R_x^{i,j}}{\partial V_x^s} =
&\ \frac{\partial R_x^{i,j}}{\partial \tau_{xy}^{i+1,j}}
   \frac{\partial \tau_{xy}^{i+1,j}}{\partial \varepsilon_{xy}^{i+1,j}}
   \frac{\partial \varepsilon_{xy}^{i+1,j}}{\partial V_x^s}.
\end{aligned}
```
