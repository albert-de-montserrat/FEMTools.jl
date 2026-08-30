# Discrete adjoint of the 3-D sinking block: one transpose solve delivers the
# objective's sensitivity to every material parameter at once. The forward
# problem and the adjoint share the Q2/P1-disc Hex27 discretization and run
# matrix-free.
#
# The solve runs on any KernelAbstractions backend. For CUDA, set `isGPU = true`.
const isGPU = false

@static if isGPU
    using CUDA
    const backend = CUDABackend()
else
    using KernelAbstractions: CPU
    const backend = CPU()
end

include("sinking_block_3D_setup.jl")

result = solve_sinking_block_adjoint_3d(
    run_sinking_block_3d(; backend, write_output = false),
)
@info "3D sinking-block adjoint" result.objective result.density_gradient result.viscosity_gradient
