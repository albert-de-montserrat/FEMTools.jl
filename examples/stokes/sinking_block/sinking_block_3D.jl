# Viscous sinking block in three dimensions: a dense, stiff cube embedded in a
# lighter, weaker matrix descends under gravity. The discretization is Q2
# velocity / P1-discontinuous pressure on Hex27 cells, solved matrix-free.
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

result = run_sinking_block_3d(; backend)
@info "3D sinking-block forward solve" result.mesh.nnodes result.mesh.nels result.solve_stats.err
