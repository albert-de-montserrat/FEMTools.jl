module FEMToolsAMDGPUExt

using AMDGPU
using FEMTools
using KernelAbstractions: ROCBackend

FEMTools.TA(::ROCBackend) = ROCArray

end # module FEMToolsAMDGPUExt
