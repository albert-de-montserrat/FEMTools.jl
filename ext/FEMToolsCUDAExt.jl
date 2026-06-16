module FEMToolsCUDAExt

using CUDA
using FEMTools
using KernelAbstractions: CUDABackend

FEMTools.TA(::CUDABackend) = CuArray

end # module FEMToolsCUDAExt
