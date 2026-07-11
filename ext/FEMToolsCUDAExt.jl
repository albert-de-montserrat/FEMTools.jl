module FEMToolsCUDAExt

using CUDA: CuArray, CUDABackend
using FEMTools

FEMTools.TA(::CUDABackend) = CuArray

end # module FEMToolsCUDAExt
