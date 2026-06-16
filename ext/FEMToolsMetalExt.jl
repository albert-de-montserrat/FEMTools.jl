module FEMToolsMetalExt

using Metal
using FEMTools
using Metal: MetalBackend

FEMTools.TA(::MetalBackend) = MtlArray

end # module FEMToolsMetalExt
