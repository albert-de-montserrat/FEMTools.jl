"""
    VelocityBoundaryCondition(x, y, z)

Container bundling the boundary specifications for the three velocity
components `x`, `y`, and `z`.
"""
struct VelocityBoundaryCondition{Tx, Ty, Tz} <: AbstractBoundaryCondition
    x::Tx 
    y::Ty
    z::Tz 
    function VelocityBoundaryCondition(x::Tx, y::Ty, z::Tz) where {Tx, Ty, Tz}
        new{Tx, Ty, Tz}(x, y, z)
    end
end