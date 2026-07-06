struct VelocityBoundaryCondition{Tx, Ty, Tz} <: AbstractBoundaryCondition
    x::Tx 
    y::Ty
    z::Tz 
    function VelocityBoundaryCondition(x::Tx, y::Ty, z::Tz) where {Tx, Ty, Tz}
        new{Tx, Ty, Tz}(x, y, z)
    end
end