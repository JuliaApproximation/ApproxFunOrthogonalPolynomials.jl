

include("IntervalSpace.jl")
include("PolynomialSpace.jl")

## Union

# union_rule dictates how to create a space that both spaces can be converted to
# in this case, it means
function union_rule(s1::PiecewiseSpace{S1},s2::PiecewiseSpace{S2}) where {S1<:Tuple{Vararg{PolynomialSpace}},
                    S2<:Tuple{Vararg{PolynomialSpace}}}
    PiecewiseSpace(map(Space,merge(domain(s1),domain(s2)).domains))
end

function union_rule(s1::PiecewiseSpace{S1},s2::PolynomialSpace) where S1<:Tuple{Vararg{PolynomialSpace}}
    PiecewiseSpace(map(Space,merge(domain(s1),domain(s2)).domains))
end




include("Chebyshev/Chebyshev.jl")
include("Ultraspherical/Ultraspherical.jl")
include("Jacobi/Jacobi.jl")
include("Hermite/Hermite.jl")
include("Laguerre/Laguerre.jl")
include("CurveSpace.jl")



## Heaviside


conversion_rule(sp::HeavisideSpace,sp2::PiecewiseSpace{<:NTuple{<:Any,<:PolynomialSpace}}) = sp


Conversion(a::HeavisideSpace,
        b::PiecewiseSpace{<:NTuple{<:Any,<:PolynomialSpace},<:Domain{<:Number},<:Real}) =
    ConcreteConversion(a,b)
bandwidths(::ConcreteConversion{<:HeavisideSpace,
    <:PiecewiseSpace{<:NTuple{<:Any,<:PolynomialSpace},<:Domain{<:Number},<:Real}}) = 0,0

function getindex(C::ConcreteConversion{<:HeavisideSpace,
            <:PiecewiseSpace{<:NTuple{<:Any,<:PolynomialSpace},<:Domain{<:Number},<:Real}},
        k::Integer,j::Integer)
    k ≤ dimension(domainspace(C)) && j==k ? one(eltype(C)) : zero(eltype(C))
end
