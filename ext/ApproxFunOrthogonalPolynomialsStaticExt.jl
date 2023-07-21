module ApproxFunOrthogonalPolynomialsStaticExt

using ApproxFunOrthogonalPolynomials
import ApproxFunOrthogonalPolynomials as AFOP
using Static

AFOP._onehalf(::Union{StaticInt, StaticFloat64}) = static(0.5)
AFOP._minonehalf(@nospecialize(_::Union{StaticInt, StaticFloat64})) = static(-0.5)
AFOP.isapproxminhalf(@nospecialize(a::StaticInt)) = AFOP.isequalminhalf(a)
AFOP.isequalminhalf(@nospecialize(a::StaticInt)) = false
AFOP.isequalhalf(@nospecialize(a::StaticInt)) = false

AFOP.isapproxhalfoddinteger(@nospecialize(a::StaticInt)) = false
function AFOP.isapproxhalfoddinteger(a::StaticFloat64)
    x = mod(a, static(1))
    x == AFOP._onehalf(x) || dynamic(x) ≈ 0.5
end

AFOP._maybetoint(x::Union{Integer, StaticInt}) = Int(x)

function AFOP.BandedMatrix(S::AFOP.SubOperator{T,AFOP.ConcreteConversion{Chebyshev{DD,RR},
                Ultraspherical{LT,DD,RR},T},
                NTuple{2,UnitRange{Int}}}) where {T,LT<:StaticInt,DD,RR}
    AFOP._BandedMatrix(S)
end


AFOP.hasconversion(C::Chebyshev, U::Ultraspherical{<:StaticInt}) = AFOP.domainscompatible(C,U)
AFOP.hasconversion(U::Ultraspherical{<:StaticInt}, C::Chebyshev) = false

@inline function AFOP._Multiplication(f::Fun{<:Chebyshev}, sp::Ultraspherical{<:StaticInt})
    if order(sp) == 1
        cfs = AFOP.coefficients(f)
        AFOP.MultiplicationWrapper(f,
            AFOP.SpaceOperator(
                AFOP.SymToeplitzOperator(cfs/2) +
                    AFOP.HankelOperator(view(cfs,3:length(cfs))/(-2)),
                sp, sp)
        )
    else
        AFOP.ConcreteMultiplication(f,sp)
    end
end
@static if VERSION >= v"1.8"
    Base.@constprop aggressive AFOP.Multiplication(f::Fun{<:Chebyshev}, sp::Ultraspherical{<:StaticInt}) =
        AFOP._Multiplication(f, sp)
else
    AFOP.Multiplication(f::Fun{<:Chebyshev}, sp::Ultraspherical{<:StaticInt}) = AFOP._Multiplication(f, sp)
end

function Base.getindex(M::AFOP.ConcreteConversion{<:Chebyshev,U,T},
        k::Integer,j::Integer) where {T, U<:Ultraspherical{<:StaticInt}}
   # order must be 1
    if k==j==1
        one(T)
    elseif k==j
        one(T)/2
    elseif j==k+2
        -one(T)/2
    else
        zero(T)
    end
end


function Base.getindex(M::AFOP.ConcreteConversion{U1,U2,T},
        k::Integer,j::Integer) where {DD,RR,
            U1<:Ultraspherical{<:Union{Integer, StaticInt},DD,RR},
            U2<:Ultraspherical{<:Union{Integer, StaticInt},DD,RR},T}
    #  we can assume that λ==m+1
    λ=order(rangespace(M))
    c=λ-one(T)  # this supports big types
    if k==j
        c/(k - 2 + λ)
    elseif j==k+2
        -c/(k + λ)
    else
        zero(T)
    end
end

AFOP.bandwidths(C::AFOP.ConcreteConversion{<:Chebyshev,<:Ultraspherical{<:StaticInt}}) = 0,2  # order == 1
AFOP.bandwidths(C::AFOP.ConcreteConversion{<:Ultraspherical{<:Union{Integer,StaticInt}},<:Ultraspherical{<:Union{Integer,StaticInt}}}) = 0,2


Base.stride(C::AFOP.ConcreteConversion{<:Chebyshev,<:Ultraspherical{<:StaticInt}}) = 2

function AFOP.conversion_rule(a::Chebyshev, b::Ultraspherical{<:StaticInt})
    if AFOP.domainscompatible(a,b)
        a
    else
        AFOP.NoSpace()
    end
end

AFOP.conversion_rule(a::Ultraspherical{<:StaticInt}, b::Ultraspherical{<:StaticInt}) =
    AFOP._conversion_rule(a, b)


end
