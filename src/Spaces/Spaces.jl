

include("IntervalSpace.jl")
include("PolynomialSpace.jl")

## Union

# union_rule dictates how to create a space that both spaces can be converted to
# in this case, it means
function union_rule(s1::PiecewiseSpace{<:TupleOrVector{PolynomialSpace}},
        s2::PiecewiseSpace{<:TupleOrVector{PolynomialSpace}})
    PiecewiseSpace(map(Space,merge(domain(s1),domain(s2)).domains))
end

function union_rule(s1::PiecewiseSpace{<:TupleOrVector{PolynomialSpace}}, s2::PolynomialSpace)
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

# Fast conversion between common bases using FastTransforms

# The chebyshev-ultraspherical transforms are currently very slow,
# see https://github.com/JuliaApproximation/FastTransforms.jl/issues/204
# We therefore go through hoops to only call these for non-integral Ultraspherical orders

function _changepolybasis(v::StridedVector{T},
        C::MaybeNormalized{<:Chebyshev{<:ChebyshevInterval}},
        U::MaybeNormalized{<:Ultraspherical{<:Any,<:ChebyshevInterval}},
        ) where {T<:AbstractFloat}

    normcheb = C isa NormalizedPolynomialSpace
    normultra = U isa NormalizedPolynomialSpace
    Uc = normultra ? _stripnorm(U) : U
    Cc = normcheb ? _stripnorm(C) : C
    if order(U) == 1
        v = normcheb ? ApproxFunBase.mul_coefficients(ConcreteConversion(C, Cc), v) : v
        vc = ultraconversion(v)
        normultra ? ApproxFunBase.mul_coefficients!(ConcreteConversion(Uc, U), vc) : vc
    elseif isinteger(order(U))
        coefficients(v, C, Ultraspherical(1,domain(U)), U)
    else
        cheb2ultra(v, strictconvert(T, order(U)); normcheb, normultra)
    end
end
function _changepolybasis(v::StridedVector{T},
        U::MaybeNormalized{<:Ultraspherical{<:Any,<:ChebyshevInterval}},
        C::MaybeNormalized{<:Chebyshev{<:ChebyshevInterval}}) where {T<:AbstractFloat}

    normultra = U isa NormalizedPolynomialSpace
    normcheb = C isa NormalizedPolynomialSpace
    Uc = normultra ? _stripnorm(U) : U
    Cc = normcheb ? _stripnorm(C) : C
    if order(U) == 1
        v = normultra ? ApproxFunBase.mul_coefficients(ConcreteConversion(U, Uc), v) : v
        vc = ultraiconversion(v)
        normcheb ? ApproxFunBase.mul_coefficients!(ConcreteConversion(Cc, C), vc) : vc
    elseif isinteger(order(U))
        coefficients(v, U, Ultraspherical(1,domain(U)), C)
    else
        ultra2cheb(v, strictconvert(T, order(U)); normultra, normcheb)
    end
end
function _changepolybasis(v::StridedVector{T},
        U1::MaybeNormalized{<:Ultraspherical{<:Any,<:ChebyshevInterval}},
        U2::MaybeNormalized{<:Ultraspherical{<:Any,<:ChebyshevInterval}},
        ) where {T<:AbstractFloat}

    if isinteger(order(U1) - order(U2))
        defaultcoefficients(v, U1, U2)
    else
        norm1 = U1 isa NormalizedPolynomialSpace
        norm2 = U2 isa NormalizedPolynomialSpace
        ultra2ultra(v, strictconvert(T, order(U1)), strictconvert(T, order(U2)); norm1, norm2)
    end
end

function _changepolybasis(v::StridedVector{T},
        C::MaybeNormalized{<:Chebyshev{<:ChebyshevInterval}},
        J::MaybeNormalized{<:Jacobi{<:ChebyshevInterval}},
        ) where {T<:AbstractFloat}

    normcheb = C isa NormalizedPolynomialSpace
    normjac = J isa NormalizedPolynomialSpace
    Jc = _stripnorm(J)
    if Jc.a == 0 && Jc.b == 0
        cheb2leg(v; normcheb, normleg = normjac)
    else
        cheb2jac(v, strictconvert(T,Jc.a), strictconvert(T,Jc.b); normcheb, normjac)
    end
end
function _changepolybasis(v::StridedVector{T},
        J::MaybeNormalized{<:Jacobi{<:ChebyshevInterval}},
        C::MaybeNormalized{<:Chebyshev{<:ChebyshevInterval}},
        ) where {T<:AbstractFloat}

    normcheb = C isa NormalizedPolynomialSpace
    normjac = J isa NormalizedPolynomialSpace
    Jc = _stripnorm(J)
    if Jc.a == 0 && Jc.b == 0
        leg2cheb(v; normcheb, normleg = normjac)
    else
        jac2cheb(v, strictconvert(T,Jc.a), strictconvert(T,Jc.b); normcheb, normjac)
    end
end
function _changepolybasis(v::StridedVector{T},
        U::MaybeNormalized{<:Ultraspherical{<:Any,<:ChebyshevInterval}},
        J::MaybeNormalized{<:Jacobi{<:ChebyshevInterval}},
        ) where {T<:AbstractFloat}

    normultra = U isa NormalizedPolynomialSpace
    normjac = J isa NormalizedPolynomialSpace
    Jc = _stripnorm(J)
    ultra2jac(v, strictconvert(T,order(U)), strictconvert(T,Jc.a), strictconvert(T,Jc.b);
        normultra, normjac)
end
function _changepolybasis(v::StridedVector{T},
        J::MaybeNormalized{<:Jacobi{<:ChebyshevInterval}},
        U::MaybeNormalized{<:Ultraspherical{<:Any,<:ChebyshevInterval}},
        ) where {T<:AbstractFloat}

    normjac = J isa NormalizedPolynomialSpace
    normultra = U isa NormalizedPolynomialSpace
    Jc = _stripnorm(J)
    jac2ultra(v, strictconvert(T,Jc.a), strictconvert(T,Jc.b), strictconvert(T,order(U));
        normultra, normjac)
end
function _changepolybasis(v::StridedVector{T},
        J1::MaybeNormalized{<:Jacobi{<:ChebyshevInterval}},
        J2::MaybeNormalized{<:Jacobi{<:ChebyshevInterval}},
        ) where {T<:AbstractFloat}

    norm1 = J1 isa NormalizedPolynomialSpace
    norm2 = J2 isa NormalizedPolynomialSpace
    J1c = _stripnorm(J1)
    J2c = _stripnorm(J2)
    jac2jac(v, strictconvert(T,J1c.a), strictconvert(T,J1c.b), strictconvert(T,J2c.a), strictconvert(T,J2c.b);
        norm1, norm2)
end
_changepolybasis(v, a, b) = defaultcoefficients(v, a, b)

function coefficients(f::AbstractVector{T},
        a::MaybeNormalized{<:Union{Chebyshev,Ultraspherical,Jacobi}},
        b::MaybeNormalized{<:Union{Chebyshev,Ultraspherical,Jacobi}}) where T
    spacescompatible(a,b) && return f
    _changepolybasis(f, a, b)
end
