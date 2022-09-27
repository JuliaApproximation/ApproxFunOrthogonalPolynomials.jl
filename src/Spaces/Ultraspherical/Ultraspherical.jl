
export Ultraspherical, NormalizedUltraspherical

#Ultraspherical Spaces


"""
`Ultraspherical(λ)` is the space spanned by the ultraspherical polynomials
```
    C_0^{(λ)}(x),C_1^{(λ)}(x),C_2^{(λ)}(x),…
```
Note that `λ=1` this reduces to Chebyshev polynomials of the second kind:
`C_k^{(1)}(x) = U_k(x)`.
For `λ=1/2` this also reduces to Legendre polynomials:
`C_k^{(1/2)}(x) = P_k(x)`.
"""
struct Ultraspherical{T,D<:Domain,R} <: PolynomialSpace{D,R}
    order::T
    domain::D
    Ultraspherical{T,D,R}(m::T,d::D) where {T,D,R} = (@assert m ≠ 0; new(m,d))
    Ultraspherical{T,D,R}(m::Number,d::Domain) where {T,D,R} = (@assert m ≠ 0; new(strictconvert(T,m),strictconvert(D,d)))
    Ultraspherical{T,D,R}(d::Domain) where {T,D,R} = new(one(T),strictconvert(D,d))
    Ultraspherical{T,D,R}(m::Number) where {T,D,R} = (@assert m ≠ 0; new(strictconvert(T,m),D()))
end

Ultraspherical(m::Number,d::Domain) = Ultraspherical{typeof(m),typeof(d),real(prectype(d))}(m,d)
Ultraspherical(m::Number,d) = Ultraspherical(m,Domain(d))
Ultraspherical(m::Number) = Ultraspherical(m,ChebyshevInterval())
NormalizedUltraspherical(m) = NormalizedPolynomialSpace(Ultraspherical(m))
NormalizedUltraspherical(m,d) = NormalizedPolynomialSpace(Ultraspherical(m,d))


order(S::Ultraspherical) = S.order
setdomain(S::Ultraspherical,d::Domain) = Ultraspherical(order(S),d)


convert(::Type{Ultraspherical{T,D,R}}, S::Ultraspherical{T,D,R}) where {T,D,R} = S


canonicalspace(S::Ultraspherical) = Chebyshev(domain(S))
pointscompatible(A::Ultraspherical, B::Chebyshev) = domain(A) == domain(B)
pointscompatible(A::Chebyshev, B::Ultraspherical) = domain(A) == domain(B)

struct UltrasphericalPlan{CT,FT,IP}
    chebplan::CT
    cheb2legplan::FT

    UltrasphericalPlan{CT,FT}(cp,c2lp,::Val{IP}) where {CT,FT,IP} = new{CT,FT,IP}(cp,c2lp)
end

struct UltrasphericalIPlan{CT,FT,IP}
    chebiplan::CT
    leg2chebplan::FT

    UltrasphericalIPlan{CT,FT}(cp,c2lp,::Val{IP}) where {CT,FT,IP} = new{CT,FT,IP}(cp,c2lp)
end

function UltrasphericalPlan(λ::Number,vals,inplace = Val(false))
    if λ == 0.5
        cp = ApproxFunBase._plan_transform!!(inplace)(Chebyshev(),vals)
        c2lp = plan_cheb2leg(eltype(vals),length(vals))
        UltrasphericalPlan{typeof(cp),typeof(c2lp)}(cp,c2lp,inplace)
    else
        error("Not implemented")
    end
end

function UltrasphericalIPlan(λ::Number,cfs,inplace = Val(false))
    if λ == 0.5
        cp = ApproxFunBase._plan_itransform!!(inplace)(Chebyshev(),cfs)
        c2lp=plan_leg2cheb(eltype(cfs),length(cfs))
        UltrasphericalIPlan{typeof(cp),typeof(c2lp)}(cp,c2lp,inplace)
    else
        error("Not implemented")
    end
end

*(UP::UltrasphericalPlan{<:Any,<:Any,false},v::AbstractVector) =
    UP.cheb2legplan*(UP.chebplan*v)
*(UP::UltrasphericalIPlan{<:Any,<:Any,false},v::AbstractVector) =
    UP.chebiplan*(UP.leg2chebplan*v)

*(UP::UltrasphericalPlan{<:Any,<:Any,true},v::AbstractVector) =
    lmul!(UP.cheb2legplan, UP.chebplan*v)
*(UP::UltrasphericalIPlan{<:Any,<:Any,true},v::AbstractVector) =
    UP.chebiplan * lmul!(UP.leg2chebplan, v)

plan_transform(sp::Ultraspherical{Int},vals::AbstractVector) = CanonicalTransformPlan(sp,vals)
plan_transform!(sp::Ultraspherical{Int},vals::AbstractVector) = CanonicalTransformPlan(sp,vals,Val(true))
plan_transform(sp::Ultraspherical,vals::AbstractVector) = UltrasphericalPlan(order(sp),vals)
plan_transform!(sp::Ultraspherical,vals::AbstractVector) = UltrasphericalPlan(order(sp),vals,Val(true))
plan_itransform(sp::Ultraspherical{Int},cfs::AbstractVector) = ICanonicalTransformPlan(sp,cfs)
plan_itransform!(sp::Ultraspherical{Int},cfs::AbstractVector) = ICanonicalTransformPlan(sp,cfs,Val(true))
plan_itransform(sp::Ultraspherical,cfs::AbstractVector) = UltrasphericalIPlan(order(sp),cfs)
plan_itransform!(sp::Ultraspherical,cfs::AbstractVector) = UltrasphericalIPlan(order(sp),cfs,Val(true))

## Construction

#domain(S) may be any domain

ones(::Type{T},S::Ultraspherical) where {T<:Number} = Fun(S,fill(one(T),1))
ones(S::Ultraspherical) = Fun(S,fill(1.0,1))



## Fast evaluation

function Base.first(f::Fun{Ultraspherical{Int,D,R}}) where {D,R}
    n = length(f.coefficients)
    n == 0 && return zero(cfstype(f))
    n == 1 && return first(f.coefficients)
    foldr(-,coefficients(f,Chebyshev))
end

Base.last(f::Fun{Ultraspherical{Int,D,R}}) where {D,R} = reduce(+,coefficients(f,Chebyshev))

Base.first(f::Fun{Ultraspherical{O,D,R}}) where {O,D,R} = f(leftendpoint(domain(f)))
Base.last(f::Fun{Ultraspherical{O,D,R}}) where {O,D,R} = f(rightendpoint(domain(f)))

Fun(::typeof(identity), d::Ultraspherical) = Fun(Fun(identity, domain(d)),d)


## Calculus




spacescompatible(a::Ultraspherical,b::Ultraspherical) =
    order(a) == order(b) && domainscompatible(a,b)
hasfasttransform(::Ultraspherical) = true



include("UltrasphericalOperators.jl")
include("DirichletSpace.jl")
include("ContinuousSpace.jl")
