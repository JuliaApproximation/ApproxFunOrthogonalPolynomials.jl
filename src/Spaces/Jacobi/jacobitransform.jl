
points(S::Jacobi, n) = points(Chebyshev(domain(S)), n)

struct JacobiTransformPlan{T,CPLAN,CJT} <: AbstractTransformPlan{T}
    chebplan::CPLAN
    cjtplan::CJT
end

JacobiTransformPlan(chebplan::CPLAN, cjtplan::CJT) where {CPLAN,CJT} =
    JacobiTransformPlan{eltype(chebplan),CPLAN,CJT}(chebplan, cjtplan)

plan_transform(S::Jacobi, v::AbstractVector) =
    JacobiTransformPlan(plan_transform(Chebyshev(), v), plan_cheb2jac(v, S.a, S.b))
plan_transform!(S::Jacobi, v::AbstractVector) =
    JacobiTransformPlan(plan_transform!(Chebyshev(), v), plan_cheb2jac(v, S.a, S.b))
*(P::JacobiTransformPlan, vals::AbstractVector) = lmul!(P.cjtplan, P.chebplan * vals)


struct JacobiITransformPlan{T,CPLAN,CJT,inplace} <: AbstractTransformPlan{T}
    ichebplan::CPLAN
    icjtplan::CJT
end

JacobiITransformPlan(chebplan, cjtplan) = JacobiITransformPlan{false}(chebplan, cjtplan)
JacobiITransformPlan{inplace}(chebplan::CPLAN, cjtplan::CJT) where {CPLAN,CJT,inplace} =
    JacobiITransformPlan{eltype(chebplan),CPLAN,CJT,inplace}(chebplan, cjtplan)

inplace(J::JacobiITransformPlan{<:Any,<:Any,<:Any,IP}) where {IP} = IP

plan_itransform(S::Jacobi, v::AbstractVector) =
    JacobiITransformPlan(plan_itransform(Chebyshev(), v), plan_jac2cheb(v, S.a, S.b))
plan_itransform!(S::Jacobi, v::AbstractVector) =
    JacobiITransformPlan{true}(plan_itransform!(Chebyshev(), v), plan_jac2cheb(v, S.a, S.b))
icjt(P, cfs, ::Val{true}) = lmul!(P, cfs)
icjt(P, cfs, ::Val{false}) = P * cfs
function *(P::JacobiITransformPlan, cfs::AbstractVector)
    P.ichebplan * icjt(P.icjtplan, cfs, Val(inplace(P)))
end

function _changepolybasis(v::StridedVector{T}, C::Chebyshev{<:ChebyshevInterval}, J::Jacobi{<:ChebyshevInterval}) where {T<:AbstractFloat}
    if J.a == 0 && J.b == 0
        cheb2leg(v)
    else
        cheb2jac(v, strictconvert(T,J.a), strictconvert(T,J.b))
    end
end
function _changepolybasis(v::StridedVector{T}, J::Jacobi{<:ChebyshevInterval}, C::Chebyshev{<:ChebyshevInterval}) where {T<:AbstractFloat}
    if J.a == 0 && J.b == 0
        leg2cheb(v)
    else
        jac2cheb(v, strictconvert(T,J.a), strictconvert(T,J.b))
    end
end
function _changepolybasis(v::StridedVector{T}, U::Ultraspherical{<:Any,<:ChebyshevInterval}, J::Jacobi{<:ChebyshevInterval}) where {T<:AbstractFloat}
    ultra2jac(v, strictconvert(T,order(U)), strictconvert(T,J.a), strictconvert(T,J.b))
end
function _changepolybasis(v::StridedVector{T}, J::Jacobi{<:ChebyshevInterval}, U::Ultraspherical{<:Any,<:ChebyshevInterval}) where {T<:AbstractFloat}
    jac2ultra(v, strictconvert(T,J.a), strictconvert(T,J.b), strictconvert(T,order(U)))
end
function _changepolybasis(v::StridedVector{T}, J1::Jacobi{<:ChebyshevInterval}, J2::Jacobi{<:ChebyshevInterval}) where {T<:AbstractFloat}
    jac2jac(v, strictconvert(T,J1.a), strictconvert(T,J1.b), strictconvert(T,J2.a), strictconvert(T,J2.b))
end
_changepolybasis(v, a, b) = defaultcoefficients(v, a, b)

function coefficients(f::AbstractVector{T},
        a::Union{Chebyshev,Ultraspherical,Jacobi}, b::Union{Chebyshev,Ultraspherical,Jacobi}) where T
    _changepolybasis(f, a, b)
end
