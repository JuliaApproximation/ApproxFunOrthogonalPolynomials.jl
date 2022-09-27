
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


function coefficients(f::AbstractVector{T}, a::Jacobi, b::Chebyshev) where T
    if domain(a) == domain(b) && (!isapproxinteger(a.a-0.5) || !isapproxinteger(a.b-0.5))
        jac2cheb(f, strictconvert(T,a.a), strictconvert(T,a.b))
    else
        defaultcoefficients(f,a,b)
    end
end
function coefficients(f::AbstractVector{T}, a::Chebyshev, b::Jacobi) where T
    isempty(f) && return f
    if domain(a) == domain(b) && (!isapproxinteger(b.a-0.5) || !isapproxinteger(b.b-0.5))
        cheb2jac(f, strictconvert(T,b.a), strictconvert(T,b.b))
    else
        defaultcoefficients(f,a,b)
    end
end

function coefficients(f::AbstractVector,a::Jacobi,b::Jacobi)
    if domain(a) == domain(b) && (!isapproxinteger(a.a-b.a) || !isapproxinteger(a.b-b.b))
        jac2jac(f,a.a,a.b,b.a,b.b)
    else
        defaultcoefficients(f,a,b)
    end
end
