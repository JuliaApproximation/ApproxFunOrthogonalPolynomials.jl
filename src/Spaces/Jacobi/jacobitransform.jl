
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


struct JacobiITransformPlan{T,CPLAN,CJT} <: AbstractTransformPlan{T}
    ichebplan::CPLAN
    icjtplan::CJT
    inplace :: Bool
end

JacobiITransformPlan(chebplan::CPLAN, cjtplan::CJT, inplace = false) where {CPLAN,CJT} =
    JacobiITransformPlan{eltype(chebplan),CPLAN,CJT}(chebplan, cjtplan, inplace)



plan_itransform(S::Jacobi, v::AbstractVector) =
    JacobiITransformPlan(plan_itransform(Chebyshev(), v), plan_jac2cheb(v, S.a, S.b))
plan_itransform!(S::Jacobi, v::AbstractVector) =
    JacobiITransformPlan(plan_itransform!(Chebyshev(), v), plan_jac2cheb(v, S.a, S.b), true)
function *(P::JacobiITransformPlan, cfs::AbstractVector)
    c2 = P.inplace ? lmul!(P.icjtplan, cfs) : P.icjtplan * cfs
    P.ichebplan * c2
end


function coefficients(f::AbstractVector{T}, a::Jacobi, b::Chebyshev) where T
    if domain(a) == domain(b) && (!isapproxinteger(a.a-0.5) || !isapproxinteger(a.b-0.5))
        jac2cheb(f, convert(T,a.a), convert(T,a.b))
    else
        defaultcoefficients(f,a,b)
    end
end
function coefficients(f::AbstractVector{T}, a::Chebyshev, b::Jacobi) where T
    isempty(f) && return f
    if domain(a) == domain(b) && (!isapproxinteger(b.a-0.5) || !isapproxinteger(b.b-0.5))
        cheb2jac(f, convert(T,b.a), convert(T,b.b))
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
