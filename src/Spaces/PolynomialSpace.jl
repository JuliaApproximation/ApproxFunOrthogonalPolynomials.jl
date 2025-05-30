export PolynomialSpace, NormalizedPolynomialSpace

## Orthogonal polynomials

abstract type PolynomialSpace{D,R} <: Space{D,R} end

@containsconstants PolynomialSpace

Multiplication(f::Fun{U},sp::PolynomialSpace) where {U<:PolynomialSpace} = ConcreteMultiplication(f,sp)
bandwidths(M::ConcreteMultiplication{U,V}) where {U<:PolynomialSpace,V<:PolynomialSpace} =
    (ncoefficients(M.f)-1,ncoefficients(M.f)-1)
rangespace(M::ConcreteMultiplication{U,V}) where {U<:PolynomialSpace,V<:PolynomialSpace} = domainspace(M)




## Evaluation

function evaluate(f::AbstractVector,S::PolynomialSpace,x)
    # We call clenshaw in each branch to obtain type-stability
    y = tocanonical(S,x)
    if x in domain(S)
        clenshaw(S,f,y)
    elseif isambiguous(domain(S))
        length(f) == 0 && return clenshaw(S, SVector{0,eltype(f)}(),y)
        for k = 2:length(f)
            iszero(f[k]) || throw(ArgumentError("Ambiguous domains only work with constants"))
        end
        # type-stable way to evaluate the first element of the series
        return clenshaw(S, SVector{1}(f[1]),y)
    else
        # type-stable way to obtain the zero element
        clenshaw(S, SVector{0,eltype(f)}(),y)
    end
end

# we need the ... for multi-dimensional
evaluate(f::AbstractVector,S::PolynomialSpace,x,y,z...) =
    evaluate(f,S,SVector(x,y,z...))

function evaluate(f::AbstractVector, S::PolynomialSpace, x::Fun)
    if issubset(Interval(minimum(x),maximum(x)),domain(S))
        clenshaw(S,f,tocanonical(S,x))
    else
        error("Implement splitatpoints for evaluate ")
    end
end

## Extrapolation
extrapolate(f::AbstractVector,S::PolynomialSpace,x) = clenshaw(S,f,tocanonical(S,x))

######
# Recurrence encodes the recurrence coefficients
# or equivalently multiplication by x
######
struct Recurrence{S,T} <: TridiagonalOperator{T}
    space::S
end

Recurrence(sp) = Recurrence{typeof(sp),rangetype(sp)}(sp)

convert(::Type{Operator{T}},J::Recurrence{S}) where {T,S} = Recurrence{S,T}(J.space)

domainspace(R::Recurrence) = R.space
rangespace(R::Recurrence) = R.space


#####
# recα/β/γ are given by
#       x p_{n-1} =γ_n p_{n-2} + α_n p_{n-1} +  p_n β_n
#####

function getindex(R::Recurrence{S,T},k::Integer,j::Integer) where {S,T}
    if j==k-1
        recβ(T,R.space,k-1)
    elseif j==k
        recα(T,R.space,k)
    elseif j==k+1
        recγ(T,R.space,k+1)
    else
        zero(T)
    end
end

######
# JacobiZ encodes [BasisFunctional(1);(J-z*I)[2:end,:]]
# where J is the Jacobi operator
######
struct JacobiZ{S<:Space,T} <: TridiagonalOperator{T}
    space::S
    z::T
end

JacobiZ(sp::PolynomialSpace,z) =
    (T = promote_type(prectype(sp),typeof(z)); JacobiZ{typeof(sp),T}(sp,strictconvert(T,z)))

convert(::Type{Operator{T}},J::JacobiZ{S}) where {T,S} = JacobiZ{S,T}(J.space,J.z)

domainspace(::JacobiZ) = ℓ⁰
rangespace(::JacobiZ) = ℓ⁰

#####
# recα/β/γ are given by
#       x p_{n-1} =γ_n p_{n-2} + α_n p_{n-1} +  p_n β_n
#####

function getindex(J::JacobiZ{S,T},k::Integer,j::Integer) where {S,T}
    if j==k-1
        recγ(T,J.space,k)
    elseif j==k
        k == 1 ? one(T) : recα(T,J.space,k)-J.z
    elseif j==k+1 && k > 1
        recβ(T,J.space,k)
    else
        zero(T)
    end
end




#####
# Multiplication can be built from recurrence coefficients
#  multiplication by J=M[x] is Recurrence operator
#  and assume p_0 = 1, then we have
#
#   M[p_0] = 1
#   M[p_1] = (J/β_1 - α_1/β_1)*M[p_0]
#   M[p_k] = (J/β_k - α_k/β_k)*M[p_{k-1}] - γ_k/β_k*M[p_{k-2}]
#####


getindex(M::ConcreteMultiplication{C,PS,T},k::Integer,j::Integer) where {PS<:PolynomialSpace,T,C<:PolynomialSpace} = M[k:k,j:j][1,1]

if view(brand(0,0,0,0), band(0)) isa BandedMatrices.BandedMatrixBand
    dataview(V) = BandedMatrices.dataview(V)
else
#=
dataview is broken on BandedMatrices v0.17.6 and older.
We copy the function over from BandedMatrices.jl, which is distributed under the MIT license
See https://github.com/JuliaLinearAlgebra/BandedMatrices.jl/blob/master/LICENSE
=#
    function dataview(V)
        A = parent(parent(V))
        b = first(parentindices(V)).band.i
        m,n = size(A)
        l,u = bandwidths(A)
        data = BandedMatrices.bandeddata(A)
        view(data, u - b + 1, max(b,0)+1:min(n,m+b))
    end
end

_view(::Any, A, b) = view(A, b)
function _view(::Val{true}, A::BandedMatrix, b::Band)
    l, u = bandwidths(A)
    -l <= b.i <= u || throw(ArgumentError("invalid band $b for bandwidths $((-l,u))"))
    dataview(view(A, b))
end

function _get_bands(B, C, bmk, f, valB)
    Cbmk = _view(Val(true), C, band(bmk*f))
    Bm = _view(Val(true), B, band(flipsign(bmk-1, f)))
    B0 = _view(Val(true), B, band(flipsign(bmk, f)))
    Bp = _view(valB, B, band(flipsign(bmk+1, f)))
    Cbmk, Bm, B0, Bp
end

# Fast implementation of C[:,:] = α*J*B+β*C where the bandediwth of B is
# specified by b, not by the parameters in B
function jac_gbmm!(α, J, B, β, C, b, valB)
    if β ≠ 1
        lmul!(β,C)
    end

    n = size(J,1)
    Cn, Cm = size(C)

    Jp = _view(Val(true), J, band(1))
    J0 = _view(Val(true), J, band(0))
    Jm = _view(Val(true), J, band(-1))

    kr = intersect(-1:b-1, b-Cm+1:b-1+Cn)

    # unwrap the loops to forward indexing to the data wherever applicable
    # this might also help with cache localization
    k = -1
    if k in kr
        Cbmk, Bm, B0, Bp = _get_bands(B, C, b-k, 1, valB)
        for i in 1:n-b+k
            Cbmk[i] += α * Bm[i+1] * Jp[i]
        end
    end

    k = 0
    if k in kr
        Cbmk, Bm, B0, Bp = _get_bands(B, C, b-k, 1, valB)
        for i in 1:n-b+k
            Cbmk[i] += α * (Bm[i+1] * Jp[i] + B0[i] * J0[i])
        end
    end

    for k in max(1, first(kr)):last(kr)
        Cbmk, Bm, B0, Bp = _get_bands(B, C, b-k, 1, valB)
        Cbmk[1] += α * (Bm[2] * Jp[1] + B0[1] * J0[1])
        for i in 2:n-b+k
            Cbmk[i] += α * (Bm[i+1] * Jp[i] + B0[i] * J0[i] + Bp[i-1] * Jm[i-1])
        end
    end

    kr = intersect(-1:b-1, 1-Cn+b:Cm-1+b)

    k = -1
    if k in kr
        Ckmb, Bp, B0, Bm = _get_bands(B, C, b-k, -1, valB)
        for (i, Ji) in enumerate(b-k:n-1)
            Ckmb[i] += α * Bp[i] * Jm[Ji]
        end
    end

    k = 0
    if k in kr
        Ckmb, Bp, B0, Bm = _get_bands(B, C, b-k, -1, valB)
        Ckmb[1] += α * Bp[1] * Jm[b-k]
        for (i, Ji) in enumerate(b-k+1:n-1)
            Ckmb[i] += α * B0[i] * J0[Ji]
            Ckmb[i+1] += α * Bp[i+1] * Jm[Ji]
        end
        Ckmb[n-(b-k)] += α * B0[n-(b-k)] * J0[n]
    end

    for k = max(1, first(kr)):last(kr)
        Ckmb, Bp, B0, Bm = _get_bands(B, C, b-k, -1, Val(true))
        Ckmb[1] += α * Bp[1] * Jm[b-k]
        for (i, Ji) in enumerate(b-k+1:n-1)
            Ckmb[i] += α * (Bm[i] * Jp[Ji] + B0[i] * J0[Ji])
            Ckmb[i+1] += α * Bp[i+1] * Jm[Ji]
        end
        Ckmb[n-(b-k)] += α * B0[n-(b-k)] * J0[n]
    end

    C0 = _view(Val(true), C, band(0))
    Bm = _view(Val(true), B, band(-1))
    Bp = _view(Val(true), B, band(1))
    B0 = _view(Val(true), B, band(0))
    for i in 1:n-1
        C0[i] += α * (B0[i] * J0[i] + Bm[i] * Jp[i])
        C0[i+1] += α * Bp[i] * Jm[i]
    end
    C0[n] += α * B0[n] * J0[n]

    return C
end

function BandedMatrix(S::SubOperator{T,ConcreteMultiplication{C,PS,T},
                                     NTuple{2,UnitRange{Int}}}) where {PS<:PolynomialSpace,T,C<:PolynomialSpace}
    M=parent(S)
    kr,jr=parentindices(S)
    f=M.f
    a=f.coefficients
    sp=space(f)
    n=length(a)

    if n==0
        return BandedMatrix(Zeros, S)
    elseif n==1
        ret = BandedMatrix(Zeros, S)
        shft=kr[1]-jr[1]
        ret[band(shft)] .= a[1]
        return ret
    elseif n==2
        # we have U_x = [1 α-x; 0 β]
        # for e_1^⊤ U_x\a == a[1]*I-(α-J)*a[2]/β == (a[1]-α*a[2]/β)*I + J*a[2]/β
        # implying
        α,β=recα(T,sp,1),recβ(T,sp,1)
        ret=Operator{T}(Recurrence(M.space))[kr,jr]
        lmul!(a[2]/β,ret)
        shft=kr[1]-jr[1]
        @views ret[band(shft)] .+= a[1]-α*a[2]/β
        return ret
    end

    jkr=max(1,min(jr[1],kr[1])-(n-1)÷2):max(jr[end],kr[end])+(n-1)÷2

    #Multiplication is transpose
    J=Operator{T}(Recurrence(M.space))[jkr,jkr]

    B=n-1  # final bandwidth

    # Clenshaw for operators
    Bk2 = BandedMatrix(Zeros{T}(size(J)), (B,B))
    dataview(view(Bk2, band(0))) .= a[n]/recβ(T,sp,n-1)
    α,β = recα(T,sp,n-1),recβ(T,sp,n-2)
    Bk1 = lmul!(-α/β, copy(Bk2))
    dataview(view(Bk1, band(0))) .+= a[n-1]/β
    jac_gbmm!(one(T)/β,J,Bk2,one(T),Bk1,0, Val(true))
    b=1  # we keep track of bandwidths manually to reuse memory
    for k=n-2:-1:2
        # b goes from 1:
        α,β,γ=recα(T,sp,k),recβ(T,sp,k-1),recγ(T,sp,k+1)
        lmul!(-γ/β,Bk2)
        dataview(view(Bk2, band(0))) .+= a[k]/β
        jac_gbmm!(1/β,J,Bk1,one(T),Bk2,b,Val(true))
        LinearAlgebra.axpy!(-α/β,Bk1,Bk2)
        Bk2,Bk1=Bk1,Bk2
        b+=1
    end
    α,γ=recα(T,sp,1),recγ(T,sp,2)
    lmul!(-γ,Bk2)
    dataview(view(Bk2, band(0))) .+= a[1]
    jac_gbmm!(one(T),J,Bk1,one(T),Bk2,b,Val(false))
    LinearAlgebra.axpy!(-α,Bk1,Bk2)

    # relationship between jkr and kr, jr
    kr2,jr2=kr.-jkr[1].+1,jr.-jkr[1].+1

    # TODO: reuse memory of Bk2, though profile suggests it's not too important
    BandedMatrix(view(Bk2,kr2,jr2))
end

## General clenshaw
clenshaw(sp::PolynomialSpace,c::AbstractVector,x::AbstractArray) = clenshaw(c,x,
                                            ClenshawPlan(promote_type(eltype(c),eltype(x)),sp,length(c),length(x)))
clenshaw(sp::PolynomialSpace,c::AbstractMatrix,x::AbstractArray) = clenshaw(c,x,ClenshawPlan(promote_type(eltype(c),eltype(x)),sp,size(c,1),length(x)))
clenshaw(sp::PolynomialSpace,c::AbstractMatrix,x) = clenshaw(c,x,ClenshawPlan(promote_type(eltype(c),typeof(x)),sp,size(c,1),size(c,2)))

clenshaw!(sp::PolynomialSpace,c::AbstractVector,x::AbstractVector)=clenshaw!(c,x,ClenshawPlan(promote_type(eltype(c),eltype(x)),sp,length(x)))

function clenshaw(sp::PolynomialSpace,c::AbstractVector,x)
    N,T = length(c),promote_type(prectype(sp),eltype(c),typeof(x))
    TT = eltype(T)
    if isempty(c)
        return zero(T)
    end

    bk1,bk2 = zero(T),zero(T)
    A,B,C = recA(TT,sp,N-1),recB(TT,sp,N-1),recC(TT,sp,N)
    for k = N:-1:2
        bk2, bk1 = bk1, muladd(muladd(A,x,B),bk1,muladd(-C,bk2,c[k])) # muladd(-C,bk2,muladd(muladd(A,x,B),bk1,c[k])) # (A*x+B)*bk1+c[k]-C*bk2
        A,B,C = recA(TT,sp,k-2),recB(TT,sp,k-2),recC(TT,sp,k-1)
    end
    muladd(muladd(A,x,B),bk1,muladd(-C,bk2,c[1])) # muladd(-C,bk2,muladd(muladd(A,x,B),bk1,c[1])) # (A*x+B)*bk1+c[1]-C*bk2
end



# evaluate polynomial
# indexing starts from 0
function forwardrecurrence(::Type{T},S::Space,r::AbstractRange,x::Number) where T
    if isempty(r)
        return T[]
    end
    n=maximum(r)+1
    v=Vector{T}(undef, n)  # x may be complex
    if n > 0
        v[1]=1
        if n > 1
            v[2] = muladd(recA(T,S,0),x,recB(T,S,0))
            @inbounds for k=2:n-1
                v[k+1]=muladd(muladd(recA(T,S,k-1),x,recB(T,S,k-1)),v[k],-recC(T,S,k-1)*v[k-1])
            end
        end
    end

    return v[r.+1]
end

getindex(op::ConcreteEvaluation{<:PolynomialSpace}, k::Integer) = op[k:k][1]

_evalpt(op, x::Number) = x
_evalpt(op, x::SpecialEvalPtType) = boundaryfn(x)(domain(op))

function getindex(op::ConcreteEvaluation{<:PolynomialSpace},kr::AbstractRange)
    _getindex_evaluation(eltype(op), op.space, op.order, _evalpt(op, evaluation_point(op)), kr)
end

function _getindex_evaluation(::Type{T}, sp, order, x, kr::AbstractRange) where {T}
    Base.require_one_based_indexing(kr)
    isempty(kr) && return zeros(T, 0)
    if order == 0
        forwardrecurrence(T,sp,kr .- 1,tocanonical(sp,x))
    else
        z = Zeros{T}(length(range(minimum(kr), order, step=step(kr))))
        kr_red = kr .- (order + 1)
        polydegrees = reverse(range(maximum(kr_red), max(0, minimum(kr_red)), step=-step(kr)))
        D = Derivative(sp, order)
        if !isempty(polydegrees)
            P = forwardrecurrence(T, rangespace(D), polydegrees, tocanonical(sp,x))
            # in case the derivative has only one non-zero band, the matrix-vector
            # multiplication simplifies considerably.
            # This branch is particularly useful for ConcreteDerivatives where
            # indexing is fast, as the non-zero band may be computed without
            # evaluating the matrix
            if bandwidth(D, 1) == -bandwidth(D, 2)
                d = nonzeroband(T, D, polydegrees, order)
                Pd = P .* d
                if !isempty(z)
                    Pd = T[z; Pd]
                end
            else
                # in general, this is a banded matrix-vector product
                Dtv = view(transpose(D), kr, 1:length(P))
                Pd = strictconvert(Vector{T},  mul_coefficients(Dtv, P))
            end
        else
            Pd = T[z;]
        end
        Pd
    end
end

function nonzeroband(::Type{T}, D::ConcreteDerivative, polydegrees, order) where {T}
    bw = bandwidth(D, 2)
    T[D[k+1, k+1+bw] for k in polydegrees]
end
function nonzeroband(::Type{T}, D, polydegrees, order) where {T}
    bw = bandwidth(D, 2)
    rows = 1:(maximum(polydegrees) + order + 1)
    B = D[rows, rows .+ bw]
    Bv = @view B[diagind(B)]
    Bv[polydegrees .+ 1]
end


struct NormalizedPolynomialSpace{S,D,R} <: Space{D,R}
    space::S
    NormalizedPolynomialSpace{S,D,R}(space) where {S,D,R} = new{S,D,R}(space)
end

@containsconstants NormalizedPolynomialSpace

@inline function getproperty(N::NormalizedPolynomialSpace, v::Symbol)
    ((v == :a) || (v == :b)) && return getproperty(getfield(N, :space), v)
    getfield(N, v)
end

domain(S::NormalizedPolynomialSpace) = domain(S.space)
canonicalspace(S::NormalizedPolynomialSpace) = S.space
setdomain(NS::NormalizedPolynomialSpace, d::Domain) = NormalizedPolynomialSpace(setdomain(canonicalspace(NS), d))

NormalizedPolynomialSpace(space::PolynomialSpace{D,R}) where {D,R} = NormalizedPolynomialSpace{typeof(space),D,R}(space)

normalizedspace(S::PolynomialSpace) = NormalizedPolynomialSpace(S)
normalizedspace(S::NormalizedPolynomialSpace) = S

supportsinplacetransform(N::NormalizedPolynomialSpace) = supportsinplacetransform(N.space)

function Conversion(L::NormalizedPolynomialSpace, M::PolynomialSpace)
    if L.space == M
        ConcreteConversion(L, M)
    else
        sp = L.space
        ConversionWrapper(
            TimesOperator(Conversion(sp, M), ConcreteConversion(L, sp)),
            L, M,
        )
    end
end

function Conversion(L::PolynomialSpace, M::NormalizedPolynomialSpace)
    if M.space == L
        ConcreteConversion(L, M)
    else
        sp = M.space
        ConversionWrapper(
            TimesOperator(ConcreteConversion(sp, M), Conversion(L, sp)),
            L, M,
        )
    end
end

function Conversion(L::NormalizedPolynomialSpace{<:PolynomialSpace},
        M::NormalizedPolynomialSpace{<:PolynomialSpace})

    L == M && return Conversion(L)

    C1 = ConcreteConversion(L, canonicalspace(L))
    C2 = Conversion(canonicalspace(L), canonicalspace(M))
    C3 = ConcreteConversion(canonicalspace(M), M)
    ConversionWrapper(C3 * C2 * C1, L, M)
end

function Fun(::typeof(identity), S::NormalizedPolynomialSpace)
    C = canonicalspace(S)
    f = Fun(identity, C)
    coeffs = coefficients(f)
    CS = ConcreteConversion(C, S)
    ApproxFunBase.mul_coefficients!(CS, coeffs)
    Fun(S, coeffs)
end

function conversion_rule(a::NormalizedPolynomialSpace, b::PolynomialSpace)
    conversion_type(a.space, b)
end

function maxspace_rule(a::NormalizedPolynomialSpace, b::PolynomialSpace)
    maxspace(a.space, b)
end

function conversion_rule(a::NormalizedPolynomialSpace, b::NormalizedPolynomialSpace)
    S = conversion_type(a.space, b.space)
    S isa NoSpace ? S : NormalizedPolynomialSpace(S)
end

function maxspace_rule(a::NormalizedPolynomialSpace, b::NormalizedPolynomialSpace)
    S = maxspace(a.space, b.space)
    S isa NoSpace ? S : NormalizedPolynomialSpace(S)
end

# Conversion

bandwidths(::ConcreteConversion{<:NormalizedPolynomialSpace{S},S}) where {S} = (0, 0)
bandwidths(::ConcreteConversion{S,<:NormalizedPolynomialSpace{S}}) where {S} = (0, 0)

function getindex(C::ConcreteConversion{<:NormalizedPolynomialSpace{S,<:IntervalOrSegment},S,T},k::Integer,j::Integer) where {S,T}
    if j==k
        inv(sqrt(normalization(T, C.rangespace, k-1)*arclength(domain(C.rangespace))/2))
    else
        zero(T)
    end
end

function getindex(C::ConcreteConversion{S,<:NormalizedPolynomialSpace{S,<:IntervalOrSegment},T},k::Integer,j::Integer) where {S,T}
    if j==k
        sqrt(normalization(T, C.domainspace, k-1)*arclength(domain(C.domainspace))/2)
    else
        zero(T)
    end
end

function getindex(C::ConcreteConversion{<:NormalizedPolynomialSpace{S,<:Ray},S,T},k::Integer,j::Integer) where {S,T}
    if j==k
        inv(sqrt(normalization(T, C.rangespace, k-1)))
    else
        zero(T)
    end
end

function getindex(C::ConcreteConversion{S,<:NormalizedPolynomialSpace{S,<:Ray},T},k::Integer,j::Integer) where {S,T}
    if j==k
        sqrt(normalization(T, C.domainspace, k-1))
    else
        zero(T)
    end
end

const MaybeNormalized{S<:PolynomialSpace} = Union{S, NormalizedPolynomialSpace{S}}
const MaybeNormalizedTensorSpace{P1,P2} = TensorSpace{<:Tuple{MaybeNormalized{P1},MaybeNormalized{P2}}}

# this is only evaluated if FillArrays >= v1 is used
@static if isdefined(FillArrays, :OneElement)
    ## Special OneElement conversion
    function _mul_coefficients_concreteconv(C, v::OneElement)
        Base.require_one_based_indexing(v)
        nzind = v.ind[1]
        Cnzind = C[nzind, nzind]
        OneElement(Cnzind * v.val, v.ind, axes(v))
    end
    function mul_coefficients(C::ConcreteConversion{<:NormalizedPolynomialSpace{S}, S},
                                v::OneElement{<:Any,1}) where {S<:PolynomialSpace}
        _mul_coefficients_concreteconv(C, v)
    end
    function mul_coefficients(C::ConcreteConversion{S, <:NormalizedPolynomialSpace{S}},
                                v::OneElement{<:Any,1}) where {S<:PolynomialSpace}
        _mul_coefficients_concreteconv(C, v)
    end
    function _mul_coefficients_deriv(D::ConcreteDerivative, v::OneElement)
        Base.require_one_based_indexing(v)
        nzind = v.ind[1]
        order = D.order
        rowind = nzind-D.order
        Dnzind = D[rowind, nzind]
        OneElement(Dnzind * v.val, rowind, rowind)
    end
    function mul_coefficients(C::ConcreteDerivative{<:MaybeNormalized}, v::OneElement{<:Any,1})
        _mul_coefficients_deriv(C, v)
    end
end

# Evaluation
function getindex(op::ConcreteEvaluation{<:NormalizedPolynomialSpace}, k::Integer)
    S = domainspace(op)
    ec = Evaluation(canonicalspace(S), op.x, op.order)[k]
    nrm = ConcreteConversion(S, canonicalspace(S))[k,k]
    nrm * ec
end
function getindex(op::ConcreteEvaluation{<:NormalizedPolynomialSpace}, kr::AbstractRange)
    S = domainspace(op)
    ec = Evaluation(canonicalspace(S), op.x, op.order)[kr]
    C = ConcreteConversion(S, canonicalspace(S))
    nrm = [C[k,k] for k in kr]
    nrm .* ec
end

spacescompatible(a::NormalizedPolynomialSpace,b::NormalizedPolynomialSpace) = spacescompatible(a.space,b.space)
hasconversion(a::PolynomialSpace,b::NormalizedPolynomialSpace) = hasconversion(a,b.space)
hasconversion(a::NormalizedPolynomialSpace,b::PolynomialSpace) = hasconversion(a.space,b)
hasconversion(a::NormalizedPolynomialSpace,b::NormalizedPolynomialSpace) = hasconversion(a.space,b)

function isdiag(D::DerivativeWrapper{<:Operator, <:NormalizedPolynomialSpace})
    sp = domainspace(D)
    csp = canonicalspace(sp)
    isdiag(Derivative(csp, D.order))
end

# Tensor products of normalized and unnormalized spaces may have banded conversions defined
# A banded conversion exists in special cases, where both conversion operators are diagonal
_stripnorm(N::NormalizedPolynomialSpace) = canonicalspace(N)
_stripnorm(x::PolynomialSpace) = x
function _hasconversion_tensor(A, B)
    A1, A2 = A
    B1, B2 = B

    _stripnorm(A1) == _stripnorm(B1) && _stripnorm(A2) == _stripnorm(B2)
end

function hasconversion(A::MaybeNormalizedTensorSpace{<:P1, <:P2},
        B::MaybeNormalizedTensorSpace{<:P1, <:P2}) where {P1<:PolynomialSpace,P2<:PolynomialSpace}

    _hasconversion_tensor(factors(A), factors(B))
end

## All polynomial spaces can be converted provided domains match

isconvertible(a::MaybeNormalized, b::MaybeNormalized) = domain(a) == domain(b)
union_rule(a::MaybeNormalized{<:PolynomialSpace{D}},
            b::MaybeNormalized{<:PolynomialSpace{D}}) where {D} =
    conversion_type(a, b)

function Multiplication(f::Fun{<:PolynomialSpace}, sp::NormalizedPolynomialSpace)
    unnorm_sp = canonicalspace(sp)
    O = ConcreteConversion(unnorm_sp,sp) *
            Multiplication(f,unnorm_sp) * ConcreteConversion(sp, unnorm_sp)
    MultiplicationWrapper(f, O, sp)
end

function Multiplication(f::Fun{<:NormalizedPolynomialSpace}, sp::PolynomialSpace)
    Multiplication(ConcreteConversion(space(f), canonicalspace(f))*f, sp)
end

function Multiplication(f::Fun{<:NormalizedPolynomialSpace}, sp::NormalizedPolynomialSpace)
    fc = ConcreteConversion(space(f), canonicalspace(f))*f
    Multiplication(fc, sp)
end

ApproxFunBase.hasconcreteconversion_canonical(
    @nospecialize(::NormalizedPolynomialSpace), @nospecialize(_)) = true

rangespace(M::MultiplicationWrapper{<:PolynomialSpace,
        <:NormalizedPolynomialSpace}) = domainspace(M)

# evaluation in a normalized space may use the fact that the conversion is concrete
# this improves type-inference, and hence performance
function evaluate(f::AbstractVector, S::NormalizedPolynomialSpace, x...)
    csp = canonicalspace(S)
    C = Conversion_normalizedspace(csp, Val(:backward))
    f_csp = mul_coefficients(C, f)
    evaluate(f_csp, csp, x...)
end

# Methods for concrete operators in normalized spaces
function getindex(D::ConcreteDerivative{<:NormalizedPolynomialSpace}, k::Integer, j::Integer)
    sp = domainspace(D)
    csp = canonicalspace(sp)
    Dcsp = ConcreteDerivative(csp, D.order)
    C = Conversion_normalizedspace(csp, Val(:backward))
    Dcsp[k, j] * C[j, j]
end
