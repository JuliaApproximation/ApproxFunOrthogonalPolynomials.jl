## Derivative

# specialize Derivative so that this is type-inferred even without constant propagation
Derivative(J::Jacobi) = ConcreteDerivative(J,1)
@inline function _Derivative(J::Jacobi, k::Number)
    assert_integer(k)
    k==1 ? ConcreteDerivative(J,1) :
        DerivativeWrapper(
            TimesOperator(
                Derivative(Jacobi(J.b+1,J.a+1,J.domain),k-1),ConcreteDerivative(J,1)),
        J, k)
end
@static if VERSION >= v"1.8"
    Base.@constprop :aggressive Derivative(J::Jacobi, k::Number) =
        _Derivative(J, k)
else
    Derivative(J::Jacobi, k::Number) = _Derivative(J, k)
end


rangespace(D::ConcreteDerivative{<:Jacobi}) = Jacobi(D.space.b+D.order,D.space.a+D.order,domain(D))
bandwidths(D::ConcreteDerivative{<:Jacobi}) = -D.order,D.order
isdiag(D::ConcreteDerivative{<:Jacobi}) = false

getindex(T::ConcreteDerivative{<:Jacobi}, k::Integer, j::Integer) =
    j==k+1 ? eltype(T)((k+1+T.space.a+T.space.b)/complexlength(domain(T))) : zero(eltype(T))


# Evaluation

Evaluation(S::Jacobi,x::typeof(leftendpoint),o::Integer) =
    ConcreteEvaluation(S,x,o)
Evaluation(S::Jacobi,x::typeof(rightendpoint),o::Integer) =
    ConcreteEvaluation(S,x,o)
Evaluation(S::Jacobi,x::Number,o::Integer) = ConcreteEvaluation(S,x,o)

Evaluation(S::NormalizedPolynomialSpace{<:Jacobi},x::typeof(leftendpoint),o::Integer) =
    ConcreteEvaluation(S,x,o)
Evaluation(S::NormalizedPolynomialSpace{<:Jacobi},x::typeof(rightendpoint),o::Integer) =
    ConcreteEvaluation(S,x,o)
Evaluation(S::NormalizedPolynomialSpace{<:Jacobi},x::Number,o::Integer) = ConcreteEvaluation(S,x,o)

## Integral

function Integral(J::Jacobi,k::Number)
    assert_integer(k)
    @assert k > 0 "order of integral must be > 0"
    if k > 1
        Q=Integral(J,1)
        IntegralWrapper(TimesOperator(Integral(rangespace(Q),k-1),Q),J,k)
    elseif J.a > 0 && J.b > 0   # we have a simple definition
        ConcreteIntegral(J,1)
    else   # convert and then integrate
        sp=Jacobi(J.b+1,J.a+1,domain(J))
        C=Conversion(J,sp)
        Q=Integral(sp,1)
        IntegralWrapper(TimesOperator(Q,C),J,1)
    end
end


rangespace(D::ConcreteIntegral{<:Jacobi}) = Jacobi(D.space.b-D.order,D.space.a-D.order,domain(D))
bandwidths(D::ConcreteIntegral{<:Jacobi}) = D.order,-D.order
isdiag(D::ConcreteIntegral{<:Jacobi}) = false

function getindex(T::ConcreteIntegral{<:Jacobi}, k::Integer, j::Integer)
    @assert T.order==1
    if k≥2 && j==k-1
        complexlength(domain(T))./(k+T.space.a+T.space.b-2)
    else
        zero(eltype(T))
    end
end


## Volterra Integral operator

Volterra(d::IntervalOrSegment) = Volterra(Legendre(d))
function Volterra(S::Jacobi, order::Integer)
    @assert S.a == S.b == 0.0
    @assert order==1
    ConcreteVolterra(S,order)
end

rangespace(V::ConcreteVolterra{J}) where {J<:Jacobi}=Jacobi(-1.0,0.0,domain(V))
bandwidths(::ConcreteVolterra{<:Jacobi}) = 1,0

function getindex(V::ConcreteVolterra{J},k::Integer,j::Integer) where J<:Jacobi
    d=domain(V)
    C = complexlength(d)/2
    if k≥2
        if j==k-1
            C/(k-1.5)
        elseif j==k
            -C/(k-0.5)
        else
            zero(eltype(V))
        end
    else
        zero(eltype(V))
    end
end


for (Func,Len,Sum) in ((:DefiniteIntegral,:complexlength,:sum),(:DefiniteLineIntegral,:arclength,:linesum))
    ConcFunc = Symbol(:Concrete, Func)

    @eval begin
        $Func(S::Jacobi{<:IntervalOrSegment}) = $ConcFunc(S)

        function getindex(Σ::$ConcFunc{<:Jacobi{<:IntervalOrSegment},T}, k::Integer) where {T}
            dsp = domainspace(Σ)

            if dsp.b == dsp.a == 0
                # TODO: copy and paste
                k == 1 ? strictconvert(T,$Sum(Fun(dsp,[one(T)]))) : zero(T)
            else
                strictconvert(T,$Sum(Fun(dsp,[zeros(T,k-1);1])))
            end
        end

        function bandwidths(Σ::$ConcFunc{<:Jacobi{<:IntervalOrSegment}})
            if domainspace(Σ).b == domainspace(Σ).a == 0
                0,0  # first entry
            else
                0,ℵ₀
            end
        end
    end
end



## Conversion
# We can only increment by a or b by one, so the following
# multiplies conversion operators to handle otherwise

function Conversion(L::Jacobi,M::Jacobi)
    domain(L) == reverseorientation(domain(M)) &&
        return ConversionWrapper(Conversion(reverseorientation(L), M)*ReverseOrientation(L))

    domain(L) == domain(M) || domain(L) == reverseorientation(domain(M)) ||
        throw(ArgumentError("Domains must be the same"))

    if isapproxinteger(L.a-M.a) && isapproxinteger(L.b-M.b)
        dm=domain(M)
        if isapprox(M.a,L.a) && isapprox(M.b,L.b)
            ConversionWrapper(Operator(I,L))
        elseif (isapprox(M.b,L.b+1) && isapprox(M.a,L.a)) ||
            (isapprox(M.b,L.b) && isapprox(M.a,L.a+1))
            ConcreteConversion(L,M)
        elseif M.b >= L.b && M.a >= L.a
            # We split this into steps where a and b are changed by 1:
            # Define the intermediate space J = Jacobi(M.b, L.a, dm)
            # Conversion(L, M) == Conversion(J, M) * Conversion(L, J)
            # Conversion(L, J) = Conversion(Jacobi(L.b, L.a, dm), Jacobi(M.b, L.a, dm))
            # Conversion(J, M) = Conversion(Jacobi(M.b, L.a, dm), Jacobi(M.b, M.a, dm))
            CLJ = [ConcreteConversion(Jacobi(b-1,L.a,dm), Jacobi(b, L.a, dm)) for b in M.b:-1:L.b+1]
            CJM = [ConcreteConversion(Jacobi(M.b,a-1,dm), Jacobi(M.b, a, dm)) for a in M.a:-1:L.a+1]
            C = [CJM; CLJ]
            ConversionWrapper(TimesOperator(C))
        else
            error("Implement for $L → $M")
        end
    elseif L.a ≈ L.b ≈ 0 && M.a ≈ M.b ≈ 0.5
        Conversion(L,Ultraspherical(L),Ultraspherical(M),M)
    elseif L.a ≈ L.b ≈ 0 && M.a ≈ M.b ≈ -0.5
        Conversion(L,Ultraspherical(L),Chebyshev(M),M)
    else # L.a - M.a ≈ L.b - M.b
        error("Implement for $L → $M")
    end
end

bandwidths(::ConcreteConversion{<:Jacobi,<:Jacobi}) = (0,1)



function Base.getindex(C::ConcreteConversion{<:Jacobi,<:Jacobi,T},k::Integer,j::Integer) where {T}
    L=C.domainspace
    if L.b+1==C.rangespace.b
        if j==k
            k==1 ? strictconvert(T,1) : strictconvert(T,(L.a+L.b+k)/(L.a+L.b+2k-1))
        elseif j==k+1
            strictconvert(T,(L.a+k)./(L.a+L.b+2k+1))
        else
            zero(T)
        end
    elseif L.a+1==C.rangespace.a
        if j==k
            k==1 ? strictconvert(T,1) : strictconvert(T,(L.a+L.b+k)/(L.a+L.b+2k-1))
        elseif j==k+1
            strictconvert(T,-(L.b+k)./(L.a+L.b+2k+1))
        else
            zero(T)
        end
    else
        error("Not implemented")
    end
end




# return the space that has banded Conversion to the other
function conversion_rule(A::Jacobi,B::Jacobi)
    if isapproxinteger(A.a-B.a) && isapproxinteger(A.b-B.b)
        Jacobi(min(A.b,B.b),min(A.a,B.a),domain(A))
    else
        NoSpace()
    end
end



## Ultraspherical Conversion

# Assume m is compatible

function Conversion(A::PolynomialSpace,B::Jacobi)
    J = Jacobi(A)
    J == B ? ConcreteConversion(A,B) :
             ConversionWrapper(TimesOperator(Conversion(J,B),Conversion(A,J)))
end

function Conversion(A::Jacobi,B::PolynomialSpace)
    J = Jacobi(B)
    J == A ? ConcreteConversion(A,B) :
             ConversionWrapper(TimesOperator(Conversion(J,B),Conversion(A,J)))
end

isequalminhalf(x) = x == -0.5
isequalminhalf(@nospecialize ::Integer) = false

function Conversion(A::Jacobi,B::Chebyshev)
    if isequalminhalf(A.a) && isequalminhalf(A.b)
        ConcreteConversion(A,B)
    elseif A.a == A.b == 0
        ConversionWrapper(
            SpaceOperator(
                ConcreteConversion(Ultraspherical(1//2),B),
                A,B))
    elseif A.a == A.b
        US = Ultraspherical(A)
        ConversionWrapper(Conversion(US,B)*ConcreteConversion(A,US))
    else
        J = Jacobi(B)
        ConcreteConversion(J,B)*Conversion(A,J)
    end
end

function Conversion(A::Chebyshev,B::Jacobi)
    if isequalminhalf(B.a) && isequalminhalf(B.b)
        ConcreteConversion(A,B)
    elseif B.a == B.b == 0
        ConversionWrapper(
            SpaceOperator(
                ConcreteConversion(A,Ultraspherical(1//2,domain(B))),
                A,B))
    elseif B.a == B.b
        US = Ultraspherical(B)
        ConcreteConversion(US,B) * Conversion(A,US)
    else
        J = Jacobi(A)
        Conversion(J,B)*ConcreteConversion(A,J)
    end
end


function Conversion(A::Jacobi,B::Ultraspherical)
    if isequalminhalf(A.a) && isequalminhalf(A.b)
        ConversionWrapper(Conversion(Chebyshev(domain(A)),B)*
            ConcreteConversion(A,Chebyshev(domain(A))))
    elseif isequalminhalf(A.a - order(B)) && isequalminhalf(A.b - order(B))
        ConcreteConversion(A,B)
    elseif A.a == A.b == 0
        ConversionWrapper(
            SpaceOperator(
                Conversion(Ultraspherical(1//2),B),
                A,B))
    elseif A.a == A.b
        US = Ultraspherical(A)
        ConversionWrapper(Conversion(US,B)*ConcreteConversion(A,US))
    else
        J = Jacobi(B)
        ConcreteConversion(J,B)*Conversion(A,J)
    end
end

function Conversion(A::Ultraspherical,B::Jacobi)
    if isequalminhalf(B.a) && isequalminhalf(B.b)
        ConversionWrapper(ConcreteConversion(Chebyshev(domain(A)),B)*
            Conversion(A,Chebyshev(domain(A))))
    elseif isequalminhalf(B.a - order(A)) && isequalminhalf(B.b - order(A))
        ConcreteConversion(A,B)
    elseif B.a == B.b == 0
        ConversionWrapper(
            SpaceOperator(
                Conversion(A,Ultraspherical(1//2,domain(B))),
                A,B))
    elseif B.a == B.b
        US = Ultraspherical(B)
        ConversionWrapper(ConcreteConversion(US,B)*Conversion(A,US))
    else
        J = Jacobi(A)
        Conversion(J,B)*ConcreteConversion(A,J)
    end
end




bandwidths(::ConcreteConversion{<:Chebyshev,<:Jacobi}) = 0,0
bandwidths(::ConcreteConversion{<:Jacobi,<:Chebyshev}) = 0,0


bandwidths(::ConcreteConversion{<:Ultraspherical,<:Jacobi}) = 0,0
bandwidths(::ConcreteConversion{<:Jacobi,<:Ultraspherical}) = 0,0

#TODO: Figure out how to unify these definitions
function getindex(::ConcreteConversion{<:Chebyshev,<:Jacobi,T}, k::Integer, j::Integer) where {T}
    if j==k
        one(T)/jacobip(T,k-1,-one(T)/2,-one(T)/2,one(T))
    else
        zero(T)
    end
end

function BandedMatrix(S::SubOperator{T,ConcreteConversion{CC,J,T},NTuple{2,UnitRange{Int}}}) where {J<:Jacobi,CC<:Chebyshev,T}
    ret=BandedMatrix(Zeros, S)
    kr,jr = parentindices(S)
    k=(kr ∩ jr)

    vals = one(T)./jacobip(T,k .- 1,-one(T)/2,-one(T)/2,one(T))

    ret[band(bandshift(S))] = vals
    ret
end


function getindex(::ConcreteConversion{<:Jacobi,<:Chebyshev,T}, k::Integer, j::Integer) where {T}
    if j==k
        jacobip(T,k-1,-one(T)/2,-one(T)/2,one(T))
    else
        zero(T)
    end
end

function BandedMatrix(S::SubOperator{T,ConcreteConversion{J,CC,T},NTuple{2,UnitRange{Int}}}) where {J<:Jacobi,CC<:Chebyshev,T}
    ret=BandedMatrix(Zeros, S)
    kr,jr = parentindices(S)
    k=(kr ∩ jr)

    vals = jacobip(T,k.-1,-one(T)/2,-one(T)/2,one(T))

    ret[band(bandshift(S))] = vals
    ret
end


function getindex(C::ConcreteConversion{<:Ultraspherical,<:Jacobi,T}, k::Integer, j::Integer) where {T}
    if j==k
        S=rangespace(C)
        jp=jacobip(T,k-1,S.a,S.b,one(T))
        um=strictconvert(Operator{T}, Evaluation(setcanonicaldomain(domainspace(C)),rightendpoint,0))[k]::T
        (um/jp)::T
    else
        zero(T)
    end
end

function BandedMatrix(S::SubOperator{T,ConcreteConversion{US,J,T},NTuple{2,UnitRange{Int}}}) where {US<:Ultraspherical,J<:Jacobi,T}
    ret=BandedMatrix(Zeros, S)
    kr,jr = parentindices(S)
    k=(kr ∩ jr)

    sp=rangespace(parent(S))
    jp=jacobip(T,k.-1,sp.a,sp.b,one(T))
    um=Evaluation(T,setcanonicaldomain(domainspace(parent(S))),rightendpoint,0)[k]
    vals = um./jp

    ret[band(bandshift(S))] = vals
    ret
end



function getindex(C::ConcreteConversion{<:Jacobi,<:Ultraspherical,T}, k::Integer, j::Integer) where {T}
    if j==k
        S=domainspace(C)
        jp=jacobip(T,k-1,S.a,S.b,one(T))
        um=Evaluation(T,setcanonicaldomain(rangespace(C)),rightendpoint,0)[k]
        jp/um::T
    else
        zero(T)
    end
end

function BandedMatrix(S::SubOperator{T,ConcreteConversion{J,US,T},NTuple{2,UnitRange{Int}}}) where {US<:Ultraspherical,J<:Jacobi,T}
    ret=BandedMatrix(Zeros, S)
    kr,jr = parentindices(S)
    k=(kr ∩ jr)

    sp=domainspace(parent(S))
    jp=jacobip(T,k.-1,sp.a,sp.b,one(T))
    um=Evaluation(T,setcanonicaldomain(rangespace(parent(S))),rightendpoint,0)[k]
    vals = jp./um

    ret[band(bandshift(S))] = vals
    ret
end


isapproxminhalf(a) = a ≈ -0.5
isapproxminhalf(::Integer) = false

function union_rule(A::Jacobi,B::Jacobi)
    if domainscompatible(A,B)
        Jacobi(min(A.b,B.b),min(A.a,B.a),domain(A))
    else
        NoSpace()
    end
end

function maxspace_rule(A::Jacobi,B::Jacobi)
    if isapproxinteger(A.a-B.a) && isapproxinteger(A.b-B.b)
        Jacobi(max(A.b,B.b),max(A.a,B.a),domain(A))
    else
        NoSpace()
    end
end


function union_rule(A::Chebyshev,B::Jacobi)
    if domainscompatible(A, B)
        if isapproxminhalf(B.a) && isapproxminhalf(B.b)
            # the spaces are the same
            A
        else
            union(Jacobi(A),B)
        end
    else
        if isapproxminhalf(B.a) && isapproxminhalf(B.b)
            union(A, Chebyshev(domain(B)))
        else
            NoSpace()
        end
    end
end
function union_rule(A::Ultraspherical,B::Jacobi)
    m=order(A)
    if domainscompatible(A, B)
        if isapproxminhalf(B.a-m) && isapproxminhalf(B.b-m)
            # the spaces are the same
            A
        else
            union(Jacobi(A),B)
        end
    else
        if isapproxminhalf(B.a-m) && isapproxminhalf(B.b-m)
            union(A, Ultraspherical(m, domain(B)))
        else
            NoSpace()
        end
    end
end

for (OPrule,OP) in ((:conversion_rule,:conversion_type), (:maxspace_rule,:maxspace))
    @eval begin
        function $OPrule(A::Chebyshev,B::Jacobi)
            if isapproxminhalf(B.a) && isapproxminhalf(B.b)
                # the spaces are the same
                A
            elseif isapproxinteger_addhalf(B.a) && isapproxinteger_addhalf(B.b)
                $OP(Jacobi(A),B)
            else
                NoSpace()
            end
        end
        function $OPrule(A::Ultraspherical,B::Jacobi)
            m = order(A)
            if isapproxminhalf(B.a - m) && isapproxminhalf(B.b - m)
                # the spaces are the same
                A
            elseif isapproxinteger_addhalf(B.a) && isapproxinteger_addhalf(B.b)
                $OP(Jacobi(A),B)
            else
                NoSpace()
            end
        end
    end
end

hasconversion(a::Jacobi,b::Chebyshev) = hasconversion(a,Jacobi(b))
hasconversion(a::Chebyshev,b::Jacobi) = hasconversion(Jacobi(a),b)

hasconversion(a::Jacobi,b::Ultraspherical) = hasconversion(a,Jacobi(b))
hasconversion(a::Ultraspherical,b::Jacobi) = hasconversion(Jacobi(a),b)




## Special Multiplication
# special multiplication operators exist when multiplying by
# (1+x) or (1-x) by _decreasing_ the parameter.  Thus the







# represents [b+(1+z)*d/dz] (false) or [a-(1-z)*d/dz] (true)
struct JacobiSD{T} <:Operator{T}
    lr::Bool
    S::Jacobi
end

JacobiSD(lr,S)=JacobiSD{Float64}(lr,S)

convert(::Type{Operator{T}},SD::JacobiSD) where {T}=JacobiSD{T}(SD.lr,SD.S)

domain(op::JacobiSD)=domain(op.S)
domainspace(op::JacobiSD)=op.S
rangespace(op::JacobiSD)=op.lr ? Jacobi(op.S.b+1,op.S.a-1,domain(op.S)) : Jacobi(op.S.b-1,op.S.a+1,domain(op.S))
bandwidths(::JacobiSD)=0,0

function getindex(op::JacobiSD,A,k::Integer,j::Integer)
    m=op.lr ? op.S.a : op.S.b
    if k==j
        k+m-1
    else
        zero(eltype(op))
    end
end
