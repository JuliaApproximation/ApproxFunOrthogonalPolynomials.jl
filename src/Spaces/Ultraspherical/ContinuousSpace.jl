export ContinuousSpace

Space(d::PiecewiseSegment) = ContinuousSpace(d)

const PiecewiseSpaceReal{CD} = PiecewiseSpace{CD,<:Any,<:Real}
const PiecewiseSpaceRealChebyDirichlet11 =
        PiecewiseSpaceReal{<:TupleOrVector{ChebyshevDirichlet{1,1}}}

conversion_rule(a::ContinuousSpace, b::PiecewiseSpaceRealChebyDirichlet11) = a

components(S::ContinuousSpace) = [ChebyshevDirichlet{1,1}(s) for s in components(domain(S))]

# We implemnt conversion between continuous space and PiecewiseSpace with Chebyshev dirichlet

function Conversion(ps::PiecewiseSpaceRealChebyDirichlet11, cs::ContinuousSpace)
    @assert ps == canonicalspace(cs)
    ConcreteConversion(ps,cs)
end

function Conversion(cs::ContinuousSpace,ps::PiecewiseSpaceRealChebyDirichlet11)
    @assert ps == canonicalspace(cs)
    ConcreteConversion(cs,ps)
end


bandwidths(C::ConcreteConversion{<:PiecewiseSpaceRealChebyDirichlet11,<:ContinuousSpace}) =
    1,ncomponents(domain(rangespace(C)))


function getindex(C::ConcreteConversion{<:PiecewiseSpaceRealChebyDirichlet11,<:ContinuousSpace,T},
        k::Integer,j::Integer) where {T}
    d=domain(rangespace(C))
    K=ncomponents(d)
    if isperiodic(d)
        if k==j==1
            one(T)
        elseif k==1 && j==K+1
            -one(T)
        elseif 2≤k≤K && (j==k-1 || j==K+k-1)
            one(T)
        elseif K<k && j==k+K
            one(T)
        else
            zero(T)
        end
    else
        if k==j==1
            one(T)
        elseif k==1 && j==K+1
            -one(T)
        elseif 2≤k≤K+1 && (j==k-1 || j==K+k-1)
            one(T)
        elseif K+1<k && j==k+K-1
            one(T)
        else
            zero(T)
        end
    end
end


bandwidths(C::ConcreteConversion{<:ContinuousSpace, <:PiecewiseSpaceRealChebyDirichlet11}) =
            isperiodic(domainspace(C)) ? (2ncomponents(domain(rangespace(C)))-1,1) :
                                         (ncomponents(domain(rangespace(C))),1)

function getindex(C::ConcreteConversion{<:ContinuousSpace,<:PiecewiseSpaceRealChebyDirichlet11,T},
        k::Integer,j::Integer) where {T}
    d=domain(domainspace(C))
    K=ncomponents(d)
    if isperiodic(d)
        if k < K && (j==k || j==k+1)
            one(T)/2
        elseif k==K && (j==K || j==1)
            one(T)/2
        elseif K+1≤k≤2K && j==k-K
            -one(T)/2
        elseif K+1≤k<2K && j==k-K+1
            one(T)/2
        elseif k==2K && j==1
            one(T)/2
        elseif k>2K && j==k-K
            one(T)
        else
            zero(T)
        end
    else
        if k≤K && (j==k || j==k+1)
            one(T)/2
        elseif K+1≤k≤2K && j==k-K
            -one(T)/2
        elseif K+1≤k≤2K && j==k-K+1
            one(T)/2
        elseif k>2K && j==k-K+1
            one(T)
        else
            zero(T)
        end
    end
end



# Dirichlet for Squares


const TensorChebyshevDirichlet = TensorSpace{<:Tuple{<:ChebyshevDirichlet{1,1,<:IntervalOrSegment},
                                  <:ChebyshevDirichlet{1,1,<:IntervalOrSegment}}}

@inline Dirichlet(S::TensorChebyshevDirichlet,k) = k == 0 ? ConcreteDirichlet(S,0) : tensor_Dirichlet(S,k)

Dirichlet(d::RectDomain) =
    Dirichlet(ChebyshevDirichlet{1,1}(factor(d,1))*ChebyshevDirichlet{1,1}(factor(d,2)))

isblockbanded(::Dirichlet{<:TensorChebyshevDirichlet}) = true

blockbandwidths(::Dirichlet{<:TensorChebyshevDirichlet}) = (0,2)

colstop(B::Dirichlet{<:TensorChebyshevDirichlet}, j::Integer) = j ≤ 3 ? 4 : 4(block(domainspace(B),j).n[1]-1)


function getindex(B::ConcreteDirichlet{<:TensorChebyshevDirichlet}, k::Integer,j::Integer)
    T = eltype(B)
    ds = domainspace(B)
    rs = rangespace(B)
    if j == 1 && k ≤ 4
        one(T)
    elseif j == 2 && k ≤ 2
        -one(T)
    elseif j == 2 && k ≤ 4
        one(T)
    elseif j == 3 && (k == 1 || k == 4)
        -one(T)
    elseif j == 3 && (k == 2 || k == 3)
        one(T)
    elseif j == 5 && (k == 2 || k == 4)
        -one(T)
    elseif j == 5 && (k == 1 || k == 3)
        one(T)
    elseif j == 5 || j ≤ 3
        zero(T)
    else
        K = Int(block(rs,k))
        J = Int(block(ds,j))
        m = mod(k-1,4)
        s,t =  blockstart(ds,J),  blockstop(ds,J)
        if K == J-1 && (m == 1  && j == s ||
                       (m == 0  && j == t))
            one(T)
        elseif K == J-1 && ((m == 3 && j == s) ||
                            (m == 2 && j == t))
            iseven(K) ? one(T) : -one(T)
        elseif K == J-2 && m == 1 && j == s+1
            one(T)
        elseif K == J-2 && m == 2 && j == t-1
            iseven(K) ? one(T) : -one(T)
        elseif K == J-2 && m == 0 && j == t-1
            -one(T)
        elseif K == J-2 && m == 3 && j == s+1
            iseven(K) ? -one(T) : one(T)
        else
            zero(T)
        end
    end
end


function BlockBandedMatrix(S::SubOperator{T,<:ConcreteDirichlet{<:TensorChebyshevDirichlet},
                                NTuple{2,UnitRange{Int}}}) where {T}
    P=parent(S)
    ret=BlockBandedMatrix(Zeros, S)
    kr,jr=parentindices(S)

    K1=block(rangespace(P),kr[1])
    Kr1=blockstart(rangespace(P),K1)
    J1=block(domainspace(P),jr[1])
    Jr1=blockstart(domainspace(P),J1)

    if ret.rows[1]>0 && ret.cols[1]==1
        view(ret,Block(1),Block(1))[:,1] = 1
    end


    if ret.rows[1] > 0 && ret.cols[2] > 0
        B=view(ret,Block(1),Block(2))

        k_sh = kr[1]-1; j_sh = max(jr[1]-2,0)
        if j_sh == 0
            # first column
            k_sh == 0 && (B[1,1-j_sh]=-1)
            k_sh ≤  1 && (B[2-k_sh,1-j_sh]=-1)
            k_sh ≤ 2 && (B[3-k_sh,1-j_sh]=1)
            B[4-k_sh,1-j_sh]=1
        end
        # second column
        k_sh == 0 && (B[1-k_sh,2-j_sh]=-1)
        k_sh ≤ 1  && (B[2-k_sh,2-j_sh]=1)
        k_sh ≤ 2  && (B[3-k_sh,2-j_sh]=1)

        B[4-k_sh,2-j_sh]=-1
    end


    if ret.rows[1] > 0 && ret.cols[3] > 0
        B=view(ret,Block(1),Block(3))

        k_sh = kr[1]-1; j_sh = max(jr[1]-4,0)
        # second column
        k_sh == 0 && (B[1-k_sh,2-j_sh]=1)
        k_sh ≤ 1 && (B[2-k_sh,2-j_sh]=-1)
        k_sh ≤ 2 && (B[3-k_sh,2-j_sh]=1)
        B[4-k_sh,2-j_sh]=-1
    end
    for K=Block(2):2:Block(min(length(ret.rows),length(ret.cols)-1))
        J = K+1  # super-diagonal block
        N = ret.rows[K.n[1]]
        M = ret.cols[J.n[1]]
        if N ≠ 0 && M ≠ 0
            # calculate shift
            k_sh = K == K1 ? kr[1]-Kr1 : 0
            j_sh = J == J1 ? jr[1]-Jr1 : 0
            B = view(ret,K,J)

            1 ≤ 2-k_sh ≤ N && j_sh == 0 && (B[2-k_sh,1-j_sh]=1)
            1 ≤ 4-k_sh ≤ N && j_sh == 0 && (B[4-k_sh,1-j_sh]=1)
            k_sh == 0 && 1 ≤ J.n[1]-j_sh ≤ M && (B[1-k_sh,J.n[1]-j_sh]=1)
            k_sh ≤ 2 &&  1 ≤ J.n[1]-j_sh ≤ M && (B[3-k_sh,J.n[1]-j_sh]=1)
        end
    end
    for K=Block(3):2:Block(min(length(ret.rows),length(ret.cols)-1))
        J = K+1  # super-diagonal block
        N = ret.rows[K.n[1]]
        M = ret.cols[J.n[1]]
        if N ≠ 0 && M ≠ 0
            # calculate shift
            k_sh = K == K1 ? kr[1]-Kr1 : 0
            j_sh = J == J1 ? jr[1]-Jr1 : 0
            B = view(ret,K,J)

            1 ≤ 2-k_sh ≤ N && j_sh == 0 && (B[2-k_sh,1-j_sh]=1)
            1 ≤ 4-k_sh ≤ N && j_sh == 0 && (B[4-k_sh,1-j_sh]=-1)
            k_sh == 0 && 1 ≤ J.n[1]-j_sh ≤ M && (B[1-k_sh,J.n[1]-j_sh]=1)
            1 ≤ 3-k_sh ≤ N &&  1 ≤ J.n[1]-j_sh ≤ M && (B[3-k_sh,J.n[1]-j_sh]=-1)
        end
    end
    for K=Block(2):2:Block(min(length(ret.rows),length(ret.cols)-2))
        J = K+2  # super-diagonal block
        N = ret.rows[K.n[1]]
        M = ret.cols[J.n[1]]

        if N ≠ 0 && M ≠ 0
            B=view(ret,K,J)
            # calculate shift
            k_sh = K == K1 ? kr[1]-Kr1 : 0
            j_sh = J == J1 ? jr[1]-Jr1 : 0
            B = view(ret,K,J)

            1 ≤ 2-k_sh ≤ N && 1 ≤ 2-j_sh ≤ M && (B[2-k_sh,2-j_sh]=1)
            1 ≤ 4-k_sh ≤ N && 1 ≤ 2-j_sh ≤ M && (B[4-k_sh,2-j_sh]=-1)
            k_sh == 0 && 1 ≤ J.n[1]-j_sh-1 ≤ M && (B[1,J.n[1]-j_sh-1]=-1)
            1 ≤ 3-k_sh ≤ N &&  1 ≤ J.n[1]-j_sh-1 ≤ M && (B[3-k_sh,J.n[1]-j_sh-1]=1)
        end
    end
    for K=Block(3):2:Block(min(length(ret.rows),length(ret.cols)-2))
        J = K+2
        B=view(ret,K,J)
        N = ret.rows[K.n[1]]
        M = ret.cols[J.n[1]]

        if N ≠ 0 && M ≠ 0
            B=view(ret,K,J)
            # calculate shift
            k_sh = K == K1 ? kr[1]-Kr1 : 0
            j_sh = J == J1 ? jr[1]-Jr1 : 0
            B = view(ret,K,J)

            1 ≤ 2-k_sh ≤ N && 1 ≤ 2-j_sh ≤ M && (B[2-k_sh,2-j_sh]=1)
            1 ≤ 4-k_sh ≤ N && 1 ≤ 2-j_sh ≤ M && (B[4-k_sh,2-j_sh]=1)
            k_sh == 0 && 1 ≤ J.n[1]-j_sh-1 ≤ M && (B[1,J.n[1]-j_sh-1]=-1)
            1 ≤ 3-k_sh ≤ N &&  1 ≤ J.n[1]-j_sh-1 ≤ M && (B[3-k_sh,J.n[1]-j_sh-1]=-1)
        end
    end

    ret
end


union_rule(A::ContinuousSpace, B::PolynomialSpace{<:IntervalOrSegment}) =
    Space(domain(A) ∪ domain(B))
