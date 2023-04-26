export symmetric_bandmatrices_eigen, bandmatrices_eigen, SymmetricEigensystem, SkewSymmetricEigensystem

abstract type EigenSystem end

"""
    SymmetricEigensystem(L::Operator, B::Operator)

Represent the eigensystem `L v = λ v` subject to `B v = 0`, where `L` is self-adjoint with respect to the standard `L2`
inner product given the boundary conditions `B`.

!!! note
    No tests are performed to assert that the system is self-adjoint, and it's the user's responsibility
    to ensure that the operators are compliant.
"""
SymmetricEigensystem

"""
    SkewSymmetricEigensystem(L::Operator, B::Operator)

Represent the eigensystem `L v = λ v` subject to `B v = 0`, where `L` is skew-symmetric with respect to the standard `L2`
inner product given the boundary conditions `B`.

!!! note
    No tests are performed to assert that the system is skew-symmetric, and it's the user's responsibility
    to ensure that the operators are compliant.
"""
SymmetricEigensystem

_quotientspace(B, ::Val{:SymmetricEigensystem}) = QuotientSpace(B)
_quotientspace(B, ::Val{:SkewSymmetricEigensystem}) = PathologicalQuotientSpace(B)

for SET in (:SymmetricEigensystem, :SkewSymmetricEigensystem)
    qv = Val(SET)
    @eval begin
        struct $SET{LT,QT} <: EigenSystem
            L :: LT
            Q :: QT

            function $SET(L, B)
                L2, B2 = promotedomainspace((L, B))
                if isambiguous(domainspace(L))
                    throw(ArgumentError("could not detect spaces, please specify the domain spaces for the operators"))
                end

                QS = _quotientspace(B2, $qv)
                S = domainspace(L2)
                Q = Conversion(QS, S)
                new{typeof(L2),typeof(Q)}(L2, Q)
            end
        end
    end
end

function basis_recombination(SE::EigenSystem)
    L, Q = SE.L, SE.Q
    S = domainspace(L)
    D1 = Conversion_normalizedspace(S)
    D2 = Conversion_normalizedspace(S, Val(:backward))
    R = D1*Q;
    C = Conversion(S, rangespace(L))
    P = cache(PartialInverseOperator(C, (0, bandwidth(L, 1) + bandwidth(R, 1) + bandwidth(C, 2))));
    A = R'D1*P*L*D2*R
    BB = R'R;

    return A, BB
end

"""
    symmetric_bandmatrices_eigen(L::Operator, B::Operator, n::Integer)

Recast the self-adjoint eigenvalue problem `L v = λ v` subject to `B v = 0` to the generalized
eigenvalue problem `SA v = λ SB v`, where `SA` and `SB` are symmetric banded operators, and
return the `n × n` matrix representations of `SA` and `SB`.

!!! note
    No tests are performed to assert that the system is self-adjoint, and it's the user's responsibility
    to ensure that the operators are compliant.
"""
function symmetric_bandmatrices_eigen(L::Operator, B::Operator, n::Integer)
    bandmatrices_eigen(SymmetricEigensystem(L, B), n)
end

function bandmatrices_eigen(S::SymmetricEigensystem, n::Integer)
    A, B = _bandmatrices_eigen(S, n)
    SA = Symmetric(A, :L)
    SB = Symmetric(B, :L)
    return SA, SB
end

"""
    bandmatrices_eigen(S::Union{SymmetricEigensystem, SkewSymmetricEigensystem}, n::Integer)

Recast the symmetric/skew-symmetric eigenvalue problem `L v = λ v` subject to `B v = 0` to the generalized
eigenvalue problem `SA v = λ SB v`, where `SA` and `SB` are banded operators, and
return the `n × n` matrix representations of `SA` and `SB`.
If `S isa SymmetricEigensystem`, the returned matrices will be `Symmetric`.

!!! note
    No tests are performed to assert that the system is symmetric/skew-symmetric, and it's the user's responsibility
    to ensure that the operators are compliant.
"""
bandmatrices_eigen(S::EigenSystem, n::Integer) = _bandmatrices_eigen(S, n)

function _bandmatrices_eigen(S::EigenSystem, n::Integer)
    AA, BB = basis_recombination(S)
    A = AA[1:n,1:n]
    B = BB[1:n,1:n]
    return A, B
end

function eigvals(S::SymmetricEigensystem, n::Integer)
    SA, SB = bandmatrices_eigen(S, n)
    eigvals(SA, SB)
end
