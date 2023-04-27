export bandmatrices_eigen, SymmetricEigensystem, SkewSymmetricEigensystem

abstract type EigenSystem end

"""
    SymmetricEigensystem(L::Operator, B::Operator, QuotientSpaceType::Type = QuotientSpace)

Represent the eigensystem `L v = λ v` subject to `B v = 0`, where `L` is self-adjoint with respect to the standard `L2`
inner product given the boundary conditions `B`.

!!! note
    No tests are performed to assert that the operator `L` is self-adjoint, and it's the user's responsibility
    to ensure that the operators are compliant.

The optional argument `QuotientSpaceType` specifies the type of space to be used to denote the quotient space in the basis
recombination process. In most cases, the default choice of `QuotientSpace` is a good one. In specific instances where `B`
is rank-deficient (e.g. it contains a column of zeros,
which typically happens if one of the basis elements already satiafies the boundary conditions),
one may need to choose this to be a `PathologicalQuotientSpace`.

!!! note
    No checks on the rank of `B` are carried out, and it's up to the user to specify the correct type.
"""
SymmetricEigensystem

"""
    SkewSymmetricEigensystem(L::Operator, B::Operator, QuotientSpaceType::Type = QuotientSpace)

Represent the eigensystem `L v = λ v` subject to `B v = 0`, where `L` is skew-symmetric with respect to the standard `L2`
inner product given the boundary conditions `B`.

!!! note
    No tests are performed to assert that the operator `L` is skew-symmetric, and it's the user's responsibility
    to ensure that the operators are compliant.

The optional argument `QuotientSpaceType` specifies the type of space to be used to denote the quotient space in the basis
recombination process. In most cases, the default choice of `QuotientSpace` is a good one. In specific instances where `B`
is rank-deficient (e.g. it contains a column of zeros,
which typically happens if one of the basis elements already satiafies the boundary conditions),
one may need to choose this to be a `PathologicalQuotientSpace`.

!!! note
    No checks on the rank of `B` are carried out, and it's up to the user to specify the correct type.
"""
SkewSymmetricEigensystem

for SET in (:SymmetricEigensystem, :SkewSymmetricEigensystem)
    @eval begin
        struct $SET{LT,QT} <: EigenSystem
            L :: LT
            Q :: QT

            function $SET(L, B, ::Type{QST} = QuotientSpace) where {QST}
                L2, B2 = promotedomainspace((L, B))
                if isambiguous(domainspace(L))
                    throw(ArgumentError("could not detect spaces, please specify the domain spaces for the operators"))
                end

                QS = QST(B2)
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
    bandmatrices_eigen(S::Union{SymmetricEigensystem, SkewSymmetricEigensystem}, n::Integer)

Recast the symmetric/skew-symmetric eigenvalue problem `L v = λ v` subject to `B v = 0` to the generalized
eigenvalue problem `SA v = λ SB v`, where `SA` and `SB` are banded operators, and
return the `n × n` matrix representations of `SA` and `SB`.
If `S isa SymmetricEigensystem`, the returned matrices will be `Symmetric`.

!!! note
    No tests are performed to assert that the system is symmetric/skew-symmetric, and it's the user's responsibility
    to ensure that the operators are compliant.
"""
function bandmatrices_eigen(S::SymmetricEigensystem, n::Integer)
    A, B = _bandmatrices_eigen(S, n)
    SA = Symmetric(A, :L)
    SB = Symmetric(B, :L)
    return SA, SB
end

function bandmatrices_eigen(S::EigenSystem, n::Integer)
    A, B = _bandmatrices_eigen(S, n)
    A2 = tril(A, bandwidth(A,1))
    B2 = tril(B, bandwidth(B,1))
    A2, B2
end

function _bandmatrices_eigen(S::EigenSystem, n::Integer)
    AA, BB = basis_recombination(S)
    A = AA[1:n,1:n]
    B = BB[1:n,1:n]
    return A, B
end

function eigvals(S::EigenSystem, n::Integer)
    SA, SB = bandmatrices_eigen(S, n)
    eigvals(SA, SB)
end
