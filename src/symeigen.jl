export symmetric_bandmatrices_eigen, SymmetricEigensystem

"""
    SymmetricEigensystem(L::Operator, B::Operator)

Represent the eigensystem `L v = λ v` subject to `B v = 0`, where `L` is self-adjoint with respect to the standard `L2`
inner product given the boundary conditions `B`.
The `domainspace` of the operators must be one of `Legendre(::Domain)` or `Ultraspherical(0.5, ::Domain)`.

!!! note
    No tests are performed to assert that the system is self-adjoint, and it's the user's responsibility
    to ensure that the operators are compliant.
"""
struct SymmetricEigensystem{LT,QT}
    L :: LT
    Q :: QT

    function SymmetricEigensystem(L, B)
        L2, B2 = promotedomainspace((L, B))
        if isambiguous(domainspace(L))
            throw(ArgumentError("could not detect spaces, please specify the domain spaces for the operators"))
        end
        d = domain(L2)
        if !(domainspace(L2) == Legendre(d) || domainspace(L2) == Ultraspherical(0.5, d))
            throw(ArgumentError("domainspace of the operators must be $(Legendre(d)) or $(Ultraspherical(0.5,d)) "*
                "for the symmetric banded conversion, received $(domainspace(L2))"))
        end

        QS = QuotientSpace(B2)
        S = domainspace(L2)
        Q = Conversion(QS, S)
        new{typeof(L2),typeof(Q)}(L2, Q)
    end
end

function basis_recombination(SE::SymmetricEigensystem)
    L, Q = SE.L, SE.Q
    S = domainspace(L)
    NS = NormalizedPolynomialSpace(S)
    D1 = ConcreteConversion(S, NS)
    D2 = ConcreteConversion(NS, S)
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
The `domainspace` of the operators must be one of `Legendre(::Domain)` or `Ultraspherical(0.5, ::Domain)`.

!!! note
    No tests are performed to assert that the system is self-adjoint, and it's the user's responsibility
    to ensure that the operators are compliant.
"""
function symmetric_bandmatrices_eigen(L::Operator, B::Operator, n::Integer)
    symmetric_bandmatrices_eigen(SymmetricEigensystem(L, B), n)
end
function symmetric_bandmatrices_eigen(S::SymmetricEigensystem, n::Integer)
    AA, BB = basis_recombination(S)
    SA = Symmetric(AA[1:n,1:n], :L)
    SB = Symmetric(BB[1:n,1:n], :L)
    return SA, SB
end
