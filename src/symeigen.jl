export symmetric_bandmatrices_eigen

function symmetrize_operator_eigen(L, B)
    L2, B2 = promotedomainspace((L, B))
    if isambiguous(domainspace(L))
        throw(ArgumentError("could not detect spaces, please specify the domain spaces for the operators"))
    end
    d = domain(L2)
    if !(domainspace(L2) == Legendre(d) || domainspace(L2) == Ultraspherical(0.5, d))
        throw(ArgumentError("domainspace of the operators must be $(Legendre(d)) or $(Ultraspherical(0.5,d)) "*
            "for the symmetric banded conversion, received $(domainspace(L2))"))
    end
    S = domainspace(L2)
    NS = NormalizedPolynomialSpace(S)
    D1 = Conversion(S, NS)
    D2 = Conversion(NS, S);
    QS = QuotientSpace(B2)
    Q = Conversion(QS, S)
    R = D1*Q;
    C = Conversion(domainspace(L2), rangespace(L2))
    P = cache(PartialInverseOperator(C, (0, bandwidth(L2, 1) + bandwidth(R, 1) + bandwidth(C, 2))));
    A = R'D1*P*L2*D2*R
    BB = R'R;

    return A, BB
end

"""
    symmetric_bandmatrices_eigen(L::Operator, B::Operator, n::Integer)

Recast the self-adjoint eigenvalue problem `L v = λ v` subject to `B v = 0` to the generalized
eigenvalue problem `SA v = λ SB v`, where `SA` and `SB` are `Symmetric(::BandedMatrix)`es, and
return `SA` and `SB`. The `domainspace` of the operators must be one of `Legedre(::Domain)` or
`Ultraspherical(0.5, ::Domain)`.

!!! note
    No tests are performed to assert that the system is self-adjoint, and it's the user's responsibility
    to ensure that this holds.
"""
function symmetric_bandmatrices_eigen(L, B, n)
    AA, BB = symmetrize_operator_eigen(L, B)
    SA = Symmetric(AA[1:n,1:n], :L)
    SB = Symmetric(BB[1:n,1:n], :L)
    return SA, SB
end
