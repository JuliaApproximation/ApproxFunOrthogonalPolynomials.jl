module EigTest

using ApproxFunOrthogonalPolynomials
using ApproxFunBase
using ApproxFunBase: bandwidth
using BandedMatrices
using LinearAlgebra
using SpecialFunctions
using Test

@testset "Eigenvalue problems" begin
    @testset "Negative Laplacian with Neumann boundary conditions" begin
        #
        # -𝒟² u = λu,  u'(±1) = 0.
        #
        d = Segment(-1..1)
        for S in (Ultraspherical(0.5, d), Legendre(d))
            L = -Derivative(S, 2)
            B = Neumann(S)

            QS = QuotientSpace(B)
            @inferred (QS -> QS.F)(QS)

            n = 50
            Seig = SymmetricEigensystem(L, B)
            SA, SB = bandmatrices_eigen(Seig, n)

            # A hack to avoid a BandedMatrices bug on Julia v1.6, with a pencil (A, B)
            # with smaller bandwidth in A.
            λ = if VERSION >= v"1.8"
                eigvals(SA, SB)
            else
                eigvals(Symmetric(Matrix(SA)), Symmetric(Matrix(SB)))
            end

            @test λ[1:round(Int, 2n/5)] ≈ (π^2/4).*(0:round(Int, 2n/5)-1).^2
        end
    end

    @testset "Schrödinger with piecewise-linear potential with Dirichlet boundary conditions" begin
        #
        # [-𝒟² + V] u = λu,  u(±1) = 0,
        #
        # where V = 100|x|.
        #
        d = Segment(-1..0)∪Segment(0..1)
        S = PiecewiseSpace(Ultraspherical.(0.5, components(d)))
        V = 100Fun(abs, S)
        L = -Derivative(S, 2) + V
        B = [Dirichlet(S); continuity(S, 0:1)]

        Seig = SymmetricEigensystem(L, B)
        n = 100
        λ = eigvals(Seig, n)

        @test λ[1] ≈ parse(BigFloat, "2.19503852085715200848808942880214615154684642693583513254593767079468401198338e+01")
    end

    @testset "Schrödinger with piecewise-constant potential with Dirichlet boundary conditions" begin
        #
        # [-𝒟² + V] u = λu,  u(±1) = 0,
        #
        # where V = 1000[χ_[-1,-1/2](x) + χ_[1/2,1](x)].
        #
        d = Segment(-1..(-0.5)) ∪ Segment(-0.5..0.5) ∪ Segment(0.5..1)
        S = PiecewiseSpace(Ultraspherical.(0.5, components(d)))

        V = Fun(x->abs(x) ≥ 1/2 ? 1000 : 0, S)
        L = -Derivative(S, 2) + V
        B = [Dirichlet(S); continuity(S, 0:1)]

        Seig = SymmetricEigensystem(L, B)

        n = 150
        λ = eigvals(Seig, n)
        # From Lee--Greengard (1997).
        λtrue = [2.95446;5.90736;8.85702;11.80147].^2
        @test norm((λ[1:4] - λtrue)./λ[1:4]) < 1e-5
    end

    @testset "Schrödinger with linear + Dirac potential with Robin boundary conditions" begin
        #
        # [-𝒟² + V] u = λu,  u(-1) = u(1) + u'(1) = 0,
        #
        # where V = x + 100δ(x-0.25).
        #
        d = Segment(-1..0.25) ∪ Segment(0.25..1)
        S = PiecewiseSpace(Ultraspherical.(0.5, components(d)))
        V = Fun(identity, S)
        L = -Derivative(S, 2) + V

        B4 = zeros(Operator{ApproxFunBase.prectype(S)}, 1, 2)
        B4[1, 1] = -Evaluation(component(S, 1), rightendpoint, 1) - 100*0.5*Evaluation(component(S, 1), rightendpoint)
        B4[1, 2] = Evaluation(component(S, 2), leftendpoint, 1)  - 100*0.5*Evaluation(component(S, 2), leftendpoint)
        B4 = ApproxFunBase.InterlaceOperator(B4, PiecewiseSpace, ApproxFunBase.ArraySpace)
        B = [Evaluation(S, -1); Evaluation(S, 1) + Evaluation(S, 1, 1); continuity(S, 0); B4]

        n = 100
        Seig = SymmetricEigensystem(L, B)
        SA, SB = bandmatrices_eigen(Seig, n)
        λ, Q = eigen(SA, SB);

        QS = QuotientSpace(B)
        k = 3
        u_QS = Fun(QS, Q[:, k])
        u_S = Fun(u_QS, S)
        u = Fun(u_S, PiecewiseSpace(Chebyshev.(components(d))))
        u /= sign(u'(-1))
        u1, u2 = components(u)

        @test norm(u(-1)) < 100eps()
        @test u(1) ≈ -u'(1)
        @test u1(0.25) ≈ u2(0.25)
        @test u2'(0.25) - u1'(0.25) ≈ 100*u(0.25)
        @test -u1'' + component(V, 1)*u1 ≈ λ[k]*u1
        @test -u2'' + component(V, 2)*u2 ≈ λ[k]*u2

        λ2, f = eigs(Seig, n, tolerance=1e-8);
        @test λ2[1:10] ≈ λ[1:10]
        @test f[k] ≈ u || f[k] ≈ -u
    end

    @testset "BigFloat negative Laplacian with Dirichlet boundary conditions" begin
        #
        # -𝒟² u = λu,  u(±1) = 0.
        #
        d = Segment(big(-1.0)..big(1.0))
        S = Ultraspherical(big(0.5), d)
        L = -Derivative(S, 2)
        C = Conversion(domainspace(L), rangespace(L))
        B = Dirichlet(S)

        n = 300
        Seig = SymmetricEigensystem(L, B)
        SA, SB = bandmatrices_eigen(Seig, n)
        BSA = BandedMatrix(SA)
        BSB = BandedMatrix(SB)
        begin
            v = zeros(BigFloat, n)
            v[1] = 1
            v[3] = 1/256
            λ = big(0.0)
            for _ = 1:7
                λ = dot(v, BSA*v)/dot(v, BSB*v)
                OP = Symmetric(SA.data-λ*SB.data, :L)
                F = ldlt!(OP)
                ldiv!(F, v)
                normalize!(v)
            end
            @test λ ≈ big(π)^2/4
        end
    end

    @testset "Skew differentiation matrices" begin
        #
        # 𝒟 u = λ u,  u(1) + u(-1) = 0.
        #
        d = Segment(-1..1)
        S = Ultraspherical(0.5, d)
        Lsk = Derivative(S)
        B = Evaluation(S, -1) + Evaluation(S, 1)
        Seig = SkewSymmetricEigensystem(Lsk, B, PathologicalQuotientSpace)

        n = 100
        λ = eigvals(Seig, n)
        λim = imag(sort!(λ, by = abs))

        @test abs.(λim[1:2:round(Int, 2n/5)]) ≈ π.*(0.5:round(Int, 2n/5)/2)
        @test abs.(λim[2:2:round(Int, 2n/5)]) ≈ π.*(0.5:round(Int, 2n/5)/2)
    end

    @testset "Complex spectra of principal finite sections of an ultraspherical discretization of a self-adjoint problem" begin
        #
        # -𝒟² u = λ u,  u'(±1) = 0.
        #
        S = Chebyshev()
        L = -Derivative(S, 2)
        B = Neumann(S)
        C = Conversion(domainspace(L), rangespace(L))
        n = 10 # ≥ 10 appears to do the trick
        λ = eigvals(Matrix([B;L][1:n,1:n]), Matrix([B-B;C][1:n,1:n]))
        @test eltype(λ) == Complex{Float64}
    end
end

end # module
