using ApproxFunOrthogonalPolynomials
using ApproxFunBase
using Test
using SpecialFunctions
using LinearAlgebra
using ApproxFunBase: maxspace, NoSpace, hasconversion,
                    reverseorientation, ReverseOrientation, transform!, itransform!
using ApproxFunBaseTest: testbandedbelowoperator, testbandedoperator, testspace, testtransforms,
                    testfunctional
using ApproxFunOrthogonalPolynomials: jacobip
using StaticArrays: SVector
using Static

@verbose @testset "Jacobi" begin
    @testset "Basic" begin
        @test jacobip(0:5,2,0.5,0.1) ≈ [1.,0.975,-0.28031249999999996,-0.8636328125,-0.0022111816406250743,0.7397117980957031]

        testspace(Jacobi(.5,2.);haslineintegral=false)

        f=Fun(exp,Jacobi(.5,2.))
        @test f(.1) ≈ exp(.1)

        f=Fun(x->cos(100x),Jacobi(.5,2.124),500)
        @test f(.1) ≈ cos(100*.1)


        sp=Jacobi(.5,2.124)
        @time f=Fun(exp,sp)
        sp2=Jacobi(1.5,2.124)
        f2=Fun(exp,sp2)
        sp3=Jacobi(1.5,3.124)
        f3=Fun(exp,sp3)
        sp4=Jacobi(2.5,4.124)
        f4=Fun(exp,sp4)
        @test norm((Fun(f,sp2)-f2).coefficients)<100eps()
        @test norm((Fun(f,sp3)-f3).coefficients)<100eps()
        @test norm((Fun(f,sp4)-f4).coefficients)<200eps()

        f = Fun(Jacobi(0,0), Float64[1,2,3])
        for x in [-1, 0, 1]
            g = Evaluation(x) * f
            @test ncoefficients(g) == 1
            @test coefficients(g)[1] == f(x)
        end

        @testset "spacescompatible" begin
            x = 1.0
            y = 1.0 + eps(1.0)
            Jx = Jacobi(x,x)
            Jy = Jacobi(y,y)
            @test ApproxFunBase.spacescompatible(Jx, Jy)
            @test ApproxFunBase.spacescompatible(Ultraspherical(Jx), Ultraspherical(Jy))
        end
    end

    @testset "Conversion" begin
        testtransforms(Jacobi(-0.5,-0.5))
        @test norm(Fun(Fun(exp),Jacobi(-0.5,-0.5))-Fun(exp,Jacobi(-0.5,-0.5))) < 300eps()

        @testset for d in (-1..1, 0..1, ChebyshevInterval())
            @testset "Chebyshev" begin
                f = Fun(x->x^2, Chebyshev(d))
                C = space(f)
                for J1 in (Jacobi(-0.5, -0.5, d), Legendre(d),
                                Jacobi(0.5, 0.5, d), Jacobi(2.5, 1.5, d))
                    @testset for J in (J1, NormalizedPolynomialSpace(J1))
                        g = Fun(f, J)
                        if !any(isnan, coefficients(g))
                            @test Conversion(C, J) * f ≈ g
                        end
                    end
                end
                # Jacobi(-0.5, -0.5) to NormalizedJacobi(-0.5, -0.5) can have a NaN in the (1,1) index
                # if the normalization is not carefully implemented,
                # so this test checks that this is not the case
                g = Conversion(Jacobi(-0.5,-0.5), NormalizedJacobi(-0.5,-0.5)) * Fun(x->x^3, Jacobi(-0.5, -0.5))
                @test coefficients(g) ≈ coefficients(Fun(x->x^3, NormalizedChebyshev()))
            end
            @testset "legendre" begin
                f = Fun(x->x^2, Legendre(d))
                C = space(f)
                for J1 in (Jacobi(-0.5, -0.5, d), Legendre(d),
                                Jacobi(0.5, 0.5, d), Jacobi(1, 1, d))
                    @testset for J in (J1, NormalizedPolynomialSpace(J1))
                        g = Fun(f, J)
                        if !any(isnan, coefficients(g))
                            @test Conversion(C, J) * f ≈ g
                        end
                    end
                end
            end
            @testset "Ultraspherical(0.5)" begin
                f = Fun(x->x^2, Ultraspherical(0.5,d))
                U = space(f)
                for J1 in (Jacobi(-0.5, -0.5, d), Legendre(d),
                                Jacobi(0.5, 0.5, d))
                    @testset for J in (J1, NormalizedPolynomialSpace(J1))
                        g = Fun(f, J)
                        if !any(isnan, coefficients(g))
                            @test Conversion(U, J) * f ≈ g
                        end
                    end
                end
            end
            @testset "Ultraspherical(1)" begin
                f = Fun(x->x^2, Ultraspherical(1,d))
                U = space(f)
                for J1 in (Legendre(d), Jacobi(0.5, 0.5, d), Jacobi(1.5, 0.5, d),
                            Jacobi(0.5, 1.5, d))
                    @testset for J in (J1,)
                        g = Fun(f, J)
                        if !any(isnan, coefficients(g))
                            @test Conversion(U, J) * f ≈ g
                        end
                    end
                end
            end
        end

        @testset for (S1, S2) in ((Legendre(), Jacobi(-0.5, -0.5)), (Jacobi(-0.5, -0.5), Legendre()),
                (Legendre(), Jacobi(1.5, 1.5)))
            @test Conversion(S1, S2) * Fun(x->x^4, S1) ≈ Fun(x->x^4, S2)
        end

        @testset "inference tests" begin
            #= Note all cases are inferred as of now,
            but as the situation eveolves in the future, more @inferred tests
            may be added
            There are also issues with static float promotion, because of which
            we don't use the floating point orders directly in tests
            See https://github.com/SciML/Static.jl/issues/97
            =#
            L = Jacobi(static(0), static(0))
            @testset "Jacobi" begin
                CLL = @inferred Conversion(L, L)
                @test convert(Number, CLL) == 1
                NL = NormalizedPolynomialSpace(L)
                CNLNL = @inferred Conversion(NL, NL)
                @test convert(Number, CNLNL) == 1
                CLNL = @inferred Conversion(L, NL)
                @test CLNL * Fun(Legendre()) ≈ Fun(NormalizedLegendre())
                CNLL = @inferred Conversion(NL, L)
                @test CNLL * Fun(NormalizedLegendre()) ≈ Fun(Legendre())
            end

            @testset "Chebyshev" begin
                CCL = @inferred Conversion(Chebyshev(), L)
                @test CCL * Fun(Chebyshev()) ≈ Fun(Legendre())
                CLC = @inferred Conversion(L, Chebyshev())
                @test CLC * Fun(Legendre()) ≈ Fun(Chebyshev())

                @inferred Conversion(Chebyshev(), Jacobi(static(-0.5), static(-0.5)))
                CCJmhalf = Conversion(Chebyshev(), Jacobi(-0.5, -0.5))
                @test CCJmhalf * Fun(Chebyshev()) ≈ Fun(Jacobi(-0.5,-0.5))
                @inferred Conversion(Jacobi(static(-0.5),static(-0.5)), Chebyshev())
                CJmhalfC = Conversion(Jacobi(-0.5,-0.5), Chebyshev())
                @test CJmhalfC * Fun(Jacobi(-0.5,-0.5)) ≈ Fun(Chebyshev())

                @inferred Conversion(Chebyshev(), Jacobi(static(0.5), static(0.5)))
                CCJmhalf = Conversion(Chebyshev(), Jacobi(0.5, 0.5))
                @test CCJmhalf * Fun(Chebyshev()) ≈ Fun(Jacobi(0.5,0.5))

                CCJ1 = Conversion(Chebyshev(), Jacobi(1,1))
                @test CCJ1 * Fun(Chebyshev()) ≈ Fun(Jacobi(1,1))

                CCJmix = Conversion(Chebyshev(), Jacobi(0.5,1.5))
                @test CCJmix * Fun(Chebyshev()) ≈ Fun(Jacobi(0.5,1.5))
            end

            @testset "Ultraspherical" begin
                CUL = @inferred Conversion(Ultraspherical(static(0.5)), L)
                @test CUL * Fun(Ultraspherical(0.5)) ≈ Fun(Legendre())
                CLU = @inferred Conversion(L, Ultraspherical(static(0.5)))
                @test CLU * Fun(Legendre()) ≈ Fun(Ultraspherical(0.5))

                @inferred Conversion(Ultraspherical(static(0.5)), Jacobi(static(1),static(1)))
                CU0J1 = Conversion(Ultraspherical(0.5), Jacobi(1,1))
                @test CU0J1 * Fun(Ultraspherical(0.5)) ≈ Fun(Jacobi(1,1))
                @inferred Conversion(Jacobi(static(1),static(1)), Ultraspherical(static(2.5)))
                CJ1U2 = Conversion(Jacobi(1,1), Ultraspherical(2.5))
                @test CJ1U2 * Fun(Jacobi(1,1)) ≈ Fun(Ultraspherical(2.5))
            end
        end

        @testset "conversion between spaces" begin
            for u in (Ultraspherical(1), Chebyshev())
                @test NormalizedPolynomialSpace(Jacobi(u)) ==
                    NormalizedJacobi(NormalizedPolynomialSpace(u))
            end
            for j in (Legendre(), Jacobi(1,1))
                @test NormalizedPolynomialSpace(Ultraspherical(j)) ==
                    NormalizedUltraspherical(NormalizedPolynomialSpace(j))
            end
        end

        @test ApproxFunOrthogonalPolynomials.normalization(ComplexF64, Jacobi(-0.5, -0.5), 0) ≈ pi
    end

    @testset "inplace transform" begin
        @testset for T in (Float32, Float64), ET in (T, complex(T))
            v = Array{ET}(undef, 10)
            v2 = similar(v)
            @testset for a in 0:0.5:3, b in 0:0.5:3, d in ((), (0..1,))
                J = Jacobi(a, b, d...)
                Slist = (J, NormalizedPolynomialSpace(J))
                @testset for S in Slist
                    test_transform!(v, v2, S)
                end
            end
            v = Array{ET}(undef, 10, 10)
            v2 = similar(v)
            @testset for a in 0:0.5:3, b in 0:0.5:3, d in ((), (0..1,))
                J = Jacobi(a, b, d...)
                Slist = (J, NormalizedPolynomialSpace(J))
                @testset for S1 in Slist, S2 in Slist
                    S = S1 ⊗ S2
                    test_transform!(v, v2, S)
                end
                @testset for S1 in Slist
                    S = S1 ⊗ Chebyshev(d...)
                    test_transform!(v, v2, S)
                    S = S1 ⊗ Chebyshev()
                    test_transform!(v, v2, S)
                end
                @testset for S2 in Slist
                    S = Chebyshev(d...) ⊗ S2
                    test_transform!(v, v2, S)
                    S = Chebyshev() ⊗ S2
                    test_transform!(v, v2, S)
                end
            end
        end
    end

    @testset "Derivative" begin
        D = @inferred Derivative(Jacobi(0.,1.,Segment(1.,0.)))
        if VERSION >= v"1.8"
            D2 = @inferred (() -> Derivative(Jacobi(0.,1.,Segment(1.,0.)), 1))()
            @test D2 == D
        end
        @time testbandedoperator(D)
        # only one band should be populated
        @test bandwidths(D, 1) == -bandwidths(D, 2)

        @test !isdiag(@inferred Derivative(Legendre()))
        T1 = typeof(Derivative(Legendre()))
        T2 = typeof(Derivative(Legendre(), 2))
        @test !isdiag(@inferred Union{T1,T2} Derivative(Legendre()))
        @test !isdiag(@inferred Derivative(NormalizedLegendre()))
        @test !isdiag(@inferred Derivative(NormalizedLegendre(), 2))

        @testset for d in [-1..1, 0..1]
            f = Fun(x->x^2, Chebyshev(d))
            C = space(f)
            for J = (Jacobi(-0.5, -0.5, d), Legendre(d))
                g = (Derivative(J) * Conversion(C, J)) * f
                h = Derivative(C) * f
                @test g ≈ h

                g = (Derivative(C) * Conversion(J, C)) * f
                h = Derivative(J) * f
                @test g ≈ h
            end
        end
        @testset for S1 in (Jacobi(0,0), Jacobi(0,0,1..2),
                            Jacobi(2,2,1..2), Jacobi(0.5,2.5,1..2)),
                S in (S1, NormalizedPolynomialSpace(S1))
            f = Fun(x->x^3 + 4x^2 + 2x + 6, S)
            @test Derivative(S) * f ≈ Fun(x->3x^2 + 8x + 2, S)
            @test Derivative(S)^2 * f ≈ Fun(x->6x+8, S)
            @test Derivative(S)^3 * f ≈ Fun(x->6, S)
            @test Derivative(S)^4 * f ≈ zeros(S)
        end

        @test (@inferred (n -> domainspace(Derivative(Jacobi(n,n), 2)))(1)) == Jacobi(1,1)
        @test (@inferred (n -> rangespace(Derivative(Jacobi(n,n), 2)))(1)) == Jacobi(3,3)

        D = Derivative(NormalizedLegendre(), 2)
        @test (@inferred rangespace(D)) == Jacobi(2,2)
    end

    @testset "identity Fun for interval domains" begin
        for d in [1..2, -1..1, 10..20], s in Any[Legendre(d), Jacobi(1, 2, d), Jacobi(1.2, 2.3, d)]
            f = Fun(identity, s)
            g = Fun(x->x, s)
            @test coefficients(f) ≈ coefficients(g)
        end
        f = Fun(identity, Legendre(-1..1))
        g = Fun(identity, Legendre())
        @test coefficients(f) ≈ coefficients(g)
        @test f(0.2) ≈ g(0.2) ≈ 0.2
    end

    @testset "Jacobi multiplication" begin
        x=Fun(identity,Jacobi(0.,0.))
        f=Fun(exp,Jacobi(0.,0.))

        @test (x*f)(.1) ≈ .1exp(.1)

        x=Fun(identity,Jacobi(0.,0., 1..2))
        f=Fun(exp,Jacobi(0.,0., 1..2))

        @test (x*f)(1.1) ≈ 1.1exp(1.1)

        x=Fun(identity,Jacobi(12.324,0.123))
        f=Fun(exp,Jacobi(0.,0.))

        @test (x*f)(.1) ≈ .1exp(.1)


        x=Fun(identity,Jacobi(12.324,0.123))
        f=Fun(exp,Jacobi(0.590,0.213))

        @test (x*f)(.1) ≈ .1exp(.1)

        g=Fun(cos,Jacobi(12.324,0.123))
        f=Fun(exp,Jacobi(0.590,0.213))

        @test (g*f)(.1) ≈ cos(.1)*exp(.1)

        @testset "Mutliplication in a normalized space" begin
            L = Jacobi(static(0), static(0))
            M = @inferred Multiplication(Fun(L), NormalizedPolynomialSpace(L))
            @test M * Fun(NormalizedLegendre()) ≈ Fun(x->x^2, NormalizedLegendre())
        end

        M1 = @inferred Multiplication(Fun(Legendre()), NormalizedLegendre())
        @test (@inferred rangespace(M1)) == NormalizedLegendre()
        @test M1 * Fun(x->x^4, NormalizedLegendre()) ≈ Fun(x->x^5, NormalizedLegendre())
        M2 = @inferred Multiplication(Fun(NormalizedLegendre()), Legendre())
        @test M2 * Fun(x->x^4, Legendre()) ≈ Fun(x->x^5, Legendre())
        M3 = @inferred Multiplication(Fun(NormalizedLegendre()), NormalizedLegendre())
        @test M3 * Fun(x->x^4, NormalizedLegendre()) ≈ Fun(x->x^5, NormalizedLegendre())
    end

    @testset "Jacobi integrate and sum" begin
        testtransforms(Legendre(0..2))
        @test sum(Fun(exp,Legendre(0..2))) ≈ sum(Fun(exp,0..2))

        a=Arc(0.,.1,0.,π/2)
        g=Fun(exp,Legendre(a))

        @test sum(g) ≈ sum(Fun(exp,a))
    end
    @testset "implementation of conversion between Chebyshev and Jacobi spaces using FastTransforms" begin
        f = Fun(x->cospi(1000x))
        g = Fun(f,Legendre())
        h = Fun(g,Chebyshev())
        @test norm(coefficients(f) - coefficients(h), Inf) < 1000eps()
        @time j = Fun(h,Legendre())
        @test norm(coefficients(g) - coefficients(j), Inf) < 10000eps()
    end

    @testset "conversion for non-compatible paramters" begin
        S=Jacobi(1.2,0.1)
        x=Fun()

        p=(S,k)->Fun(S,[zeros(k);1.])
        n=1;
        @test norm(x*p(S,n-1)-(ApproxFunOrthogonalPolynomials.recα(Float64,S,n)*p(S,n-1) + ApproxFunOrthogonalPolynomials.recβ(Float64,S,n)*p(S,n))) < 10eps()
    end

    @testset "Line sum for legendre" begin
        x = Fun(Legendre())
        @test sum(x+1) ≈ linesum(x+1)
        x=Fun(Legendre(Segment(2,1)))
        @test sum(x+1) ≈ -linesum(x+1)

        x=Fun(Segment(1,1+im))
        @test sum(x+1) ≈ im*linesum(x+1)

        x=Fun(Legendre(Segment(1,1+im)))
        @test sum(x+1) ≈ im*linesum(x+1)

        x=Fun(Legendre(Segment(im,1)))
        @test sum(x+1) ≈ (1-im)/sqrt(2)*linesum(x+1)
    end

    @testset "vector valued case" begin
        f=Fun((x,y)->real(exp(x+im*y)), Legendre(Segment(SVector(0.,0.),SVector(1.,1.))))
        @test f(0.1,0.1) ≈ real(exp(0.1+0.1im))
    end

    @testset "integer, float mixed" begin
        C=Conversion(Legendre(),Jacobi(1,0))
        testbandedoperator(C)
    end

    @testset "Addition of piecewise Legendre bug" begin
        f = Fun(exp,Legendre())
        f1 = Fun(exp,Legendre(-1..0))
        f2 = Fun(exp,Legendre(0..1))
        fp = f1+f2
        @test space(fp) isa PiecewiseSpace
        @test fp(0.1) ≈ exp(0.1)
        @test fp(0.) ≈ exp(0.)
        @test fp(-0.1) ≈ exp(-0.1)
    end

    @testset "Jacobi–Chebyshev conversion" begin
        a,b = (Jacobi(-0.5,-0.5), Legendre())
        @test maxspace(a,b) == NoSpace()
        @test union(a,b) == a
        @test !hasconversion(a,b)

        a,b = (Chebyshev(), Legendre())
        @test maxspace(a,b) == NoSpace()
        @test union(a,b) == Jacobi(-0.5,-0.5)
        @test !hasconversion(a,b)

        @testset for a in Any[Chebyshev(0..1), Ultraspherical(1, 0..1)]
            b = Jacobi(a)
            c = union(a, b)
            @test c == a

            b = ApproxFunBase.setdomain(Jacobi(a), 1..2)
            c = union(a, b)
            d = domain(c)
            @test 0 in d
            @test 1 in d
            @test 2 in d

            b = Legendre(1..2)
            c = union(a, b)
            d = domain(c)
            @test 0 in d
            @test 1 in d
            @test 2 in d

            b = Legendre(domain(a))
            c = union(a, b)
            d = domain(c)
            @test d == domain(a)
        end
    end

    @testset "Fun coefficients conversion" begin
        for d in Any[(), (0..1,)]
            sp1 = Any[Chebyshev(d...),
                Ultraspherical(1,d...), Ultraspherical(2,d...), Ultraspherical(3.5,d...),
                Jacobi(1,1,d...), Jacobi(1,2,d...)]
            sp2 = Any[Jacobi(1,1,d...), Jacobi(1,2,d...),
                Ultraspherical(1,d...), Ultraspherical(2,d...),
                Chebyshev(d...)]
            for _S1 in sp1, _S2 in sp2,
                S1 in (_S1, NormalizedPolynomialSpace(_S1)),
                S2 in (_S2, NormalizedPolynomialSpace(_S2))

                f = Fun(x->x^4, S1)
                g = Fun(f, S2)
                h = Fun(g, S1)
                @test f ≈ h
                @test coefficients(f) ≈ coefficients(h)
            end
        end
    end

    @testset "Reverse orientation" begin
        S = Jacobi(0.1,0.2)

        @test_throws ArgumentError Conversion(S, Jacobi(1.1,1.2,0..1))

        f = Fun(S, randn(10))
        @test f(0.1) ≈ (ReverseOrientation(S)*f)(0.1) ≈ reverseorientation(f)(0.1)
        @test rangespace(ReverseOrientation(S)) == space(reverseorientation(f)) ==
                    Jacobi(0.2,0.1,Segment(1,-1))

        R = Conversion(S, reverseorientation(S))
        @test (R*f)(0.1) ≈ f(0.1) ≈ reverseorientation(f)(0.1)

        S = Legendre()
        f = Fun(S, randn(10))
        @test f(0.1) ≈ (ReverseOrientation(S)*f)(0.1) ≈ reverseorientation(f)(0.1)
        @test rangespace(ReverseOrientation(S)) == space(reverseorientation(f)) ==
                    Legendre(Segment(1,-1))

        R = Conversion(S, reverseorientation(S))
        @test rangespace(R) == reverseorientation(S) ==
            space(reverseorientation(f))
        @test f(0.1) ≈ (R*f)(0.1) ≈ reverseorientation(f)(0.1)
    end

    @testset "Full Jacobi" begin
        sp = Jacobi(.5,2.124)
        f = Fun(exp,sp)
        sp2 = Jacobi(1.5,2.124)
        M = Multiplication(f,sp2)
        @time testbandedoperator(M)


        ## Legendre conversions
        testspace(Ultraspherical(1); haslineintegral=false)
        testspace(Ultraspherical(2); haslineintegral=false)
        # minpoints is a tempory fix a bug
        @time testspace(Ultraspherical(1//2); haslineintegral=false, minpoints=2)
        @test norm(Fun(exp,Ultraspherical(1//2))-Fun(exp,Jacobi(0,0))) < 100eps()

        C=Conversion(Jacobi(0,0),Chebyshev())
        @time testbandedbelowoperator(C)
        @test norm(C*Fun(exp,Jacobi(0,0))  - Fun(exp)) < 100eps()


        C=Conversion(Ultraspherical(1//2),Chebyshev())
        @time testbandedbelowoperator(C)
        @test norm(C*Fun(exp,Ultraspherical(1//2))  - Fun(exp)) < 100eps()



        C=Conversion(Chebyshev(),Ultraspherical(1//2))
        @time testbandedbelowoperator(C)
        @test norm(C*Fun(exp)-Fun(exp,Legendre())) < 100eps()


        C=Conversion(Chebyshev(),Jacobi(0,0))
        @time testbandedbelowoperator(C)
        @test norm(C*Fun(exp)  - Fun(exp,Jacobi(0,0))) < 100eps()


        C=Conversion(Chebyshev(),Jacobi(1,1))
        @time testbandedbelowoperator(C)
        @test norm(C*Fun(exp) - Fun(exp,Jacobi(1,1))) < 100eps()


        C=Conversion(Ultraspherical(1//2),Ultraspherical(1))
        @time testbandedbelowoperator(C)

        λ1 = ApproxFunOrthogonalPolynomials.order(domainspace(C))
        λ2 = ApproxFunOrthogonalPolynomials.order(rangespace(C))

        # test against version that doesn't use lgamma
        Cex = Float64[(if j ≥ k && iseven(k-j)
                gamma(λ2)*(k-1+λ2)/(gamma(λ1)*gamma(λ1-λ2))*
                    (gamma((j-k)/2+λ1-λ2)/gamma((j-k)/2+1))*
                    (gamma((k+j-2)/2+λ1)/gamma((k+j-2)/2+λ2+1))
            else
                0.0
            end) for k=1:20,j=1:20]

        @test norm(Cex - C[1:20,1:20]) < 100eps()

        @test norm(C*Fun(exp,Ultraspherical(1//2))-Fun(exp,Ultraspherical(1))) < 100eps()

        C=Conversion(Jacobi(0,0),Ultraspherical(1))
        testbandedbelowoperator(C)
        @test norm(C*Fun(exp,Jacobi(0,0))-Fun(exp,Ultraspherical(1))) < 100eps()


        C=Conversion(Ultraspherical(1),Jacobi(0,0))
        testbandedbelowoperator(C)
        @test norm(C*Fun(exp,Ultraspherical(1))-Fun(exp,Jacobi(0,0))) < 100eps()
    end

    @testset "Normalized space" begin
        @test NormalizedJacobi(1,1) isa NormalizedJacobi
        for f in Any[x -> 3x^3 + 5x^2 + 2, x->x, identity]
            for dt in Any[(), (0..1,)],
                    S in Any[Jacobi(1,1,dt...),
                             Jacobi(0.5,1.5,dt...),
                             Legendre(dt...), ]

                NS = NormalizedPolynomialSpace(S)
                fS = Fun(f, S)
                fNS = Fun(f, NS)
                @test space(fNS) == NS
                d = domain(fS)
                r = range(leftendpoint(d), rightendpoint(d), length=10)
                for x in r
                    @test fS(x) ≈ fNS(x) rtol=1e-7 atol=1e-14
                end
            end
        end

        @testset "Multiplication" begin
            xJ = Fun(NormalizedJacobi(1,1))
            xC = Fun()
            xNC = Fun(NormalizedChebyshev())
            @test (Multiplication(xC) * xJ)(0.4) ≈ (0.4)^2
            @test (Multiplication(xNC) * xJ)(0.4) ≈ (0.4)^2
            @test ApproxFunBase.isbanded(Multiplication(xC, NormalizedLegendre()))
            @test ApproxFunBase.isbanded(Multiplication(xNC, NormalizedLegendre()))
            @test ApproxFunBase.isbanded(Multiplication(xC, NormalizedJacobi(1,1)))
            @test ApproxFunBase.isbanded(Multiplication(xNC, NormalizedJacobi(1,1)))
        end

        @testset "space promotion" begin
            @test space(1 + Fun(NormalizedLegendre())) == NormalizedLegendre()
            @test space(1 + Fun(NormalizedJacobi(1,1,0..1))) == NormalizedJacobi(1,1,0..1)
        end
    end

    @testset "casting bug ApproxFun.jl#770" begin
        f = Fun((t,x)-> im*exp(t)*sinpi(x), Legendre()^2)
        @test f(0.1, 0.2) ≈ im*exp(0.1)*sinpi(0.2)
    end

    @testset "Evaluation" begin
        c = [i^2 for i in 1:4]
        @testset for d in Any[0..1, ChebyshevInterval()],
                (a,b) in Any[(1,1), (2,3), (2.5, 0.4)]
            @testset  for _sp in Any[Jacobi(a,b), Jacobi(a,b,d)],
                    sp in Any[_sp, NormalizedPolynomialSpace(_sp)]
                d = domain(sp)
                f = Fun(sp, c)
                for ep in [leftendpoint, rightendpoint],
                        ev in [ApproxFunBase.ConcreteEvaluation, Evaluation]
                    E = @inferred ev(sp, ep, 0)
                    @test E[2:4] ≈ E[1:4][2:end]
                    @test E[1:2:5] ≈ E[1:5][1:2:5]
                    @test E[2:2:6] ≈ E[1:6][2:2:6]
                    @test Number(E * f) ≈ f(ep(d))
                    E2 = @inferred ev(sp, ep(d), 0)
                    @test Number(E2 * f) ≈ f(ep(d))

                    D = @inferred ev(sp, ep, 1)
                    @test D[2:4] ≈ D[1:4][2:end]
                    @test D[1:2:5] ≈ D[1:5][1:2:5]
                    @test D[2:2:6] ≈ D[1:6][2:2:6]
                    @test Number(D * f) ≈ f'(ep(d))
                    Dp = @inferred ev(sp, ep(d), 1)
                    @test Number(Dp * f) ≈ f'(ep(d))

                    D2 = @inferred ev(sp, ep, 2)
                    @test D2[2:4] ≈ D2[1:4][2:end]
                    @test D2[1:2:5] ≈ D2[1:5][1:2:5]
                    @test D2[2:2:6] ≈ D2[1:6][2:2:6]
                    @test Number(D2 * f) ≈ f''(ep(d))
                    D2p = @inferred ev(sp, ep(d), 2)
                    @test Number(D2p * f) ≈ f''(ep(d))

                    D3 = @inferred ev(sp, ep, 3)
                    @test D3[2:4] ≈ D3[1:4][2:end]
                    @test D3[1:2:5] ≈ D3[1:5][1:2:5]
                    @test D3[2:2:6] ≈ D3[1:6][2:2:6]
                    @test Number(D3 * f) ≈ f'''(ep(d))
                    D3p = @inferred ev(sp, ep(d), 3)
                    @test Number(D3p * f) ≈ f'''(ep(d))
                end
                @test ldirichlet(sp) * f ≈ f(leftendpoint(domain(f)))
                @test rdirichlet(sp) * f ≈ f(rightendpoint(domain(f)))
                @test lneumann(sp) * f ≈ f'(leftendpoint(domain(f)))
                @test rneumann(sp) * f ≈ f'(rightendpoint(domain(f)))
                @inferred bvp(sp)[1]
            end
        end
    end

    @testset "Integral" begin
        @testset for sp in (Legendre(), Jacobi(1,1))
            Ij = Integral(sp, 1)
            @test !isdiag(Ij)
            f = Fun(sp)
            g = Ij * f
            g = Fun(g, sp)
            g = g - coefficients(g)[1]
            gexp = Fun(x->x^2/2, sp)
            gexp = gexp - coefficients(gexp)[1]
            @test g ≈ gexp
        end

        @testset for n in 3:6, d in ((), (0..1,)),
                sp in (Legendre(d...), Jacobi(1,1, d...), Jacobi(1,3, d...),
                        Jacobi(0,0.5, d...), Jacobi(0.5,0.5, d...), Jacobi(0.5, 1.5, d...))
            f = Fun(sp, Float64[zeros(n); 2])
            @test Integral(1) * (Derivative(1) * f) ≈ f
            @test Integral(2) * (Derivative(2) * f) ≈ f
            @test Integral(3) * (Derivative(3) * f) ≈ f
            @test Derivative(1) * (Integral(1) * f) ≈ f
            @test Derivative(2) * (Integral(2) * f) ≈ f
            @test Derivative(3) * (Integral(3) * f) ≈ f
        end
    end

    @testset "type inference" begin
        x = @inferred ApproxFunBase.maxspace(Jacobi(1,1), Jacobi(2,2))
        @test x == Jacobi(2,2)

        x = @inferred ApproxFunBase.union_rule(Chebyshev(), Jacobi(1,1))
        @test x == Jacobi(-0.5, -0.5)

        x = @inferred ApproxFunBase.conversion_rule(Jacobi(1,1), Jacobi(2,2))
        @test x == Jacobi(1,1)
    end

    @testset "Tensor space conversions" begin
        @test ApproxFunBase.hasconversion(Chebyshev()*Legendre(), Chebyshev()*Legendre())
        @test ApproxFunBase.hasconversion(Chebyshev()*Legendre(), Chebyshev()*NormalizedLegendre())
        @test ApproxFunBase.hasconversion(Chebyshev()*Legendre(), NormalizedChebyshev()*Legendre())
        @test ApproxFunBase.hasconversion(Chebyshev()*NormalizedLegendre(), Chebyshev()*Legendre())
        @test ApproxFunBase.hasconversion(NormalizedChebyshev()*Legendre(), Chebyshev()*Legendre())
        @test ApproxFunBase.hasconversion(NormalizedChebyshev()*NormalizedLegendre(), Chebyshev()*Legendre())
        @test ApproxFunBase.hasconversion(Chebyshev()*Legendre(), NormalizedChebyshev()*NormalizedLegendre())
    end
end
