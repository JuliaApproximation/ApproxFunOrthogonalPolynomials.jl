using ApproxFunOrthogonalPolynomials, ApproxFunBase, Test, SpecialFunctions, LinearAlgebra
import ApproxFunBase: testbandedbelowoperator, testbandedoperator, testspace, testtransforms, Vec,
                    maxspace, NoSpace, hasconversion, testfunctional,
                    reverseorientation, ReverseOrientation, transform!, itransform!
import ApproxFunOrthogonalPolynomials: jacobip

@testset "Jacobi" begin
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
    end

    @testset "Conversion" begin
        testtransforms(Jacobi(-0.5,-0.5))
        @test norm(Fun(Fun(exp),Jacobi(-.5,-.5))-Fun(exp,Jacobi(-.5,-.5))) < 300eps()

        @testset "inplace transform" begin
            for T in [Float32, Float64, BigFloat]
                for v in Any[rand(T, 10), rand(complex(T), 10)]
                    v2 = copy(v)
                    for a in 0:0.5:3, b in 0:0.5:3
                        J = Jacobi(a, b)
                        transform!(J, v)
                        @test transform(J, v2) ≈ v
                        itransform!(J, v)
                        @test v2 ≈ v
                    end
                end
            end
        end

        @testset for d in [-1..1, 0..1]
            f = Fun(x->x^2, Chebyshev(d))
            C = space(f)
            for J1 = Any[Jacobi(-0.5, -0.5, d), Legendre(d),
                            Jacobi(0.5, 0.5, d), Jacobi(2.5, 1.5, d)]
                for J in [J1, NormalizedPolynomialSpace(J1)]
                    g = Fun(f, J)
                    if !any(isnan, coefficients(g))
                        @test Conversion(C, J) * f ≈ g
                    end
                end
            end
        end
    end

    @testset "Derivative" begin
        D=Derivative(Jacobi(0.,1.,Segment(1.,0.)))
        @time testbandedoperator(D)
        # only one band should be populated
        @test bandwidths(D, 1) == -bandwidths(D, 2)

        @testset for d in [-1..1, 0..1]
            f = Fun(x->x^2, Chebyshev(d))
            C = space(f)
            for J = Any[Jacobi(-0.5, -0.5, d), Legendre(d)]
                g = (Derivative(J) * Conversion(C, J)) * f
                h = Derivative(C) * f
                @test g ≈ h

                g = (Derivative(C) * Conversion(J, C)) * f
                h = Derivative(J) * f
                @test g ≈ h
            end
        end
        @testset for S1 in Any[Jacobi(0,0),
            Jacobi(0,0,1..2), Jacobi(2,2,1..2), Jacobi(0.5,2.5,1..2)],
                S in Any[S1, NormalizedPolynomialSpace(S1)]
            f = Fun(x->x^3 + 4x^2 + 2x + 6, S)
            @test Derivative(S) * f ≈ Fun(x->3x^2 + 8x + 2, S)
            @test Derivative(S)^2 * f ≈ Fun(x->6x+8, S)
            @test Derivative(S)^3 * f ≈ Fun(x->6, S)
            @test Derivative(S)^4 * f ≈ zeros(S)
        end
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
        @test norm(f.coefficients-h.coefficients,Inf) < 1000eps()
        @time h = Fun(h,Legendre())
        @test norm(g.coefficients-h.coefficients,Inf) < 10000eps()
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
        f=Fun((x,y)->real(exp(x+im*y)), Legendre(Segment(Vec(0.,0.),Vec(1.,1.))))
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
    end
end
