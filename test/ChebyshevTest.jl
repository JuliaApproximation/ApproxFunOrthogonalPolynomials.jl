using ApproxFunOrthogonalPolynomials, ApproxFunBase, LinearAlgebra, Test
import ApproxFunBase: testspace, recA, recB, recC, transform!, itransform!
import ApproxFunOrthogonalPolynomials: forwardrecurrence

@testset "Chebyshev" begin
    @testset "Forward recurrence" begin
        @test forwardrecurrence(Float64,Chebyshev(),0:9,1.0) == ones(10)
    end

    @testset "ChebyshevInterval" begin
        @test @inferred Fun(x->2,10)(.1) ≈ 2
        @test @inferred Fun(x->2)(.1) ≈ 2
        @test @inferred Fun(Chebyshev,Float64[]).([0.,1.]) ≈ [0.,0.]
        @test @inferred Fun(Chebyshev,[])(0.) ≈ 0.
        @test @inferred Fun(x->4,Chebyshev(),1).coefficients == [4.0]
        @test @inferred Fun(x->4).coefficients == [4.0]
        @test @inferred Fun(4).coefficients == [4.0]


        @test @inferred Fun(x->4).coefficients == [4.0]
        @test @inferred Fun(4).coefficients == [4.0]

        f = @inferred Fun(ChebyshevInterval(), [1])
        @test f(0.1) == 1

        ef = if VERSION >= v"1.8"
            @inferred Fun(exp)
        else
            Fun(exp)
        end
        @test @inferred ef(0.1) ≈ exp(0.1)
        @test @inferred ef(.5) ≈ exp(.5)

        for d in (ChebyshevInterval(),Interval(1.,2.),Segment(1.0+im,2.0+2im))
            testspace(Chebyshev(d))
        end

        y = @inferred Fun() + Fun(ChebyshevInterval{BigFloat}())
        @test y == 2Fun()
    end

    @testset "Algebra" begin
        ef = @inferred Fun(exp,ChebyshevInterval())
        @test ef == @inferred -(-ef)
        @test ef == @inferred (@inferred ef-1) + 1

        if VERSION >= v"1.8"
            @test (@inferred ef / 3) == @inferred (3 \ ef)
        else
            @test ef / 3 == 3 \ ef
        end

        cf = VERSION >= v"1.8" ? @inferred(Fun(cos)) : Fun(cos)

        ecf = VERSION >= v"1.8" ? @inferred(Fun(x->cos(x)*exp(x))) : Fun(x->cos(x)*exp(x))
        eocf = VERSION >= v"1.8" ? @inferred(Fun(x->cos(x)/exp(x))) : Fun(x->cos(x)/exp(x))

        @test ef(.5) ≈ exp(.5)
        @test ecf(.123456) ≈ cos(.123456).*exp(.123456)

        r=2 .* rand(100) .- 1

        @test @inferred maximum(abs,ef.(r)-exp.(r))<200eps()
        @test @inferred maximum(abs,ecf.(r).-cos.(r).*exp.(r))<200eps()

        @test (@inferred (cf .* ef)(0.1)) ≈ ecf(0.1) ≈ cos(0.1)*exp(0.1)
        @test (@inferred domain(cf.*ef)) ≈ domain(ecf)
        @test (@inferred domain(cf.*ef)) == domain(ecf)

        @test norm(@inferred(ecf-cf.*ef).coefficients)<200eps()
        @test maximum(abs,@inferred((eocf-cf./ef)).coefficients)<1000eps()
        @test norm(((ef/3).*(3/ef)-1).coefficients)<1000eps()
    end

    @testset "Diff and cumsum" begin
        ef = Fun(exp)
        cf = Fun(cos)
        @test norm(@inferred(ef - ef').coefficients)<10E-11

        @test norm(@inferred(ef - cumsum(ef)').coefficients) < 20eps()
        @test norm(@inferred(cf - cumsum(cf)').coefficients) < 20eps()

        @test @inferred(sum(ef))  ≈ 2.3504023872876028
        @test @inferred(norm(ef))  ≈ 1.90443178083307
    end

    @testset "other domains" begin
        ef = Fun(exp,1..2)
        cf = Fun(cos,1..2)

        ecf = Fun(x->cos(x).*exp(x),1..2)
        eocf = Fun(x->cos(x)./exp(x),1..2)

        x=1.5
        @test ef(x) ≈ exp(x)

        r=rand(100) .+ 1
        @test maximum(abs,ef.(r)-exp.(r))<400eps()
        @test maximum(abs,ecf.(r).-cos.(r).*exp.(r))<100eps()

        @testset "setdomain" begin
            @test setdomain(NormalizedChebyshev(0..1), 1..2) == NormalizedChebyshev(1..2)
        end
    end

    @testset "Other interval" begin
        ef = Fun(exp,1..2)
        cf = Fun(cos,1..2)

        ecf = Fun(x->cos(x).*exp(x),1..2)
        eocf = Fun(x->cos(x)./exp(x),1..2)

        r=rand(100) .+ 1
        x=1.5

        @test ef(x) ≈ exp(x)

        @test maximum(abs,ef.(r)-exp.(r))<400eps()
        @test maximum(abs,ecf.(r).-cos.(r).*exp.(r))<100eps()
        @test norm((ecf-cf.*ef).coefficients)<500eps()
        @test maximum(abs,(eocf-cf./ef).coefficients)<1000eps()
        @test norm(((ef/3).*(3/ef)-1).coefficients)<1000eps()

        ## Diff and cumsum
        @test norm((ef - ef').coefficients)<10E-11
        @test norm((ef - cumsum(ef)').coefficients) < 10eps()
        @test norm((cf - cumsum(cf)').coefficients) < 10eps()

        @test sum(ef) ≈ 4.670774270471604
        @test norm(ef) ≈ 4.858451087240335
    end

    @testset "Roots" begin
        f=Fun(x->sin(10(x-.1)))
        @test norm(f.(roots(f)))< 1000eps()

        @test_throws ArgumentError roots(Fun(zero))
        @test_throws ArgumentError roots(Fun(Chebyshev(),Float64[]))
    end

    @testset "Aliasing" begin
        f=Fun(x->cos(50acos(x)))
        @test norm(f.coefficients-Matrix(I,ncoefficients(f),ncoefficients(f))[:,51])<100eps()
    end

    @testset "Integer values" begin
        @test Fun(x->2,10)(.1) ≈ 2
        @test Fun(x->2)(.1) ≈ 2

        @test Fun(Chebyshev,Float64[]).([0.,1.]) ≈ [0.,0.]
        @test Fun(Chebyshev,[])(0.) ≈ 0.
    end

    @testset "large intervals #121" begin
        x = Fun(identity,0..10)
        f = sin(x^2)
        g = cos(x)
        @test exp(x)(0.1) ≈ exp(0.1)
        @test f(.1) ≈ sin(.1^2)

        x = Fun(identity,0..100)
        f = sin(x^2)
        @test ≈(f(.1),sin(.1^2);atol=1E-12)
    end

    @testset "Reverse" begin
        f=Fun(exp)
        @test ApproxFunBase.default_Fun(f, Chebyshev(Segment(1 , -1)), ncoefficients(f))(0.1) ≈ exp(0.1)
        @test Fun(f,Chebyshev(Segment(1,-1)))(0.1) ≈ f(0.1)
    end

    @testset "minimum/maximum" begin
        x=Fun()
        @test minimum(x) == -1
        @test maximum(x) == 1
    end

    @testset "Do not overresolve #7" begin
        @test ncoefficients(Fun(x->sin(400*pi*x),-1..1)) ≤ 1400
    end

    @testset "Bug from Trogdon" begin
        δ = .03 # should be less than 0.03
        @test 0.0 ∈ Domain(1-8*sqrt(δ)..1+8*sqrt(δ))
        @test 0.00001 ∈ Domain(1-8*sqrt(δ)..1+8*sqrt(δ))

        ϕfun = Fun(x -> 1/sqrt(2*pi*δ)*exp(-abs2.(x-1)/(2*δ)), 1-8sqrt(δ)..1+8sqrt(δ))
        ϕfun(0.00001) ≈ 1/sqrt(2*pi*δ)*exp(-abs2.(0.00001-1)/(2*δ))

        iϕfun = 1-cumsum(ϕfun)
        @test iϕfun(0.00001) ≈ 1
    end

    @testset "Large scaling" begin
        w = Fun(x -> 1e5/(x*x+1), 283.72074879785936 .. 335.0101119042838)
        @test w(leftendpoint(domain(w))) ≈ 1e5/(leftendpoint(domain(w))^2+1)
    end

    @testset "supremum norm" begin
        x = Fun()
        f = 1/(1 + 25*(x^2))
        @test norm(f, Inf) ≈ 1.0
    end

    @testset "Jacobi" begin
        S=Chebyshev()
        @test S.a==-0.5
        @test S.b==-0.5
    end

    @testset "inplace transform" begin
        @testset for T in [Float32, Float64], ET in Any[T, complex(T)]
            v = Array{ET}(undef, 10)
            v2 = similar(v)
            M = Array{ET}(undef, 10, 10)
            M2 = similar(M)
            A = Array{ET}(undef, 10, 10, 10)
            A2 = similar(A)
            @testset for d in Any[(), (0..1,)]
                C = Chebyshev(d...)
                Slist = Any[C, NormalizedPolynomialSpace(C)]
                @testset for S in Slist
                    test_transform!(v, v2, S)
                end
                @testset for S1 in Slist, S2 in Slist
                    S = S1 ⊗ S2
                    test_transform!(M, M2, S)
                end
                @testset for S1 in Slist, S2 in Slist, S3 in Slist
                    S = S1 ⊗ S2 ⊗ S3
                    test_transform!(A, A2, S)
                end
            end
        end
    end

    @testset "Normalized space" begin
        for f in Any[x -> 3x^3 + 5x^2 + 2, x->x, identity]
            for dt in Any[(), (0..1,)]
                S = Chebyshev(dt...)
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

        @testset "derivative in normalized space" begin
            s1 = NormalizedChebyshev(-1..1)
            s2 = NormalizedChebyshev()
            @test s1 == s2
            D1 = Derivative(s1)
            D2 = Derivative(s2)
            f = x -> 3x^2 + 5x
            f1 = Fun(f, s1)
            f2 = Fun(f, s2)
            @test f1 == f2
            @test D1 * f1 == D2 * f2
        end

        @testset "space promotion" begin
            s = NormalizedChebyshev()
            f = (Derivative() + Fun(s)) * Fun(s)
            g = ones(s) + Fun(s)^2
            @test f ≈ g
        end
    end

    @testset "Operator exponentiation" begin
        for M in Any[Multiplication(Fun(), Chebyshev()), Multiplication(Fun())]
            N = @inferred (M -> M^0)(M)
            @test N * Fun() == Fun()
            N = @inferred (M -> M^1)(M)
            @test N == M
            N = @inferred (M -> M^2)(M)
            @test N == M*M
            @test M^3 == M * M * M
            @test M^4 == M * M * M * M
            @test M^10 == foldr(*, fill(M, 10))
        end
    end

    @testset "values for ArraySpace Fun" begin
        f = Fun(Chebyshev() ⊗ Chebyshev())
        @test f.(points(f)) == points(f)
        @test values(f) == itransform(space(f), coefficients(f))
        a = transform(space(f), values(f))
        b = coefficients(f)
        nmin = min(length(a), length(b))
        @test a[1:nmin] ≈ b[1:nmin]
    end
end
