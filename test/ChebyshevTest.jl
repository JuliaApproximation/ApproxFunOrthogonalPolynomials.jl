using ApproxFunOrthogonalPolynomials
using ApproxFunBase
using LinearAlgebra
using Test
using ApproxFunBase: recA, recB, recC, transform!, itransform!
using ApproxFunBaseTest: testspace
using ApproxFunOrthogonalPolynomials: forwardrecurrence

@verbose @testset "Chebyshev" begin
    @testset "Forward recurrence" begin
        @test forwardrecurrence(Float64,Chebyshev(),0:9,1.0) == ones(10)
    end

    @testset "ChebyshevInterval" begin
        @test @inferred(Fun(x->2,10))(.1) ≈ 2
        @test Fun(x->2)(.1) ≈ 2
        @test @inferred(Fun(Chebyshev, Float64[])).([0.,1.]) ≈ [0.,0.]
        @test Fun(Chebyshev, [])(0.) ≈ 0.
        @test @inferred(Fun(x->4, Chebyshev(), 1)).coefficients == [4.0]
        @test Fun(x->4).coefficients == [4.0]
        @test @inferred(Fun(4)).coefficients == [4.0]

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

    @testset "points" begin
        p = points(Chebyshev(-1..1), 4)
        @test points(Chebyshev(), 4) ≈ p
        @test contains(summary(p), "ShiftedChebyshevGrid{Float64}")
    end

    @testset "inference in Space(::Interval)" begin
        S = @inferred Space(0..1)
        f = Fun(S)
        g = Fun(domain(S))
        @test f(0.2) ≈ g(0.2) ≈ 0.2

        S = Space(0.0..1.0)
        f = Fun(S)
        g = Fun(domain(S))
        @test f(0.2) ≈ g(0.2) ≈ 0.2
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

        @test @inferred(maximum(abs,ef.(r)-exp.(r))) < 200eps()
        @test @inferred(maximum(abs,ecf.(r).-cos.(r).*exp.(r))) < 200eps()

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

        ecf = Fun(x->cos(x)*exp(x),1..2)
        eocf = Fun(x->cos(x)/exp(x),1..2)

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

        f = Fun(Chebyshev(), [1,2,3])
        @test sort(roots(f)) ≈ 1/6*[-1-√13, -1+√13]
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
        # precision bug on v1.6 on Windows
        windowsv16 = v"1.6" <= VERSION < v"1.7" && Sys.iswindows()
        atol_test = windowsv16 ? 1E-11 : 1E-12
        @test ≈(f(.1),sin(.1^2);atol = atol_test)
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
        @testset for T in (Float32, Float64), ET in (T, complex(T))
            v = Array{ET}(undef, 6)
            v2 = similar(v)
            M = Array{ET}(undef, 6, 6)
            M2 = similar(M)
            A = Array{ET}(undef, 6, 6, 6)
            A2 = similar(A)
            @testset for d in ((), (0..1,))
                C = Chebyshev(d...)
                Slist = (C, NormalizedPolynomialSpace(C))
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
        @test NormalizedChebyshev() isa NormalizedChebyshev
        for f in (x -> 3x^3 + 5x^2 + 2, x->x, identity)
            for dt in ((), (0..1,))
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
            D1 = if VERSION >= v"1.8"
                @inferred Derivative(s1)
            else
                Derivative(s1)
            end
            D2 = if VERSION >= v"1.8"
                @inferred Derivative(s2)
            else
                Derivative(s2)
            end
            f = x -> 3x^2 + 5x
            f1 = Fun(f, s1)
            f2 = Fun(f, s2)
            @test f1 == f2
            @test D1 * f1 == D2 * f2

            @test (@inferred (D1 -> eltype(D1 + D1))(D1)) == eltype(D1)
        end

        @testset "space promotion" begin
            s = NormalizedChebyshev()
            f = (Derivative() + Fun(s)) * Fun(s)
            g = ones(s) + Fun(s)^2
            @test f ≈ g

            @test space(1 + Fun(NormalizedChebyshev())) == NormalizedChebyshev()
            @test space(1 + Fun(NormalizedChebyshev(0..1))) == NormalizedChebyshev(0..1)
        end
    end

    @testset "Operator exponentiation" begin
        for M in (Multiplication(Fun(), Chebyshev()), Multiplication(Fun()))
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

    @testset "Constant propagation" begin
        @testset "Dirichlet" begin
            D = if VERSION >= v"1.8"
                @inferred (r -> Dirichlet(r))(Chebyshev(0..1))
            else
                Dirichlet(Chebyshev(0..1))
            end
            # Dirichlet constraints don't depend on the domain
            D2 = Dirichlet(Chebyshev())
            @test Matrix(D[:, 1:4]) == Matrix(D2[:, 1:4])

            D = @inferred (() -> Dirichlet(Chebyshev(), 2))()
            D2 = @inferred (() -> Dirichlet(Chebyshev(-1..1), 2))()
            @test Matrix(D[:, 1:4]) == Matrix(D2[:, 1:4])
        end

        @testset "Multiplication" begin
            f = () -> Fun(0..1) * Derivative(Chebyshev(0..1))
            A = VERSION >= v"1.8" ? (@inferred f()) : f()
            @test (A * Fun(0..1))(0.5) ≈ 0.5

            f = () -> Fun() * Derivative(Chebyshev())
            A = VERSION >= v"1.8" ? (@inferred f()) : f()
            @test (A * Fun())(0.5) ≈ 0.5
        end
    end

    @testset "Inference in PlusOperator" begin
        f = () -> Derivative(Chebyshev(0..1)) + Derivative(Chebyshev(0..1))
        A = VERSION >= v"1.8" ? (@inferred f()) : f()
        @test (A * Fun(0..1))(0.5) ≈ 2.0

        f = () -> Derivative(Chebyshev()) + Derivative(Chebyshev())
        A = VERSION >= v"1.8" ? (@inferred f()) : f()
        @test (A * Fun())(0.5) ≈ 2.0
    end

    @testset "Evaluation" begin
        c = [i^2 for i in 1:4]
        @testset for d in Any[0..1, ChebyshevInterval()]
            @testset for _sp in Any[Chebyshev(), Chebyshev(d)],
                    sp in Any[_sp, NormalizedPolynomialSpace(_sp)]
                d = domain(sp)
                f = Fun(sp, c)
                @testset for ep in [leftendpoint, rightendpoint],
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
            end
        end

        @testset "ChebyshevDirichlet" begin
            function Evaluation2(sp::ChebyshevDirichlet,x,ord)
                S=Space(domain(sp))
                ApproxFunBase.EvaluationWrapper(sp,x,ord,Evaluation(S,x,ord)*Conversion(sp,S))
            end
            sp = ChebyshevDirichlet()
            d = domain(sp)
            @testset for ep in [leftendpoint, rightendpoint]
                A = @inferred ApproxFunBase.ConcreteEvaluation(sp, ep, 1)
                B = @inferred Evaluation2(sp, ep, 1)
                @test A[1:10] ≈ B[1:10]
            end
        end
    end

    @testset "inference in SumSpace" begin
        s = Chebyshev() + Chebyshev(0..1)
        blk = @inferred ApproxFunBase.block(s, 2)
        @test Int(blk) == 1
    end
end
