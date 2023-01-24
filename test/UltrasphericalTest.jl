using ApproxFunOrthogonalPolynomials
using ApproxFunBase
using Test
using SpecialFunctions
using LinearAlgebra

@testset "Ultraspherical" begin
    @testset "identity fun" begin
        for d in (ChebyshevInterval(), 3..4, Segment(2, 5), Segment(1, 4im)), order in (1, 2, 0.5)
            s = Ultraspherical(order, d)
            f = Fun(s)
            xl = leftendpoint(domain(s))
            xr = rightendpoint(domain(s))
            xm = (xl + xr)/2
            @test f(xl) ≈ xl
            @test f(xr) ≈ xr
            @test f(xm) ≈ xm
        end
    end
    @testset "Conversion" begin
        # Tests whether invalid/unimplemented arguments correctly throws ArgumentError
        @test_throws ArgumentError Conversion(Ultraspherical(2), Ultraspherical(1))
        @test_throws ArgumentError Conversion(Ultraspherical(3), Ultraspherical(1.9))

        # Conversions from Chebyshev to Ultraspherical should lead to a small union of types
        Tallowed = Union{
            ApproxFunBase.ConcreteConversion{
                Chebyshev{ChebyshevInterval{Float64}, Float64},
                Ultraspherical{Int64, ChebyshevInterval{Float64}, Float64}, Float64},
            ApproxFunBase.ConversionWrapper{TimesOperator{Float64, Tuple{Int64, Int64}}, Float64}};
        @inferred Tallowed Conversion(Chebyshev(), Ultraspherical(1));
        @inferred Tallowed Conversion(Chebyshev(), Ultraspherical(2));
        # Conversions between Ultraspherical should lead to a small union of types
        Tallowed = Union{
            ApproxFunBase.ConcreteConversion{
                Ultraspherical{Int64, ChebyshevInterval{Float64}, Float64},
                Ultraspherical{Int64, ChebyshevInterval{Float64}, Float64}, Float64},
            ApproxFunBase.ConversionWrapper{
                ConstantOperator{Float64,
                    Ultraspherical{Int64, ChebyshevInterval{Float64}, Float64}}, Float64},
                    ApproxFunBase.ConversionWrapper{TimesOperator{Float64, Tuple{Int64, Int64}}, Float64}};
        @inferred Tallowed Conversion(Ultraspherical(1), Ultraspherical(2));
    end

    @testset "Normalized space" begin
        @test NormalizedUltraspherical(1) isa NormalizedUltraspherical
        for f in (x -> 3x^3 + 5x^2 + 2, x->x, identity)
            for dt in ((), (0..1,)),
                    S in (Ultraspherical(1, dt...),
                             Ultraspherical(0.5,dt...),
                             Ultraspherical(3, dt...))

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

    @testset "inplace transform" begin
        function ultra2leg(U::Ultraspherical)
            @assert ApproxFunOrthogonalPolynomials.order(U) == 0.5
            Legendre(domain(U))
        end
        function ultra2leg(U::NormalizedPolynomialSpace{<:Ultraspherical})
            L = ultra2leg(ApproxFunBase.canonicalspace(U))
            NormalizedPolynomialSpace(L)
        end
        @testset for T in (Float32, Float64), ET in (T, complex(T))
            v = Array{ET}(undef, 10)
            v2 = similar(v)
            M = Array{ET}(undef, 10, 10)
            M2 = similar(M)
            A = Array{ET}(undef, 10, 10, 10)
            A2 = similar(A)
            @testset for d in ((), (0..1,)), order in (0.5, 1, 3)
                U = Ultraspherical(order, d...)
                Slist = (U, NormalizedPolynomialSpace(U))
                @testset for S in Slist
                    if order == 0.5
                        L = ultra2leg(S)
                        v .= rand.(eltype(v))
                        @test transform(S, v) ≈ transform(L, v)
                    end
                    test_transform!(v, v2, S)
                end
                @testset for S1 in Slist, S2 in Slist
                    S = S1 ⊗ S2
                    if order == 0.5
                        L = ultra2leg(S1) ⊗ ultra2leg(S2)
                        M .= rand.(eltype(M))
                        @test transform(S, M) ≈ transform(L, M)
                    end
                    test_transform!(M, M2, S)
                end
                @testset for S1 in Slist, S2 in Slist, S3 in Slist
                    S = S1 ⊗ S2 ⊗ S3
                    if order == 0.5
                        L = ultra2leg(S1) ⊗ ultra2leg(S2) ⊗ ultra2leg(S3)
                        A .= rand.(eltype(A))
                        @test transform(S, A) ≈ transform(L, A)
                    end
                    test_transform!(A, A2, S)
                end
            end
        endend
        end
    end

    @testset "casting bug ApproxFun.jl#770" begin
        f = Fun((t,x)->im*exp(t)*sinpi(x), Ultraspherical(2)^2)
        @test f(0.1, 0.2) ≈ im*exp(0.1)*sinpi(0.2)
    end

    @testset "Evaluation" begin
        c = [i^2 for i in 1:4]
        @testset for d in (0..1, ChebyshevInterval()), order in (1, 2, 0.5)
            @testset for _sp in (Ultraspherical(order), Ultraspherical(order,d)),
                    sp in (_sp, NormalizedPolynomialSpace(_sp))
                d = domain(sp)
                f = Fun(sp, c)
                for ep in (leftendpoint, rightendpoint),
                        ev in (ApproxFunBase.ConcreteEvaluation, Evaluation)
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
    end

    @testset "Multiplication" begin
        M = Multiplication(Fun(), Ultraspherical(0.5))
        f = Fun(Ultraspherical(0.5))
        f2 = Fun(x->x^2, Ultraspherical(0.5))
        @test M * f ≈ f2
    end

    @testset "Integral" begin
        d = 0..1
        A = @inferred Integral(0..1)
        x = Derivative() * A * Fun(d)
        @test x(0.2) ≈ 0.2
        d = 0.0..1.0
        A = @inferred Integral(d)
        x = Derivative() * A * Fun(d)
        @test x(0.2) ≈ 0.2

        @testset for sp in (Ultraspherical(1), Ultraspherical(2), Ultraspherical(0.5))
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
    end
end
