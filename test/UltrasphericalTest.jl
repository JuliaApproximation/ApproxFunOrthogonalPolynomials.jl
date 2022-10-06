using ApproxFunOrthogonalPolynomials, ApproxFunBase, Test, SpecialFunctions, LinearAlgebra
import ApproxFunBase: testbandedbelowoperator, testbandedoperator, testspace, testtransforms, Vec,
                    maxspace, NoSpace, hasconversion, testfunctional,
                    reverseorientation, ReverseOrientation
import ApproxFunOrthogonalPolynomials: jacobip

@testset "Ultraspherical" begin
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
        for f in Any[x -> 3x^3 + 5x^2 + 2, x->x, identity]
            for dt in Any[(), (0..1,)],
                    S in Any[Ultraspherical(1, dt...),
                             Ultraspherical(0.5,dt...),
                             Ultraspherical(3, dt...)]

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
        @testset for T in Any[Float32, Float64], ET in Any[T, complex(T)]
            v = Array{ET}(undef, 10)
            v2 = similar(v)
            M = Array{ET}(undef, 10, 10)
            M2 = similar(M)
            A = Array{ET}(undef, 10, 10, 10)
            A2 = similar(A)
            @testset for d in Any[(), (0..1,)], order in Any[0.5, 1, 3]
                U = Ultraspherical(order, d...)
                Slist = Any[U, NormalizedPolynomialSpace(U)]
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
        end
    end

    @testset "casting bug ApproxFun.jl#770" begin
        f = Fun((t,x)->im*exp(t)*sinpi(x), Ultraspherical(2)^2)
        @test f(0.1, 0.2) ≈ im*exp(0.1)*sinpi(0.2)
    end
end
