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
        @testset for T in [Float32, Float64, ComplexF32, ComplexF64]
            v = rand(T, 4)
            vc = copy(v)
            @testset for d in Any[(), (0..1,)], order in Any[0.5, 1, 3]
                S = Ultraspherical(order, d...)
                v2 = transform(S, v)
                if order == 0.5
                    @test v2 ≈ transform(Legendre(domain(S)), v)
                end
                transform!(S, v)
                @test v ≈ v2
                itransform!(S, v)
                @test v ≈ vc
                @test v ≈ itransform(S, v2)
            end
        end
    end
end
