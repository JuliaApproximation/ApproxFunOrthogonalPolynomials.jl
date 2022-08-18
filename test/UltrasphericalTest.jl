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
        f = x -> 3x^3 + 5x^2 + 2
        for dt in Any[(), (0..1,)],
                S in Any[Ultraspherical(1, dt...),
                         Ultraspherical(0.5,dt...),
                         Ultraspherical(3, dt...)]

            NS = NormalizedPolynomialSpace(S)
            f = Fun(f, S)
            g = Fun(f, NS)
            @test space(g) == NS
            d = domain(f)
            r = range(leftendpoint(d), rightendpoint(d), length=10)
            for x in r
                @test f(x) ≈ g(x)
            end
        end
    end
end
