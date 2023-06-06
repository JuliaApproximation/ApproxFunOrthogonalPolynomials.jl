module HermiteTest

using ApproxFunOrthogonalPolynomials
using ApproxFunBase
using ApproxFunBaseTest: testbandedoperator
using LinearAlgebra
using SpecialFunctions
using Test

@testset "Hermite and GaussWeight" begin
    @testset "Evaluation" begin
        f = Fun(x-> x + x^2, Hermite())
        @test f(1.0) ≈ 2.0
        @test values(f) ≈ points(f) + points(f).^2

        w = Fun(GaussWeight(), [1.0])
        @test w(0.1) ≈ exp(-0.1^2)
        w = Fun(GaussWeight(), Float64[])
        @test w(1) == 0

        w = Fun(GaussWeight(), Float64[1.0])
        f = Fun(x-> 1 + x + x^2, Hermite()) * w
        @test f(0.1) ≈ exp(-0.1^2) * (1+0.1+0.1^2)

        f = Fun(x-> 1 + x + x^2, Hermite(2)) * w
        @test f(0.1) ≈ exp(-0.1^2) * (1+0.1+0.1^2)

        w = Fun(GaussWeight(Hermite(2), 0), [1.0,2.0,3.0]);
        w̃ = Fun(Hermite(2), [1.0,2.0,3.0]);
        @test w(0.1) == w̃(0.1)

        @test ApproxFunOrthogonalPolynomials.Recurrence(Hermite())[1:10,1:10]/sqrt(2) ≈ ApproxFunOrthogonalPolynomials.Recurrence(Hermite(2))[1:10,1:10]

        @test points(Hermite(),10) == sqrt(2)points(Hermite(2),10)

        @test Fun(Hermite(),[1])(0.5) == 1
        @test Fun(Hermite(),[0,1])(0.5) == 2*0.5

        @test Fun(Hermite(2),[1])(0.5) == 1
        @test Fun(Hermite(2),[0,1])(0.5) == 2*sqrt(2)*0.5

        @test Fun(x-> 1 + x + x^2, Hermite())(0.5) ≈ Fun(x-> 1 + x + x^2, Hermite(2))(0.5) ≈ 1.75

        p = points(Hermite(),3)
        @test values(Fun(x-> 1 + x + x^2, Hermite())) ≈ 1 .+ p .+ p.^2

        p = points(Hermite(2),3)
        @test values(Fun(x-> 1 + x + x^2, Hermite(2))) ≈ 1 .+ p .+ p.^2

        w = Fun(GaussWeight(Hermite(2), 2), Float64[1.0])
        f = Fun(x-> 1 + x + x^2, Hermite()) * w
        @test space(f) == space(w)
        @test f(0.1) ≈ exp(-2*0.1^2) * (1+0.1+0.1^2)
        f = Fun(x-> 1 + x + x^2, Hermite(2)) * w
        @test space(f) == space(w)
        @test f(0.1) ≈ exp(-2*0.1^2) * (1+0.1+0.1^2)


        L = 1.3; x = 1.2;
        H₀ = Fun(Hermite(), [1.0])
        H̃₀ = Fun(Hermite(L), [1.0])
        @test H̃₀(x) ≈ H₀(sqrt(L) * x)

        L = 1.3; x = 1.2;
        H₁ = Fun(Hermite(), [0.0,1.0])
        H̃₁ = Fun(Hermite(L), [0.0,1.0])
        @test H̃₁(x) ≈ H₁(sqrt(L) * x)
    end


    @testset "Derivative" begin
        D = Derivative(Hermite())
        testbandedoperator(D)
        @test !isdiag(D)

        f = Fun( x-> x + x^2, Hermite())
        g = D * f
        @test g(1.) ≈ 3.
    end
end

end # module
