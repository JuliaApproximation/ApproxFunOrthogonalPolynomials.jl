@testset "Domain" begin
    @test reverseorientation(Arc(1,2,(0.1,0.2))) == Arc(1,2,(0.2,0.1))
end

@testset "helpers" begin
    for f in [isequalminhalf, isapproxminhalf]
        @test f(-0.5)
        @test f(static(-0.5))
        @test f(half(Odd(-1)))
        @test !f(-0.2)
        @test !f(half(Odd(1)))
        @test !f(1)
        @test !f(static(1))
    end
    @test !isequalhalf(-0.5)
    @test !isequalhalf(static(-0.5))
    @test !isequalhalf(half(Odd(-1)))
    @test !isequalhalf(-0.2)
    @test isequalhalf(0.5)
    @test isequalhalf(static(0.5))
    @test isequalhalf(half(Odd(1)))
    @test !isequalhalf(1)
    @test !isequalhalf(static(1))

    @test isapproxhalfoddinteger(0.5)
    @test isapproxhalfoddinteger(static(0.5))
    @test isapproxhalfoddinteger(half(Odd(1)))
    @test !isapproxhalfoddinteger(1)
    @test !isapproxhalfoddinteger(static(1))

    @test ApproxFunOrthogonalPolynomials._minonehalf(2) == -0.5
    @test ApproxFunOrthogonalPolynomials._onehalf(2) == 0.5
end
