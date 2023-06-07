module ApproxFunOrthogonalPolynomials_Runtests

using ApproxFunOrthogonalPolynomials
using ApproxFunOrthogonalPolynomials: isapproxminhalf, isequalminhalf, isequalhalf, isapproxhalfoddinteger
using LinearAlgebra
using Test
using Aqua
using Static
using HalfIntegers
using OddEvenIntegers

@testset "Project quality" begin
    Aqua.test_all(ApproxFunOrthogonalPolynomials, ambiguities=false,
        stale_deps=(; ignore=[:ApproxFunBaseTest]), piracy = false,
        # only test formatting on VERSION >= v1.7
        # https://github.com/JuliaTesting/Aqua.jl/issues/105#issuecomment-1551405866
        project_toml_formatting = VERSION >= v"1.9")
end

@testset "Domain" begin
    @test reverseorientation(Arc(1,2,(0.1,0.2))) == Arc(1,2,(0.2,0.1))
end

# missing import bug
@test ApproxFunOrthogonalPolynomials.Matrix === Base.Matrix

include("testutils.jl")

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

include("ClenshawTest.jl"); GC.gc()
include("ChebyshevTest.jl"); GC.gc()
# There are weird non-deterministic `ReadOnlyMemoryError`s on Windows,
# so this test is disabled for now
if !Sys.iswindows()
    include("UltrasphericalTest.jl"); GC.gc()
end
include("JacobiTest.jl"); GC.gc()
include("LaguerreTest.jl"); GC.gc()
include("HermiteTest.jl"); GC.gc()
include("SpacesTest.jl"); GC.gc()
include("ComplexTest.jl"); GC.gc()
include("broadcastingtest.jl"); GC.gc()
include("OperatorTest.jl"); GC.gc()
include("ODETest.jl"); GC.gc()
include("EigTest.jl"); GC.gc()
include("VectorTest.jl"); GC.gc()
include("MultivariateTest.jl"); GC.gc()
include("PDETest.jl"); GC.gc()

include("SpeedTest.jl"); GC.gc()
include("SpeedODETest.jl"); GC.gc()
include("SpeedPDETest.jl"); GC.gc()
include("SpeedOperatorTest.jl"); GC.gc()
include("showtest.jl"); GC.gc()
include("miscAFBase.jl"); GC.gc()

end # module
