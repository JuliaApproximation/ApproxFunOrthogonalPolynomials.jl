using ApproxFunOrthogonalPolynomials, LinearAlgebra, Test
using Aqua

@testset "Project quality" begin
    Aqua.test_all(ApproxFunOrthogonalPolynomials, ambiguities=false)
end

@testset "Domain" begin
    @test reverseorientation(Arc(1,2,(0.1,0.2))) == Arc(1,2,(0.2,0.1))
end

function test_transform!(v, v2, S)
    v .= rand.(eltype(v))
    v2 .= v
    @test itransform(S, transform(S, v)) ≈ v
    @test transform(S, itransform(S, v)) ≈ v
    transform!(S, v)
    @test transform(S, v2) ≈ v
    itransform!(S, v)
    @test v2 ≈ v
end

macro verbose(ex)
    head = ex.head
    args = ex.args
    @assert args[1] == Symbol("@testset")
    name = args[3] isa String ? args[3] : nothing
    if VERSION >= v"1.8"
        insert!(args, 3, Expr(:(=), :verbose, true))
    end
    Expr(head, args...)
end

include("ClenshawTest.jl")
include("ChebyshevTest.jl")
include("ComplexTest.jl")
include("broadcastingtest.jl")
include("OperatorTest.jl")
include("ODETest.jl")
include("EigTest.jl")
include("VectorTest.jl")
include("JacobiTest.jl")
include("LaguerreTest.jl")
include("HermiteTest.jl")
include("SpacesTest.jl")
include("MultivariateTest.jl")
include("PDETest.jl")

include("SpeedTest.jl")
include("SpeedODETest.jl")
include("SpeedPDETest.jl")
include("SpeedOperatorTest.jl")
include("showtest.jl")
include("miscAFBase.jl")
