using ApproxFunOrthogonalPolynomials, LinearAlgebra, Test
using Aqua

@testset "Project quality" begin
    Aqua.test_all(ApproxFunOrthogonalPolynomials, ambiguities=false, undefined_exports=false)
end

@testset "Domain" begin
    @test reverseorientation(Arc(1,2,(0.1,0.2))) == Arc(1,2,(0.2,0.1))
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
