module ApproxFunOrthogonalPolynomials_Runtests

using ApproxFunOrthogonalPolynomials
using LinearAlgebra
using Test
using Aqua

@testset "Project quality" begin
    Aqua.test_all(ApproxFunOrthogonalPolynomials, ambiguities=false,
        stale_deps=(; ignore=[:ApproxFunBaseTest]))
end

@testset "Domain" begin
    @test reverseorientation(Arc(1,2,(0.1,0.2))) == Arc(1,2,(0.2,0.1))
end

include("testutils.jl")

include("ClenshawTest.jl"); GC.gc()
include("ChebyshevTest.jl"); GC.gc()
include("ComplexTest.jl"); GC.gc()
include("broadcastingtest.jl"); GC.gc()
include("OperatorTest.jl"); GC.gc()
include("ODETest.jl"); GC.gc()
include("EigTest.jl"); GC.gc()
include("VectorTest.jl"); GC.gc()
include("JacobiTest.jl"); GC.gc()
include("LaguerreTest.jl"); GC.gc()
include("HermiteTest.jl"); GC.gc()
include("SpacesTest.jl"); GC.gc()
include("MultivariateTest.jl"); GC.gc()
include("PDETest.jl"); GC.gc()

include("SpeedTest.jl"); GC.gc()
include("SpeedODETest.jl"); GC.gc()
include("SpeedPDETest.jl"); GC.gc()
include("SpeedOperatorTest.jl"); GC.gc()
include("showtest.jl"); GC.gc()
include("miscAFBase.jl"); GC.gc()

end # module
