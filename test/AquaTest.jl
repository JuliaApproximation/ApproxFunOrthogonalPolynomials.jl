using Aqua
@testset "Project quality" begin
    Aqua.test_all(ApproxFunOrthogonalPolynomials, ambiguities=false, piracies = false)
end
