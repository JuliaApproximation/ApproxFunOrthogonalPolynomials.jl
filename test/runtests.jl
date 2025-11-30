module ApproxFunOrthogonalPolynomials_Runtests

using ApproxFunOrthogonalPolynomials
using ParallelTestRunner

const init_code = quote
    using ApproxFunOrthogonalPolynomials
    using ApproxFunOrthogonalPolynomials: isapproxminhalf, isequalminhalf, isequalhalf, isapproxhalfoddinteger
    using LinearAlgebra
    using Test
    using Static
    using HalfIntegers
    using OddEvenIntegers
    using Test
    include("testutils.jl")
end

# Start with autodiscovered tests
testsuite = find_tests(pwd())

# Parse arguments
args = parse_args(ARGS)

if filter_tests!(testsuite, args)
    # There are weird non-deterministic `ReadOnlyMemoryError`s on Windows,
    # so this test is disabled for now
    delete!(testsuite, "testutils.jl")
    if Sys.iswindows()
        delete!(testsuite, "UltrasphericalTest.jl")
    end
end

runtests(ApproxFunOrthogonalPolynomials, args; init_code, testsuite)

end # module
