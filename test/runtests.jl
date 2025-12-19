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

if "--downstream_integration_test" in ARGS
    delete!(testsuite, "AquaTest")
end
filtered_args = filter(!=("--downstream_integration_test"), ARGS)

# Parse arguments
args = parse_args(filtered_args)

if filter_tests!(testsuite, args)
    delete!(testsuite, "testutils")
    # There are weird non-deterministic `ReadOnlyMemoryError`s on Windows,
    # so this test is disabled for now
    if Sys.iswindows()
        delete!(testsuite, "UltrasphericalTest")
    end
end

runtests(ApproxFunOrthogonalPolynomials, args; init_code, testsuite)

end # module
