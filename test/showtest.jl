module ShowTest

using ApproxFunOrthogonalPolynomials
using ApproxFunBase

@testset "show" begin
	@test repr(Chebyshev()) == "Chebyshev()"
	@test repr(NormalizedChebyshev()) == "NormalizedChebyshev()"
	@test repr(Ultraspherical(1)) == "Ultraspherical(1)"
	@test repr(Legendre()) == "Legendre()"
	@test repr(Jacobi(1,2)) == "Jacobi(1,2)"
	@test repr(Jacobi(1.0,2.0)) == "Jacobi(1.0,2.0)"

	@test repr(Chebyshev(0..1)) == "Chebyshev(0..1)"
	@test repr(NormalizedChebyshev(0..1)) == "NormalizedChebyshev(0..1)"
	@test repr(Ultraspherical(1,0..1)) == "Ultraspherical(1,0..1)"
	@test repr(Legendre(0..1)) == "Legendre(0..1)"
	@test repr(Jacobi(1,2,0..1)) == "Jacobi(1,2,0..1)"
	@test repr(Jacobi(1.0,2.0,0..1)) == "Jacobi(1.0,2.0,0..1)"

	io = IOBuffer()
	@testset "Derivative" begin
		D = Derivative()
		summarystr = summary(D)
		@test repr(D) == summarystr
		show(io, MIME"text/plain"(), D)
		@test contains(String(take!(io)), summarystr)

		D = Derivative(Chebyshev())
		summarystr = summary(D)
		show(io, MIME"text/plain"(), D)
		@test contains(String(take!(io)), summarystr)
	end
	@testset "SubOperator" begin
		D = Derivative(Chebyshev())
		S = @view D[1:10, 1:10]
		summarystr = summary(S)
		show(io, MIME"text/plain"(), S)
		@test contains(String(take!(io)), summarystr)
	end
	@testset "Evaluation" begin
		E = Evaluation(Chebyshev(), 0)
		summarystr = summary(E)
		show(io, MIME"text/plain"(), E)
		@test contains(String(take!(io)), summarystr)

		EA = Evaluation(Chebyshev(), 0)'
		summarystr = summary(EA)
		show(io, MIME"text/plain"(), EA)
		@test contains(String(take!(io)), summarystr)

		EA = transpose(Evaluation(Chebyshev(), 0))
		summarystr = summary(EA)
		show(io, MIME"text/plain"(), EA)
		@test contains(String(take!(io)), summarystr)
	end
end

end # module
