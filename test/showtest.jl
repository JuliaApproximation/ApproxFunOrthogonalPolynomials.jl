@testset "show" begin
	@test repr(Chebyshev()) == "Chebyshev()"
	@test repr(NormalizedChebyshev()) == "NormalizedChebyshev()"
	@test repr(Ultraspherical(1)) == "Ultraspherical(1)"
	@test repr(Legendre()) == "Legendre()"
	@test repr(Jacobi(1,2)) == "Jacobi(1.0,2.0)"

	@test repr(Chebyshev(0..1)) == "Chebyshev(0..1)"
	@test repr(NormalizedChebyshev(0..1)) == "NormalizedChebyshev(0..1)"
	@test repr(Ultraspherical(1,0..1)) == "Ultraspherical(1,0..1)"
	@test repr(Legendre(0..1)) == "Legendre(0..1)"
	@test repr(Jacobi(1,2,0..1)) == "Jacobi(1.0,2.0,0..1)"
end
