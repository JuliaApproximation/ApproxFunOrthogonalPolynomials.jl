module ApproxFunOrthogonalPolynomialsPolynomialsExt

using ApproxFunOrthogonalPolynomials
import Polynomials
import ApproxFunOrthogonalPolynomials.ApproxFunBase: Fun

Fun(C::Polynomials.ChebyshevT, s::Chebyshev{<:ChebyshevInterval}) =
	Fun(s, float.(Polynomials.coeffs(C)))

Polynomials.ChebyshevT(f::Fun{<:Chebyshev{<:ChebyshevInterval}}) =
	Polynomials.ChebyshevT(copy(coefficients(f)))

end
