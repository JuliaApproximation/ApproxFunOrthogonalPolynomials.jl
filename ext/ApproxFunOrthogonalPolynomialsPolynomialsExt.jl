module ApproxFunOrthogonalPolynomialsPolynomialsExt

using ApproxFunOrthogonalPolynomials
import Polynomials
import ApproxFunBase: Fun

Fun(C::Polynomials.ChebyshevT, s::Chebyshev{<:ChebyshevInterval}) =
	Fun(s, float.(Polynomials.coeffs(C)))

end
