module ApproxFunOrthogonalPolynomialsPolynomialsExt

using ApproxFunOrthogonalPolynomials
import Polynomials
# Specifying the full namespace is necessary because of https://github.com/JuliaLang/julia/issues/48533
# See https://github.com/JuliaStats/LogExpFunctions.jl/pull/63
import ApproxFunOrthogonalPolynomials.ApproxFunBase: Fun

Fun(C::Polynomials.ChebyshevT, s::Chebyshev{<:ChebyshevInterval}) =
	Fun(s, float.(Polynomials.coeffs(C)))

Polynomials.ChebyshevT(f::Fun{<:Chebyshev{<:ChebyshevInterval}}) =
	Polynomials.ChebyshevT(copy(coefficients(f)))

end
