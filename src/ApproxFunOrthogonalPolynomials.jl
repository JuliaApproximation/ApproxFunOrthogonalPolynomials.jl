module ApproxFunOrthogonalPolynomials
using Base, LinearAlgebra, Reexport, BandedMatrices, BlockBandedMatrices, AbstractFFTs, FFTW, BlockArrays, FillArrays, FastTransforms, IntervalSets, 
            DomainSets, Statistics, SpecialFunctions, FastGaussQuadrature, OrthogonalPolynomialsQuasi
            
@reexport using ApproxFunBase

import ApproxFunBase: Space, Fun

import AbstractFFTs: Plan, fft, ifft
import FFTW: plan_r2r!, fftwNumber, REDFT10, REDFT01, REDFT00, RODFT00, R2HC, HC2R,
                r2r!, r2r,  plan_fft, plan_ifft, plan_ifft!, plan_fft!

import DomainSets: Domain, indomain, UnionDomain, ProductDomain, FullSpace, Point, elements, DifferenceDomain,
            Interval, ChebyshevInterval, boundary, ∂, rightendpoint, leftendpoint,
            dimension, WrappedDomain
            
import BandedMatrices: bandrange, bandshift,
                inbands_getindex, inbands_setindex!, bandwidth, AbstractBandedMatrix,
                colstart, colstop, colrange, rowstart, rowstop, rowrange,
                bandwidths, _BandedMatrix, BandedMatrix            

import Base: values, convert, getindex, setindex!, *, +, -, ==, <, <=, >, |, !, !=, eltype, iterate,
                >=, /, ^, \, ∪, transpose, size, tail, broadcast, broadcast!, copyto!, copy, to_index, (:),
                similar, map, vcat, hcat, hvcat, show, summary, stride, sum, cumsum, sign, imag, conj, inv,
                complex, reverse, exp, sqrt, abs, abs2, sign, issubset, values, in, first, last, rand, intersect, setdiff,
                isless, union, angle, join, isnan, isapprox, isempty, sort, merge, promote_rule,
                minimum, maximum, extrema, argmax, argmin, findmax, findmin, isfinite,
                zeros, zero, one, promote_rule, repeat, length, resize!, isinf,
                getproperty, findfirst, unsafe_getindex, fld, cld, div, real, imag,
                @_inline_meta, eachindex, firstindex, lastindex, keys, isreal, OneTo,
                Array, Vector, Matrix, view, ones, @propagate_inbounds, print_array,
                split

import LinearAlgebra: BlasInt, BlasFloat, norm, ldiv!, mul!, det, eigvals, dot, cross,
                qr, qr!, rank, isdiag, istril, istriu, issymmetric, ishermitian,
                Tridiagonal, diagm, diagm_container, factorize, nullspace,
                Hermitian, Symmetric, adjoint, transpose, char_uplo                        

import FastTransforms: ChebyshevTransformPlan, IChebyshevTransformPlan, plan_chebyshevtransform,
                        plan_chebyshevtransform!, plan_ichebyshevtransform, plan_ichebyshevtransform!,
                        pochhammer, lgamma, chebyshevtransform!, ichebyshevtransform!

import BlockBandedMatrices: blockbandwidths, subblockbandwidths

# we need to import all special functions to use Calculus.symbolic_derivatives_1arg
# we can't do importall Base as we replace some Base definitions
import SpecialFunctions: sinpi, cospi, airy, besselh,
                    asinh, acosh,atanh, erfcx, dawson, erf, erfi,
                    sin, cos, sinh, cosh, airyai, airybi, airyaiprime, airybiprime,
                    hankelh1, hankelh2, besselj, besselj0, bessely, besseli, besselk,
                    besselkx, hankelh1x, hankelh2x, exp2, exp10, log2, log10,
                    tan, tanh, csc, asin, acsc, sec, acos, asec,
                    cot, atan, acot, sinh, csch, asinh, acsch,
                    sech, acosh, asech, tanh, coth, atanh, acoth,
                    expm1, log1p, lfact, sinc, cosc, erfinv, erfcinv, beta, lbeta,
                    eta, zeta, gamma, polygamma, invdigamma, digamma, trigamma,
                    abs, sign, log, expm1, tan, abs2, sqrt, angle, max, min, cbrt, log,
                    atan, acos, asin, erfc, inv


Space(::ChebyshevInterval{T}) where T = ChebyshevT{T}()
Fun(x::Number, sp::OrthogonalPolynomial) = Fun(sp, [x; zeros(typeof(x),∞)])

# points(d::IntervalOrSegmentDomain{T},n::Integer; kind::Int=1) where {T} =
#     fromcanonical.(Ref(d), chebyshevpoints(float(real(eltype(T))), n; kind=kind))  # eltype to handle point
# bary(v::AbstractVector{Float64},d::IntervalOrSegmentDomain,x::Float64) = bary(v,tocanonical(d,x))
                    
# include("bary.jl")


# include("ultraspherical.jl")
# include("Domains/Domains.jl")
# include("Spaces/Spaces.jl")
# include("roots.jl")
# include("specialfunctions.jl")
# include("fastops.jl")

end
