module ApproxFunOrthogonalPolynomials

using BandedMatrices
using BlockArrays
using BlockBandedMatrices
using DomainSets
using FastGaussQuadrature
using FastTransforms
using FillArrays
using HalfIntegers
using IntervalSets
using LinearAlgebra
using OddEvenIntegers
using Reexport
using SpecialFunctions

@reexport using ApproxFunBase

import ApproxFunBase: Fun, SubSpace, WeightSpace, NoSpace, HeavisideSpace,
                    IntervalOrSegment, AnyDomain, ArraySpace,
                    AbstractTransformPlan, TransformPlan, ITransformPlan,
                    ConcreteConversion, ConcreteMultiplication, ConcreteDerivative,
                    ConcreteDefiniteIntegral, ConcreteDefiniteLineIntegral,
                    ConcreteVolterra, Volterra, Evaluation, EvaluationWrapper,
                    MultiplicationWrapper, ConversionWrapper, DerivativeWrapper,
                    Conversion, defaultcoefficients, default_Fun, Multiplication,
                    Derivative, bandwidths, ConcreteEvaluation, ConcreteIntegral,
                    DefiniteLineIntegral, DefiniteIntegral, IntegralWrapper,
                    ReverseOrientation, ReverseOrientationWrapper, ReverseWrapper,
                    Reverse, NegateEven, Dirichlet, ConcreteDirichlet,
                    TridiagonalOperator, SubOperator, Space, @containsconstants, spacescompatible,
                    canonicalspace, domain, setdomain, prectype, domainscompatible,
                    plan_transform, plan_itransform, plan_transform!, plan_itransform!,
                    transform, itransform, hasfasttransform,
                    CanonicalTransformPlan, ICanonicalTransformPlan,
                    Integral, domainspace, rangespace, boundary,
                    maxspace, hasconversion, points,
                    union_rule, conversion_rule, maxspace_rule, conversion_type,
                    linesum, differentiate, integrate, linebilinearform, bilinearform,
                    Segment, IntervalOrSegmentDomain, PiecewiseSegment, isambiguous,
                    eps, isperiodic, arclength, complexlength,
                    invfromcanonicalD, fromcanonical, tocanonical, fromcanonicalD,
                    tocanonicalD, canonicaldomain, setcanonicaldomain, mappoint,
                    reverseorientation, checkpoints, evaluate, extrapolate, mul_coefficients,
                    coefficients, isconvertible, clenshaw, ClenshawPlan,
                    toeplitz_axpy!, sym_toeplitz_axpy!, hankel_axpy!,
                    ToeplitzOperator, SymToeplitzOperator, SpaceOperator, cfstype,
                    alternatesign!, mobius, chebmult_getindex, intpow, alternatingsum,
                    extremal_args, chebyshev_clenshaw, recA, recB, recC, roots,
                    diagindshift, rangetype, weight, isapproxinteger, default_Dirichlet, scal!,
                    components, promoterangespace,
                    block, blockstart, blockstop, blocklengths, isblockbanded,
                    pointscompatible, affine_setdiff, complexroots,
                    ℓ⁰, recα, recβ, recγ, ℵ₀, ∞, RectDomain,
                    assert_integer, supportsinplacetransform, ContinuousSpace, SpecialEvalPtType,
                    isleftendpoint, isrightendpoint, evaluation_point, boundaryfn, ldiffbc, rdiffbc,
                    LeftEndPoint, RightEndPoint, normalizedspace, promotedomainspace,
                    bandmatrices_eigen, SymmetricEigensystem, SkewSymmetricEigensystem,
                    mean, # differs from Statistics.mean after https://github.com/JuliaApproximation/ApproxFunBase.jl/pull/506
                    israggedbelow, bandwidthssum

import DomainSets: Domain, indomain, UnionDomain, FullSpace, Point,
            Interval, ChebyshevInterval, boundary, rightendpoint, leftendpoint,
            dimension, WrappedDomain

import BandedMatrices: bandshift, bandwidth, colstop, bandwidths, BandedMatrix

import Base: convert, getindex, eltype, <, <=, +, -, *, /, ^, ==,
                show, stride, sum, cumsum, conj, inv,
                complex, exp, sqrt, abs, sign, issubset,
                first, last, rand, intersect, setdiff,
                isless, union, angle, isnan, isapprox, isempty,
                minimum, maximum, extrema, zeros, one, promote_rule,
                getproperty, real, imag, max, min, log, acos,
                sin, cos, asinh, acosh, atanh, ones,
                Matrix, size
                # atan, tan, tanh, asin, sec, sinh, cosh,
                # split

using FastTransforms: plan_chebyshevtransform, plan_chebyshevtransform!,
                        plan_ichebyshevtransform, plan_ichebyshevtransform!,
                        pochhammer, lgamma

import BlockBandedMatrices: blockbandwidths

# we need to import all special functions to use Calculus.symbolic_derivatives_1arg
import SpecialFunctions: erfcx, dawson,
                    hankelh1, hankelh2, besselj, bessely, besseli, besselk,
                    besselkx, hankelh1x, hankelh2x
                    # The following are not extended here.
                    # Some of these are extended in ApproxFunBase
                    # erf, erfinv, erfc, erfcinv, erfi, gamma, lgamma, digamma, invdigamma,
                    # trigamma, airyai, airybi, airyaiprime, airybiprime, besselj0,
                    # besselj1, bessely0, bessely1

using StaticArrays: SVector

import LinearAlgebra: isdiag, eigvals, eigen

export bandmatrices_eigen, SymmetricEigensystem, SkewSymmetricEigensystem

points(d::IntervalOrSegmentDomain{T},n::Integer) where {T} =
    _maybefromcanonical(d, chebyshevpoints(float(real(eltype(T))), n))  # eltype to handle point
_maybefromcanonical(d, pts) = fromcanonical.(Ref(d), pts)
_maybefromcanonical(::ChebyshevInterval, pts::FastTransforms.ChebyshevGrid) = pts
function _maybefromcanonical(d::IntervalOrSegment{<:Union{Number, SVector}}, pts::FastTransforms.ChebyshevGrid)
    shift = mean(d)
    scale = complexlength(d) / 2
    ShiftedChebyshevGrid(pts, shift, scale)
end

struct ShiftedChebyshevGrid{T, S, C<:FastTransforms.ChebyshevGrid} <: AbstractVector{T}
    grid :: C
    shift :: S
    scale :: S
end
function ShiftedChebyshevGrid(grid::G, shift::S, scale::S) where {G,S}
    T = typeof(zero(eltype(G)) * zero(S))
    ShiftedChebyshevGrid{T,S,G}(grid, shift, scale)
end
size(S::ShiftedChebyshevGrid) = size(S.grid)
Base.@propagate_inbounds getindex(S::ShiftedChebyshevGrid, i::Int) = S.shift + S.grid[i] * S.scale
function Base.showarg(io::IO, S::ShiftedChebyshevGrid, toplevel::Bool)
    print(io, "ShiftedChebyshevGrid{", eltype(S), "}")
end

bary(v::AbstractVector{Float64},d::IntervalOrSegmentDomain,x::Float64) = bary(v,tocanonical(d,x))

strictconvert(T::Type, x) = convert(T, x)::T

convert_vector(v::AbstractVector) = convert(Vector, v)
convert_vector(t::Tuple) = [t...]

convert_absvector(t::Tuple) = SVector(t)
convert_absvector(v::AbstractVector) = v

convert_vector_or_svector(v::AbstractVector) = convert(Vector, v)
convert_vector_or_svector(t::Tuple) = SVector(t)

const TupleOrVector{T} = Union{Tuple{T,Vararg{T}},AbstractVector{<:T}}

# If any of the orders is an Integer, use == for an exact comparison, else fall back to isapprox
# We assume that Integer orders are deliberately chosen to be exact
compare_op(::Union{Integer, HalfInteger}, args...) = ==
compare_op() = ≈
compare_op(::Any, args...) = compare_op(args...)
compare_orders(a::Number, b::Number) = compare_op(a, b)(a, b)

# work around type promotions to preserve types for StepRanges involving HalfOddIntegers with a unit step
const HalfOddInteger{T<:Integer} = Half{Odd{T}}

# return 1/2, possibly preserving types but not being too fussy
_onehalf(x) = onehalf(x)
_onehalf(::Integer) = half(Odd(1))

# return -1/2, possibly preserving types but not being too fussy
_minonehalf(x) = -onehalf(x)
_minonehalf(::Integer) = half(Odd(-1))

isapproxminhalf(a) = a ≈ _minonehalf(a)
isapproxminhalf(::Integer) = false

isequalminhalf(x) = x == _minonehalf(x)
isequalminhalf(::Integer) = false

isequalhalf(x) = x == _onehalf(x)
isequalhalf(x::Integer) = false

isapproxhalfoddinteger(a) = !isapproxinteger(a) && isapproxinteger(a+_onehalf(a))
isapproxhalfoddinteger(a::HalfOddInteger) = true
isapproxhalfoddinteger(::Integer) = false

include("bary.jl")


include("ultraspherical.jl")
include("Domains/Domains.jl")
include("Spaces/Spaces.jl")
include("roots.jl")
include("specialfunctions.jl")
include("fastops.jl")
include("show.jl")

if !isdefined(Base, :get_extension)
    include("../ext/ApproxFunOrthogonalPolynomialsStaticExt.jl")
    include("../ext/ApproxFunOrthogonalPolynomialsPolynomialsExt.jl")
end

end
