module ApproxFunOrthogonalPolynomials
using Base, LinearAlgebra, Reexport, AbstractFFTs, FFTW, InfiniteArrays, FillArrays, FastTransforms, IntervalSets, 
            DomainSets, Statistics
            
@reexport using ApproxFunBase

import AbstractFFTs: Plan, fft, ifft
import FFTW: plan_r2r!, fftwNumber, REDFT10, REDFT01, REDFT00, RODFT00, R2HC, HC2R,
                r2r!, r2r,  plan_fft, plan_ifft, plan_ifft!, plan_fft!

import ApproxFunBase: normalize!, flipsign, FiniteRange, Fun, MatrixFun, UnsetSpace, VFun, RowVector,
                    UnivariateSpace, AmbiguousSpace, SumSpace, SubSpace, WeightSpace, NoSpace, Space,
                    HeavisideSpace, PointSpace,
                    IntervalOrSegment, RaggedMatrix, AlmostBandedMatrix,
                    AnyDomain, ZeroSpace, TrivialInterlacer, BlockInterlacer, 
                    AbstractTransformPlan, TransformPlan, ITransformPlan,
                    ConcreteConversion, ConcreteMultiplication, ConcreteDerivative, ConcreteIntegral,
                    ConcreteVolterra, Volterra, VolterraWrapper,
                    MultiplicationWrapper, ConversionWrapper, DerivativeWrapper, Evaluation,
                    Conversion, Multiplication, Derivative, Integral, bandwidths, 
                    ConcreteEvaluation, ConcreteDefiniteLineIntegral, ConcreteDefiniteIntegral, ConcreteIntegral,
                    DefiniteLineIntegral, DefiniteIntegral, ConcreteDefiniteIntegral, ConcreteDefiniteLineIntegral,
                    ReverseOrientation, ReverseOrientationWrapper, ReverseWrapper, Reverse, NegateEven, Dirichlet, ConcreteDirichlet,
                    TridiagonalOperator, SubOperator, Space, @containsconstants, spacescompatible,
                    hasfasttransform, canonicalspace, domain, setdomain, prectype, domainscompatible, 
                    plan_transform, plan_itransform, plan_transform!, plan_itransform!, transform, itransform, hasfasttransform, Integral, 
                    domainspace, rangespace, boundary, 
                    union_rule, conversion_rule, maxspace_rule, conversion_type, maxspace, hasconversion, points, 
                    rdirichlet, ldirichlet, lneumann, rneumann, ivp, bvp, 
                    linesum, differentiate, integrate, linebilinearform, bilinearform, 
                    UnsetNumber, coefficienttimes,
                    Segment, IntervalOrSegmentDomain, PiecewiseSegment, isambiguous, Vec, eps, isperiodic,
                    arclength, complexlength,
                    invfromcanonicalD, fromcanonical, tocanonical, fromcanonicalD, tocanonicalD, canonicaldomain, setcanonicaldomain, mappoint,
                    reverseorientation, checkpoints, evaluate, mul_coefficients, coefficients, isconvertible,
                    clenshaw, ClenshawPlan, sineshaw,
                    toeplitz_getindex, toeplitz_axpy!, ToeplitzOperator, hankel_getindex, 
                    SpaceOperator, ZeroOperator, InterlaceOperator,
                    interlace!, reverseeven!, negateeven!, cfstype, pad!,
                    extremal_args, hesseneigvals, chebyshev_clenshaw, recA, recB, recC, roots, chebmult_getindex, intpow, alternatingsum

import DomainSets: Domain, indomain, UnionDomain, ProductDomain, FullSpace, Point, elements, DifferenceDomain,
            Interval, ChebyshevInterval, boundary, ∂, rightendpoint, leftendpoint,
            dimension, Domain1d, Domain2d         

import Base: values, convert, getindex, setindex!, *, +, -, ==, <, <=, >, |, !, !=, eltype, iterate,
                >=, /, ^, \, ∪, transpose, size, reindex, tail, broadcast, broadcast!, copyto!, copy, to_index, (:),
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

import InfiniteArrays: Infinity, InfRanges, AbstractInfUnitRange, OneToInf                    

import FastTransforms: ChebyshevTransformPlan, IChebyshevTransformPlan, plan_chebyshevtransform,
                        plan_chebyshevtransform!, plan_ichebyshevtransform, plan_ichebyshevtransform!,
                        pochhammer

include("ultraspherical.jl")
include("Domains/Domains.jl")
include("Spaces/Spaces.jl")
include("roots.jl")
include("specialfunctions.jl")

end