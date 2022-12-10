Space(d::IntervalOrSegment) = Chebyshev(d)
Space(d::FullSpace{<:Real}) = Chebyshev(Line())

# TODO: mode these functions to ApproxFunBase
# Currently, Space(d::Interval) isn't type-stable, so the spaces are
# explicitly listed in these calls.
Fun(::typeof(identity), d::IntervalOrSegment{<:Number}) =
    Fun(Chebyshev(d), [mean(d), complexlength(d)/2])

# the default domain space is higher to avoid negative ultraspherical spaces
Integral(d::IntervalOrSegment,n::Integer) = Integral(Ultraspherical(1,d), n)
