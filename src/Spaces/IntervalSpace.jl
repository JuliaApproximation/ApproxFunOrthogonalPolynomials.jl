Space(d::IntervalOrSegment) = Chebyshev(d)
Space(d::FullSpace{<:Real}) = Chebyshev(Line())

Fun(::typeof(identity), d::IntervalOrSegment{<:Number}) =
    Fun(Chebyshev(d), [mean(d), complexlength(d)/2])
