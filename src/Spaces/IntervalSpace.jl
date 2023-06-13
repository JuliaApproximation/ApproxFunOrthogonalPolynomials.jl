Space(d::IntervalOrSegment) = Chebyshev(d)
Space(d::FullSpace{<:Real}) = Chebyshev(Line())

# TODO: mode these functions to ApproxFunBase
function Fun(::typeof(identity), d::IntervalOrSegment{<:Number})
    Fun(Space(d), [mean(d), complexlength(d)/2])
end

# the default domain space is higher to avoid negative ultraspherical spaces
function Integral(d::IntervalOrSegment, n::Number)
    assert_integer(n)
    Integral(Ultraspherical(1,d), n)
end
