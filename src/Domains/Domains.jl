include("Ray.jl")
include("Arc.jl")
include("Line.jl")
include("IntervalCurve.jl")

# sort

isless(d1::IntervalOrSegment{T1},d2::Ray{false,T2}) where {T1<:Real,T2<:Real} = d1 ≤ d2.center
isless(d2::Ray{true,T2},d1::IntervalOrSegment{T1}) where {T1<:Real,T2<:Real} = d2.center ≤ d1
