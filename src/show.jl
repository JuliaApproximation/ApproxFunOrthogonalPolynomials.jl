function show(io::IO,d::Line)
    if d.center == angle(d) == 0 && d.α == d.β == -1.
        print(io,"ℝ")
    elseif  d.α == d.β == -1.
        print(io,"Line(", d.center, ",",  angle(d), ")")
    else
        print(io,"Line(", d.center, ",", angle(d), ",", d.α, ",", d.β, ")")
    end
end

function show(io::IO,d::Ray)
    if d.orientation && angle(d)==0
        print(io,"【", d.center, ",∞❫")
    elseif  d.orientation && angle(d)==1.0π
        print(io,"【", d.center, ",-∞❫")
    elseif  d.orientation
        print(io,"【", d.center, ",exp(", angle(d), "im)∞❫")
    elseif !d.orientation  && angle(d)==0
        print(io,"❪∞,", d.center, "】")
    elseif !d.orientation && angle(d)==1.0π
        print(io,"❪-∞,", d.center, "】")
    else
        print(io,"❪exp(", angle(d), "im)∞,", d.center, "】")
    end
end

## Spaces
_maybetoint(x::Union{Integer, StaticInt}) = Int(x)
_maybetoint(x) = x

_spacename(io, ::Chebyshev) = print(io, "Chebyshev(")
_spacename(io, S::Ultraspherical) = print(io,"Ultraspherical(", _maybetoint(order(S)))
function _spacename(io, S::Jacobi)
    if S.a == S.b == 0
        print(io,"Legendre(")
    else
        print(io,"Jacobi(", _maybetoint(S.b), ",", _maybetoint(S.a))
    end
end

function _maybeshowdomain(io, d)
    if !(d isa ChebyshevInterval)
        show(io, d)
    end
end

_showsorders(C::Chebyshev) = false
_showsorders(C::Ultraspherical) = true
_showsorders(C::Jacobi) = !(C.b == 0 && C.a == 0)

function show(io::IO, S::Union{Chebyshev, Ultraspherical, Jacobi})
    _spacename(io, S)
    !(domain(S) isa ChebyshevInterval) && _showsorders(S) && print(io, ",")
    _maybeshowdomain(io, domain(S))
    print(io,")")
end

show(io::IO, S::NormalizedPolynomialSpace) = (print(io, "Normalized"); show(io, S.space))
