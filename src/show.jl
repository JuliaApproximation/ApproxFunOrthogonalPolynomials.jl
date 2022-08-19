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

function show(io::IO, S::Chebyshev)
    print(io, "Chebyshev(")
    show(io,domain(S))
    print(io,")")
end

function show(io::IO, S::Ultraspherical)
    print(io,"Ultraspherical(", order(S), ",")
    show(io,domain(S))
    print(io,")")
end

function show(io::IO,S::Jacobi)
    S.a == S.b == 0 ? print(io,"Legendre(") : print(io,"Jacobi(", S.b, ",", S.a,",")
    show(io,domain(S))
    print(io,")")
end

show(io::IO, S::Chebyshev{<:ChebyshevInterval}) = print(io, "Chebyshev()")
show(io::IO, S::Ultraspherical{<:Any,<:ChebyshevInterval}) =
    print(io, "Ultraspherical(", order(S), ")")
show(io::IO, S::Jacobi{<:ChebyshevInterval}) =
    S.a == S.b == 0 ? print(io,"Legendre()") : print(io,"Jacobi(", S.b, ",", S.a,")")

show(io::IO, S::NormalizedPolynomialSpace) = (print(io, "Normalized"); show(io, S.space))
