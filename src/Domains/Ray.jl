export Ray

"""
    Ray{a}(c,L,o)

represents a scaled ray (with scale factor L) at angle `a` starting at `c`, with orientation out to
infinity (`o = true`) or back from infinity (`o = false`).
"""
struct Ray{angle,T<:Number} <: SegmentDomain{T}
    center::T
    L::T
    orientation::Bool
    Ray{angle,T}(c,L,o) where {angle,T} = new{angle,T}(c,L,o)
    Ray{angle,T}(c,o) where {angle,T} = new{angle,T}(c,one(T),o)
    Ray{angle,T}(c) where {angle,T} = new{angle,T}(c,one(T),true)
    Ray{angle,T}() where {angle,T} = new{angle,T}(zero(T),one(T),true)
    Ray{angle,T}(r::Ray{angle,T}) where {angle,T} = r
end

Ray{a}(c,L,o) where {a} = Ray{a,typeof(c)}(c,L,o)
Ray{a}(c,o) where {a} = Ray{a,typeof(c)}(c,one(typeof(c)),o)
Ray{a}(c::Number) where {a} = Ray{a,typeof(c)}(c)
Ray{a}() where {a} = Ray{a,Float64}()

angle(d::Ray{a}) where {a} = a*π

# ensure the angle is always in (-1,1]
@inline function _Ray(c,a,L,o)
    angle = if iszero(a)
        false
    else
        (abs(a)≈(1.0π) ? true : mod(a/π-1,-2)+1)
    end
    Ray{angle,typeof(c)}(c,L,o)
end
@static if VERSION >= v"1.8"
    Base.@constprop :aggressive Ray(c,a,L,o) = _Ray(c,a,L,o)
else
    Ray(c,a,L,o) = _Ray(c,a,L,o)
end
Ray(c,a,o) = Ray(c,a,one(typeof(c)),o)
Ray(c,a) = Ray(c,a,one(typeof(c)),true)

Ray() = Ray{false}()



##deal with vector

_rayangle(x) = angle(x)
_rayangle(x::Real) = signbit(x) * oftype(float(x), pi)

function convert(::Type{Ray}, d::AbstractInterval)
    a,b = endpoints(d)
    @assert abs(a)==Inf || abs(b)==Inf

    if abs(b)==Inf
        Ray(a,_rayangle(b),one(typeof(a)),true)
    else #abs(a)==Inf
        Ray(b,_rayangle(a),one(typeof(a)),false)
    end
end
Ray(d::AbstractInterval) = strictconvert(Ray, d)


isambiguous(d::Ray)=isnan(d.center)
convert(::Type{Ray{a,T}},::AnyDomain) where {a,T<:Number} = Ray{a,T}(NaN,true)
convert(::Type{IT},::AnyDomain) where {IT<:Ray} = Ray(NaN,NaN)


isempty(::Ray) = false

## Map interval

function mobiuspars(d::Ray)
    s=(d.orientation ? 1 : -1)
    α=conj(cisangle(d))/d.L
    c=d.center
    s*α,-s*(1+α*c),α,1-α*c
end


for OP in (:mobius,:mobiusinv,:mobiusD,:mobiusinvD)
    @eval $OP(a::Ray,z) = $OP(mobiuspars(a)...,z)
end

ray_tocanonical(x) = isinf(x) ? one(x) : (x-1)/(1+x)
ray_tocanonicalD(x) = isinf(x) ? zero(x) : 2*(1/(1+x))^2
ray_fromcanonical(x) = (1+x)/(1-x)
ray_fromcanonicalD(x) = 2*(1/(x-1))^2
ray_invfromcanonicalD(x) = (x-1)^2/2


for op in (:ray_tocanonical,:ray_tocanonicalD)
    @eval $op(L,o,x)= (o ? 1 : -1)*$op(x/L)
end

ray_fromcanonical(L,o,x)=L*ray_fromcanonical((o ? 1 : -1)*x)
ray_fromcanonicalD(L,o,x)=L*(o ? 1 : -1)*ray_fromcanonicalD((o ? 1 : -1)*x)
ray_invfromcanonicalD(L,o,x)=L*(o ? 1 : -1)*ray_invfromcanonicalD((o ? 1 : -1)*x)

cisangle(::Ray{a}) where {a}=cis(a*π)
cisangle(::Ray{false})=1
cisangle(::Ray{true})=-1

tocanonical(d::Ray,x) =
    ray_tocanonical(d.L,d.orientation,conj(cisangle(d)).*(x-d.center))
tocanonicalD(d::Ray,x) =
    conj(cisangle(d)).*ray_tocanonicalD(d.L,d.orientation,conj(cisangle(d)).*(x-d.center))
fromcanonical(d::Ray,x) = cisangle(d)*ray_fromcanonical(d.L,d.orientation,x)+d.center
fromcanonical(d::Ray{false},x) = ray_fromcanonical(d.L,d.orientation,x)+d.center
fromcanonical(d::Ray{true},x) = -ray_fromcanonical(d.L,d.orientation,x)+d.center
fromcanonicalD(d::Ray,x) = cisangle(d)*ray_fromcanonicalD(d.L,d.orientation,x)
invfromcanonicalD(d::Ray,x) = conj(cisangle(d))*ray_invfromcanonicalD(d.L,d.orientation,x)
arclength(d::Ray) = Inf

==(d::Ray{a},m::Ray{a}) where {a} = d.center == m.center && d.L == m.L


mappoint(a::Ray{false}, b::Ray{false}, x::Number) =
    b.center + b.L/a.L*(x - a.center)


function mappoint(a::Ray, b::Ray, x::Number)
    d = x - a.center;
    k = b.L/a.L * d * exp((angle(b)-angle(d))*im)
    k + b.center
end
