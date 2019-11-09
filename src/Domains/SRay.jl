export SRay

"""
    SRay{a}(c,L,o)

represents a scaled ray (with scale factor L) at angle `a` starting at `c`, with orientation out to
infinity (`o = true`) or back from infinity (`o = false`).
"""
struct SRay{angle,T<:Number} <: SegmentDomain{T}
    center::T
    L::T
    orientation::Bool  # orientation is obsolete because L < 0 is allowed
    SRay{angle,T}(c,L,o) where {angle,T} = new{angle,T}(c,L,o)
    SRay{angle,T}(c) where {angle,T} = new{angle,T}(c,one(T),true)
    SRay{angle,T}() where {angle,T} = new{angle,T}(zero(T),one(T),true)
    SRay{angle,T}(r::SRay{angle,T}) where {angle,T} = r
    SRay{angle,T}(r::Ray{angle,T}) where {angle,T} = new{angle,T}(r.center,one(T),r.orientation)
end

SRay{a}(c,L,o) where {a} = SRay{a,typeof(c)}(c,L,o)
SRay{a}(c::Number) where {a} = SRay{a,typeof(c)}(c)
SRay{a}() where {a} = SRay{a,Float64}()

angle(d::SRay{a}) where {a} = a*π

# ensure the angle is always in (-1,1]
SRay(c,a,L,o) = SRay{a==0 ? false : (abs(a)≈(1.0π) ? true : mod(a/π-1,-2)+1),typeof(c)}(c,L,o)
SRay(c,a,L) = SRay(c,a,L,true)

SRay() = SRay{false}()



##deal with vector

function convert(::Type{SRay}, d::AbstractInterval)
    a,b = endpoints(d)
    @assert abs(a)==Inf || abs(b)==Inf

    if abs(b)==Inf
        SRay(a,angle(b),one(typeof(a)),true)
    else #abs(a)==Inf
        SRay(b,angle(a),one(typeof(a)),false)
    end
end
SRay(d::AbstractInterval) = convert(SRay, d)


isambiguous(d::SRay)=isnan(d.center)
convert(::Type{SRay{a,T}},::AnyDomain) where {a,T<:Number} = SRay{a,T}(NaN,true)
convert(::Type{IT},::AnyDomain) where {IT<:Ray} = SRay(NaN,NaN)


isempty(::SRay) = false

## Map interval

function mobiuspars(d::SRay)
    s=(d.orientation ? 1 : -1)
    α=conj(cisangle(d))/d.L
    c=d.center
    s*α,-s*(1+α*c),α,1-α*c
end


for OP in (:mobius,:mobiusinv,:mobiusD,:mobiusinvD)
    @eval $OP(a::SRay,z) = $OP(mobiuspars(a)...,z)
end

# Already defined in Ray.jl
#ray_tocanonical(x,L) = isinf(x) ? one(x) : (x-1)/(1+x)
#ray_tocanonicalD(x) = isinf(x) ? zero(x) : 2*(1/(1+x))^2
#ray_fromcanonical(x) = (1+x)/(1-x)
#ray_fromcanonicalD(x) = 2*(1/(x-1))^2
#ray_invfromcanonicalD(x) = (x-1)^2/2


for op in (:ray_tocanonical,:ray_tocanonicalD)
    @eval $op(L,o,x)=L*(o ? 1 : -1)*$op(x)
end
ray_fromcanonical(L,o,x)=ray_fromcanonical((o ? 1 : -1)*x/L)
ray_fromcanonicalD(L,o,x)=(o ? 1 : -1)*ray_fromcanonicalD((o ? 1 : -1)*x/L)
ray_invfromcanonicalD(L,o,x)=(o ? 1 : -1)*ray_invfromcanonicalD((o ? 1 : -1)*x/L)

cisangle(::SRay{a}) where {a}=cis(a*π)
cisangle(::SRay{false})=1
cisangle(::SRay{true})=-1

tocanonical(d::SRay,x) =
    ray_tocanonical(d.L,d.orientation,conj(cisangle(d)).*(x-d.center))
tocanonicalD(d::SRay,x) =
    conj(cisangle(d)).*ray_tocanonicalD(d.L,d.orientation,conj(cisangle(d)).*(x-d.center))
fromcanonical(d::SRay,x) = cisangle(d)*ray_fromcanonical(d.L,d.orientation,x)+d.center
fromcanonical(d::SRay{false},x) = ray_fromcanonical(d.L,d.orientation,x)+d.center
fromcanonical(d::SRay{true},x) = -ray_fromcanonical(d.L,d.orientation,x)+d.center
fromcanonicalD(d::SRay,x) = cisangle(d)*ray_fromcanonicalD(d.L,d.orientation,x)
invfromcanonicalD(d::SRay,x) = conj(cisangle(d))*ray_invfromcanonicalD(d.L,d.orientation,x)
arclength(d::SRay) = Inf

==(d::SRay{a},m::SRay{a}) where {a} = d.center == m.center && d.L == m.L


mappoint(a::SRay{false}, b::SRay{false}, x::Number) =
    b.center + b.L/a.L*(x - a.center)


function mappoint(a::SRay, b::SRay, x::Number)
    d = x - a.center;
    k = b.L/a.L * d * exp((angle(b)-angle(d))*im)
    k + b.center
end
