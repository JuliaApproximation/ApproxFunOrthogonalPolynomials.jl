

# project to interval if we are not on the interview
# TODO: need to work out how to set piecewise domain


scaleshiftdomain(f::Fun,sc,sh) = setdomain(f,sc*domain(f)+sh)

/(c::Number,f::Fun{Ultraspherical{λ,DD,RR}}) where {λ,DD,RR} = c/Fun(f,Chebyshev(domain(f)))
/(c::Number,f::Fun{<:PolynomialSpace{<:IntervalOrSegment}}) = c/Fun(f,Chebyshev(domain(f)))

/(c::Number,f::Fun{S}) where {S<:ContinuousSpace} = Fun(map(f->c/f,components(f)),PiecewiseSpace)
^(f::Fun{S},c::Integer) where {S<:ContinuousSpace} = Fun(map(f->f^c,components(f)),PiecewiseSpace)
^(f::Fun{S},c::Number) where {S<:ContinuousSpace} = Fun(map(f->f^c,components(f)),PiecewiseSpace)


/(c::Number,f::Fun{C}) where {C<:Chebyshev}=setdomain(c/setcanonicaldomain(f),domain(f))
function /(c::Number,f::Fun{Chebyshev{DD,RR}}) where {DD<:IntervalOrSegment,RR}
    fc = setcanonicaldomain(f)
    d=domain(f)
    # if domain f is small then the pts get projected in
    tol = 200eps(promote_type(typeof(c),cfstype(f)))*norm(f.coefficients,1)

    # we prune out roots at the boundary first
    if ncoefficients(f) == 1
        return Fun(c/f.coefficients[1],space(f))
    elseif ncoefficients(f) == 2
        if isempty(roots(f))
            return \(Multiplication(f,space(f)),c;tolerance=0.05tol)
        elseif isapprox(fc.coefficients[1],fc.coefficients[2])
            # we check directly for f*(1+x)
            return Fun(JacobiWeight(-1,0,space(f)),[c/fc.coefficients[1]])
        elseif isapprox(fc.coefficients[1],-fc.coefficients[2])
            # we check directly for f*(1-x)
            return Fun(JacobiWeight(0,-1,space(f)),[c/fc.coefficients[1]])
        else
            # we need to split at the only root
            return c/splitatroots(f)
        end
    elseif abs(first(fc)) ≤ tol
        #left root
        g=divide_singularity((1,0),fc)
        p=c/g
        x = Fun(identity,domain(p))
        return scaleshiftdomain(p/(1+x),complexlength(d)/2,mean(d) )
    elseif abs(last(fc)) ≤ tol
        #right root
        g=divide_singularity((0,1),fc)
        p=c/g
        x=Fun(identity,domain(p))
        return scaleshiftdomain(p/(1-x),complexlength(d)/2,mean(d) )
    else
        r = roots(fc)

        if length(r) == 0
            return \(Multiplication(f,space(f)),c;tolerance=0.05tol)
        elseif abs(last(r)+1.0)≤tol  # double check
            #left root
            g=divide_singularity((1,0),fc)
            p=c/g
            x=Fun(identity,domain(p))
            return scaleshiftdomain(p/(1+x),complexlength(d)/2,mean(d) )
        elseif abs(last(r)-1.0)≤tol  # double check
            #right root
            g=divide_singularity((0,1),fc)
            p=c/g
            x=Fun(identity,domain(p))
            return scaleshiftdomain(p/(1-x),complexlength(d)/2,mean(d) )
        else
            # no roots on the boundary
            return c/splitatroots(f)
        end
    end
end

^(f::Fun{<:PolynomialSpace},k::Integer) = intpow(f,k)
function ^(f::Fun{<:PolynomialSpace}, k::Real)
    T = cfstype(f)
    RT = real(T)
    # Need to think what to do if this is ever not the case..
    sp = space(f)
    fc = setcanonicaldomain(f) #Project to interval
    csp = space(fc)

    r = sort(roots(fc))
    #TODO divideatroots
    @assert length(r) <= 2

    if length(r) == 0
        setdomain(Fun((x->x^k) ∘ fc,csp),domain(f))  # using ∘ supports fast transforms for fc
    elseif length(r) == 1
        @assert isapprox(abs(r[1]),1)

        if isapprox(r[1], 1)
            Fun(JacobiWeight(zero(RT),k,sp),coefficients(divide_singularity(true,fc)^k,csp))
        else
            Fun(JacobiWeight(k,zero(RT),sp),coefficients(divide_singularity(false,fc)^k,csp))
        end
    else
        @assert isapprox(r[1],-1)
        @assert isapprox(r[2],1)

        Fun(JacobiWeight(k,k,sp),coefficients(divide_singularity(fc)^k,csp))
    end
end

#TODO: implement
^(f::Fun{Jacobi},k::Integer) = intpow(f,k)
^(f::Fun{Jacobi},k::Real) = Fun(f,Chebyshev)^k


# function log{MS<:MappedSpace}(f::Fun{MS})
#     g=log(Fun(f.coefficients,space(f).space))
#     Fun(g.coefficients,MappedSpace(domain(f),space(g)))
# end

# project first to [-1,1] to avoid issues with
# complex derivative
function log(f::Fun{<:PolynomialSpace{<:ChebyshevInterval}})
    r = sort(roots(f))
    #TODO divideatroots
    @assert length(r) <= 2

    if length(r) == 0
        cumsum(differentiate(f)/f)+log(first(f))
    elseif length(r) == 1
        @assert isapprox(abs(r[1]),1)

        if isapprox(r[1],1.)
            g=divide_singularity(true,f)
            lg=Fun(LogWeight(0.,1.,Chebyshev()),[1.])
            if isapprox(g,1.)  # this means log(g)~0
                lg
            else # log((1-x)) + log(g)
                lg⊕log(g)
            end
        else
            g=divide_singularity(false,f)
            lg=Fun(LogWeight(1.,0.,Chebyshev()),[1.])
            if isapprox(g,1.)  # this means log(g)~0
                lg
            else # log((1+x)) + log(g)
                lg⊕log(g)
            end
       end
    else
        @assert isapprox(r[1],-1)
        @assert isapprox(r[2],1)

        g=divide_singularity(f)
        lg=Fun(LogWeight(1.,1.,Chebyshev()),[1.])
        if isapprox(g,1.)  # this means log(g)~0
            lg
        else # log((1+x)) + log(g)
            lg⊕log(g)
        end
    end
end

function log(f::Fun{<:PolynomialSpace{<:IntervalOrSegment}})
    g = log(setdomain(f, ChebyshevInterval()))
    setdomain(g, domain(f))
end



# ODE gives the first order ODE a special function op satisfies,
# RHS is the right hand side
# growth says what to use to choose a good point to impose an initial condition
for (op,ODE,RHS,growth) in ((:(exp),"D-f'","0",:(real)),
                            (:(asinh),"sqrt(f^2+1)*D","f'",:(real)),
                            (:(acosh),"sqrt(f^2-1)*D","f'",:(real)),
                            (:(atanh),"(1-f^2)*D","f'",:(real)),
                            (:(erfcx),"D-2f*f'","-2f'/sqrt(π)",:(real)),
                            (:(dawson),"D+2f*f'","f'",:(real)))
    L,R = Meta.parse(ODE),Meta.parse(RHS)
    @eval $op(f::Fun{<:ContinuousSpace}) = Fun(map(f->$op(f),components(f)),PiecewiseSpace)
end


for OP in (:abs,:sign,:log,:angle)
    @eval $OP(f::Fun{<:ContinuousSpace{<:Any,<:Real},<:Real}) =
            Fun(map($OP,components(f)),PiecewiseSpace)
end

# JacobiWeight explodes, we want to ensure the solution incorporates the fact
# that exp decays rapidly
exp(f::Fun{<:JacobiWeight}) = setdomain(exp(setdomain(f, ChebyshevInterval())), domain(f))
function exp(f::Fun{<:JacobiWeight{<:Any,<:ChebyshevInterval}})
    S=space(f)
    q=Fun(S.space,f.coefficients)
    if isapprox(S.α,0.) && isapprox(S.β,0.)
        exp(q)
    elseif S.β < 0 && isapprox(first(q),0.)
        # this case can remove the exponential decay
        exp(Fun(f,JacobiWeight(S.β+1,S.α,S.space)))
    elseif S.α < 0 && isapprox(last(q),0.)
        exp(Fun(f,JacobiWeight(S.β,S.α+1,S.space)))
    elseif S.β > 0 && isapproxinteger(S.β)
        exp(Fun(f,JacobiWeight(0.,S.α,S.space)))
    elseif S.α > 0 && isapproxinteger(S.α)
        exp(Fun(f,JacobiWeight(S.β,0.,S.space)))
    else
        #find normalization point
        xmax,opfxmax,opmax=specialfunctionnormalizationpoint(exp,real,f)

        if S.α < 0 && S.β < 0
            # provided both are negative, we get exponential decay on both ends
            @assert real(first(q)) < 0 && real(last(q)) < 0
            s=JacobiWeight(2.,2.,domain(f))
        elseif S.β < 0 && isapprox(S.α,0.)
            @assert real(first(q)) < 0
            s=JacobiWeight(2.,0.,domain(f))
        elseif S.α < 0 && isapprox(S.β,0.)
            @assert real(last(q)) < 0
            s=JacobiWeight(0.,2.,domain(f))
        else
            error("exponential of fractional power, not implemented")
        end

        D=Derivative(s)
        B=Evaluation(s,xmax)

        \([B,D-f'],Any[opfxmax,0.];tolerance=eps(cfstype(f))*opmax)
    end
end

sin(f::Fun{S,T}) where {S<:Union{Ultraspherical,Chebyshev},T<:Real} = imag(exp(im*f))
cos(f::Fun{S,T}) where {S<:Union{Ultraspherical,Chebyshev},T<:Real} = real(exp(im*f))


## Second order functions with parameter ν

for (op,ODE,RHS,growth) in ((:(hankelh1),"f^2*f'*D^2+(f*f'^2-f^2*f'')*D+(f^2-ν^2)*f'^3","0",:(imag)),
                            (:(hankelh2),"f^2*f'*D^2+(f*f'^2-f^2*f'')*D+(f^2-ν^2)*f'^3","0",:(imag)),
                            (:(besselj),"f^2*f'*D^2+(f*f'^2-f^2*f'')*D+(f^2-ν^2)*f'^3","0",:(imag)),
                            (:(bessely),"f^2*f'*D^2+(f*f'^2-f^2*f'')*D+(f^2-ν^2)*f'^3","0",:(imag)),
                            (:(besseli),"f^2*f'*D^2+(f*f'^2-f^2*f'')*D-(f^2+ν^2)*f'^3","0",:(real)),
                            (:(besselk),"f^2*f'*D^2+(f*f'^2-f^2*f'')*D-(f^2+ν^2)*f'^3","0",:(real)),
                            (:(besselkx),"f^2*f'*D^2+((-2f^2+f)*f'^2-f^2*f'')*D-(f+ν^2)*f'^3","0",:(real)),
                            (:(hankelh1x),"f^2*f'*D^2+((2im*f^2+f)*f'^2-f^2*f'')*D+(im*f-ν^2)*f'^3","0",:(imag)),
                            (:(hankelh2x),"f^2*f'*D^2+((-2im*f^2+f)*f'^2-f^2*f'')*D+(-im*f-ν^2)*f'^3","0",:(imag)))
    L,R = Meta.parse(ODE),Meta.parse(RHS)
    @eval begin
        function $op(ν,fin::Fun{S,T}) where {S<:Union{Ultraspherical,Chebyshev},T}
            f=setcanonicaldomain(fin)

            g=chop($growth(f),eps(T))
            xmin = isempty(g.coefficients) ? leftendpoint(domain(g)) : argmin(g)
            xmax = isempty(g.coefficients) ? rightendpoint(domain(g)) : argmax(g)
            opfxmin,opfxmax = $op(ν,f(xmin)),$op(ν,f(xmax))
            opmax = maximum(abs,(opfxmin,opfxmax))
            while opmax≤10eps(T) || abs(f(xmin)-f(xmax))≤10eps(T)
                xmin,xmax = rand(domain(f)),rand(domain(f))
                opfxmin,opfxmax = $op(ν,f(xmin)),$op(ν,f(xmax))
                opmax = maximum(abs,(opfxmin,opfxmax))
            end
            D=Derivative(space(f))
            B=[Evaluation(space(f),xmin),Evaluation(space(f),xmax)]
            u=\([B;eval($L)],[opfxmin;opfxmax;eval($R)];tolerance=eps(T)*opmax)

            setdomain(u,domain(fin))
        end
    end
end


#TODO ≤,≥




## Piecewise Space

# Return the locations of jump discontinuities
#
# Non Piecewise Spaces are assumed to have no jumps.
function jumplocations(f::Fun)
    eltype(domain(f))[]
end

# Return the locations of jump discontinuities
function jumplocations(f::Fun{S}) where{S<:Union{PiecewiseSpace,ContinuousSpace}}
    d = domain(f)

    if ncomponents(d) < 2
      return eltype(domain(f))[]
    end

    dtol=10eps(eltype(d))
    ftol=10eps(cfstype(f))

    dc = components(d)
    fc = components(f)

    isjump = isapprox.(leftendpoint.(dc[2:end]), rightendpoint.(dc[1:end-1]), rtol=dtol) .&
           .!isapprox.(first.(fc[2:end]), last.(fc[1:end-1]), rtol=ftol)

    locs = rightendpoint.(dc[1:end-1])
    locs[isjump]
end



