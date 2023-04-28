using PrecompileTools

@setup_workload begin
    splist = Any[Jacobi(1, 1), Chebyshev(), Ultraspherical(1)]
    append!(splist, Any[Jacobi(1, 1, 0..1), Chebyshev(0..1), Ultraspherical(1, 0..1)])
    spreal = copy(splist)
    push!(splist, Chebyshev(Segment(1.0+im,2.0+2im)))
    # special functions
    spfns = (sin, cos, exp)
    v = ones(2)
    m = ones(2,2)
    a = ones(2,2,2)
    @compile_workload begin
        for S in splist
        	f = Fun(S,v)
            f(0.1)
            1/(f^2+1)
        	abs(DefiniteIntegral()*f - sum(f))
        	norm(Derivative()*f-f')
        	norm(first(cumsum(f)))
            for spfn in spfns
                spfn(f)
            end
        end
        for S in spreal
            transform(S, v)
            itransform(S, v)
            transform!(S, v)
            itransform!(S, v)
            transform(S * S, m)
            itransform(S * S, m)

            [Derivative(S)^2 + 1; Dirichlet(S)] \ [1,0]

            f2 = Fun((x,y) -> x*y, S^2); f2(0.1, 0.1)
            P = ProductFun((x,y) -> x*y, S^2); P(0.1, 0.1)
            L = LowRankFun((x,y) -> x*y, S^2); L(0.1, 0.1)
        end
        roots(Fun(sin,Interval(big"0.0", big"1.0")))
    end
end
