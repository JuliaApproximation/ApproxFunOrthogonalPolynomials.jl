using SnoopPrecompile

@precompile_setup begin
    splist = Any[Jacobi(1, 1), Chebyshev(), Ultraspherical(1)]
    append!(splist, Any[Jacobi(1, 1, 0..1), Chebyshev(0..1), Ultraspherical(1, 0..1)])
    push!(splist, Chebyshev(Segment(1.0+im,2.0+2im)))
    a = Fun(ChebyshevInterval{BigFloat}(),BigFloat[1,2,3])
    @precompile_all_calls begin
        for S in splist
        	v = [0.0, 1.0]
        	f = Fun(S,v)
        	abs(DefiniteIntegral()*f - sum(f))
        	norm(Derivative()*f-f')
        	norm(differentiate(integrate(f)) - f)
        	norm(differentiate(cumsum(f))-f)
        	norm(first(cumsum(f)))
        end
        Fun(sin,Interval(big"0.0", big"1.0")
    end
end