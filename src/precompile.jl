using SnoopPrecompile

@precompile_setup begin
    splist = Any[Jacobi(1, 1), Chebyshev(), Ultraspherical(1)]
    append!(splist, Any[Jacobi(1, 1, 0..1), Chebyshev(0..1), Ultraspherical(1, 0..1)])
    push!(splist, Chebyshev(Segment(1.0+im,2.0+2im)))
    # special functions
    spfns = (sin, cos, exp)
    v = ones(2)
    @precompile_all_calls begin
        for S in splist
        	f = Fun(S,v)
            1/(f^2+1)
        	abs(DefiniteIntegral()*f - sum(f))
        	norm(Derivative()*f-f')
        	norm(first(cumsum(f)))
            for spfn in spfns
                spfn(f)
            end
        end
        roots(Fun(sin,Interval(big"0.0", big"1.0")))
    end
end
