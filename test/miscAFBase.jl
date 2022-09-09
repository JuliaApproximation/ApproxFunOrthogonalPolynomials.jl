using ApproxFunBase
@testset "ApproxFunOrthogonalPolynomials" begin
    @test (@inferred Fun()) == Fun(x->x, Chebyshev())
    @test (@inferred norm(Fun())) ≈ norm(Fun(), 2) ≈ √(2/3) # √∫x^2 dx over -1..1

    v = rand(4)
    v2 = transform(NormalizedChebyshev(), v)
    @test itransform(NormalizedChebyshev(), v2) ≈ v

    f = @inferred Fun(x->x^2, Chebyshev())
    v = @inferred coefficients(f, Chebyshev(), Legendre())
    @test eltype(v) == eltype(coefficients(f))
    @test v ≈ coefficients(Fun(x->x^2, Legendre()))

    # inference check for coefficients
    v = @inferred coefficients(Float64[0,0,1], Chebyshev(), Ultraspherical(1))
    @test v ≈ [-0.5, 0, 0.5]

    @testset "int coeffs" begin
        f = Fun(Chebyshev(), [0,1])
        @test f(0.4) ≈ 0.4
        f = Fun(NormalizedChebyshev(), [0,1])
        @test f(0.4) ≈ 0.4 * √(2/pi)

        f = Fun(Chebyshev(), [1])
        @test f(0.4) ≈ 1
        f = Fun(NormalizedChebyshev(), [1])
        @test f(0.4) ≈ √(1/pi)
    end

    @testset "pad" begin
        @testset "Fun" begin
            f = Fun()
            zf = zero(f)
            @test (@inferred pad([f], 3)) == [f, zf, zf]
            @test (@inferred pad([f, zf], 1)) == [f]
            v = [f, zf]
            @test @inferred pad!(v, 1) == [f]
            @test length(v) == 1
        end
    end

    @testset "inplace transform" begin
        @testset for sp_c in Any[Legendre(), Chebyshev(), Jacobi(1,2), Jacobi(0.3, 2.3),
                Ultraspherical(1), Ultraspherical(2)]
            @testset for sp in Any[sp_c, NormalizedPolynomialSpace(sp_c)]
                v = rand(10)
                v2 = copy(v)
                @test itransform!(sp, transform!(sp, v)) ≈ v
                @test transform!(sp, v) ≈ transform(sp, v2)
                @test itransform(sp, v) ≈ v2
                @test itransform!(sp, v) ≈ v2

                # different vector
                p_fwd = ApproxFunBase.plan_transform!(sp, v)
                p_inv = ApproxFunBase.plan_itransform!(sp, v)
                @test p_inv * copy(p_fwd * copy(v)) ≈ v
            end
        end
    end

    @testset "conversion" begin
        C12 = Conversion(Chebyshev(), NormalizedLegendre())
        C21 = Conversion(NormalizedLegendre(), Chebyshev())
        @test Matrix((C12 * C21)[1:10, 1:10]) ≈ I
        @test Matrix((C21 * C12)[1:10, 1:10]) ≈ I

        C12 = Conversion(Chebyshev(), NormalizedPolynomialSpace(Ultraspherical(1)))
        C1C2 = Conversion(Ultraspherical(1), NormalizedPolynomialSpace(Ultraspherical(1))) *
                Conversion(Chebyshev(), Ultraspherical(1))
        @test Matrix(C12[1:10, 1:10]) ≈ Matrix(C1C2[1:10, 1:10])
    end

    @testset "union" begin
        @test union(Chebyshev(), NormalizedLegendre()) == Jacobi(Chebyshev())
        @test union(Chebyshev(), Legendre()) == Jacobi(Chebyshev())
    end

    @testset "Fun constructor" begin
        # we make the fun go through somewhat complicated chains of functions
        # that break inference of the space
        # however, the type of coefficients should be inferred correctly.
        f = Fun(Chebyshev(0..1))
        newfc(f) = coefficients(Fun(Fun(f, Legendre(0..1)), space(f)))
        newvals(f) = values(Fun(Fun(f, Legendre(0..1)), space(f)))
        @test newfc(f) ≈ coefficients(f)
        @test newvals(f) ≈ values(f)

        newfc2(f) = coefficients(chop(pad(f, 10)))
        @test newfc2(f) == coefficients(f)

        f2 = Fun(space(f), view(Float64[1:4;], :))
        f3 = Fun(space(f), Float64[1:4;])
        @test newvals(f2) ≈ values(f3)
        @test values(f2) ≈ values(f3)

        # Ensure no trailing zeros
        f = Fun(Ultraspherical(0.5, 0..1))
        cf = coefficients(f)
        @test findlast(!iszero, cf) == length(cf)

        @testset "OneHotVector" begin
            for n in [1, 3, 10_000]
                f = Fun(Chebyshev(), [zeros(n-1); 1])
                g = ApproxFunBase.basisfunction(Chebyshev(), n)
                @test f == g
                @test f(0.5) == g(0.5)
            end
        end
    end

    @testset "multiplication of Funs" begin
        f = Fun(Chebyshev(), Float64[1:101;])
        g = Fun(Chebyshev(), Float64[1:101;]*im)
        @test f(0.5)*g(0.5) ≈ (f*g)(0.5)
    end

    @testset "Multivariate" begin
        @testset for S in Any[Chebyshev(), Legendre()]
            f = Fun(x->ones(2,2), S)
            @test (f+1) * f ≈ (1+f) * f ≈ f^2 + f
            @test (f-1) * f ≈ f^2 - f
            @test (1-f) * f ≈ f - f^2
            @test f + f ≈ 2f ≈ f*2
        end
    end

    @testset "static coeffs" begin
        f = Fun(Chebyshev(), SA[1,2,3])
        g = Fun(Chebyshev(), [1,2,3])
        @test coefficients(f^2) == coefficients(g^2)
    end

    @testset "special functions" begin
        for f in Any[Fun(), Fun(-0.5..1), Fun(Segment(1.0+im,2.0+2im))]
            for spfn in Any[sin, cos, exp]
                p = leftendpoint(domain(f))
                @test spfn(f)(p) ≈ spfn(p) atol=1e-14
            end
        end
    end

    @testset "Derivative" begin
        @test Derivative() == Derivative()
        for d in Any[(), (0..1,)]
            for ST in Any[Chebyshev, Legendre,
                    (x...) -> Jacobi(2,2,x...), (x...) -> Jacobi(1.5,2.5,x...)]
                S1 = ST(d...)
                for S in [S1, NormalizedPolynomialSpace(S1)]
                    @test Derivative(S) == Derivative(S,1)
                    @test Derivative(S)^2 == Derivative(S,2)
                    f = Fun(x->x^3, S)
                    @test Derivative(S) * f ≈ Fun(x->3x^2, S)
                    @test Derivative(S,2) * f ≈ Fun(x->6x, S)
                    @test Derivative(S,3) * f ≈ Fun(x->6, S)
                    @test Derivative(S,4) * f ≈ zeros(S)
                end
            end
        end
        @test Derivative(Chebyshev()) != Derivative(Chebyshev(), 2)
        @test Derivative(Chebyshev()) != Derivative(Legendre())
    end

    @testset "SubOperator" begin
        D = Derivative(Chebyshev())
        S = @view D[1:10, 1:10]
        @test rowrange(S, 1) == 2:2
        @test colrange(S, 2) == 1:1
        @test (@inferred BandedMatrix(S)) == (@inferred Matrix(S))
    end

    @testset "CachedOperator" begin
        C = cache(Derivative())
        C = C : Chebyshev() → Ultraspherical(2)
        D = Derivative() : Chebyshev() → Ultraspherical(2)
        @test C[1:2, 1:0] == D[1:2, 1:0]
        @test C[1:10, 1:10] == D[1:10, 1:10]
        for col in 1:5, row in 1:5
            @test C[row, col] == D[row, col]
        end
    end

    @testset "PartialInverseOperator" begin
        @testset "sanity check" begin
            A = UpperTriangular(rand(10, 10))
            B = inv(A)
            for I in CartesianIndices(B)
                @test B[I] ≈ ApproxFunBase._getindexinv(A, Tuple(I)..., UpperTriangular)
            end
        end
        C = Conversion(Chebyshev(), Ultraspherical(1))
        P = PartialInverseOperator(C, (0, 6))
        Iapprox = (P * C)[1:10, 1:10]
        @test all(isone, diag(Iapprox))
        for k in axes(Iapprox,1), j in k + 1:min(k + bandwidths(P,2), size(Iapprox, 2))
            @test Iapprox[k,j] ≈ 0 atol=eps(eltype(Iapprox))
        end
        B = AbstractMatrix(P[1:10, 1:10])
        @testset for I in CartesianIndices(B)
            @test B[I] ≈ P[Tuple(I)...] rtol=1e-8 atol=eps(eltype(B))
        end
    end

    @testset "istriu/istril" begin
        for D in Any[Derivative(Chebyshev()),
                Conversion(Chebyshev(), Legendre()),
                Multiplication(Fun(Chebyshev()), Chebyshev())]
            D2 = D[1:3, 1:3]
            for f in Any[istriu, istril]
                @test f(D) == f(D2)
                @test f(D') == f(D2')
            end
        end
    end

    @testset "inplace ldiv" begin
        @testset for T in [Float32, Float64, ComplexF32, ComplexF64]
            v = rand(T, 4)
            v2 = copy(v)
            ApproxFunBase.ldiv_coefficients!(Conversion(Chebyshev(), Ultraspherical(1)), v)
            @test ApproxFunBase.ldiv_coefficients(Conversion(Chebyshev(), Ultraspherical(1)), v2) ≈ v
        end
    end

    @testset "specialfunctionnormalizationpoint" begin
        a = @inferred ApproxFunBase.specialfunctionnormalizationpoint(exp,real,Fun())
        @test a[1] == 1
        @test a[2] ≈ exp(1)
    end
end