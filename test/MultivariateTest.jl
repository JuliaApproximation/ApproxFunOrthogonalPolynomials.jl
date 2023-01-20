using ApproxFunBase
using ApproxFunOrthogonalPolynomials
using LinearAlgebra
using SpecialFunctions
using BlockBandedMatrices
using Test
using ApproxFunBase: factor, Block, cfstype, blocklengths, block, tensorizer, ArraySpace, ∞
using ApproxFunBaseTest: testbandedblockbandedoperator, testraggedbelowoperator,
                    testblockbandedoperator
using ApproxFunOrthogonalPolynomials: chebyshevtransform
using StaticArrays: SVector

using Base: oneto

@verbose @testset "Multivariate" begin
    @testset "vectorization order" begin
        @testset "2D trivial" begin
            S = Chebyshev()^2
            it = tensorizer(S)
            expected_order = [(1, 1)
                            (1,2)
                            (2,1)
                            (1,3)
                            (2,2)
                            (3,1)
                            (1,4)
                            (2,3)]

            for (k,i) in enumerate(it)
                if k>length(expected_order)
                    break
                end
                @test i == expected_order[k]
            end
        end

        @testset "3D trivial" begin
            S = Chebyshev()^3
            it = tensorizer(S)
            expected_order = [(1, 1, 1)
                            (1,1,2)
                            (1,2,1)
                            (2,1,1)
                            (1,1,3)
                            (1,2,2)
                            (2,1,2)
                            (1,3,1)
                            (2,2,1)
                            (3,1,1)
                            (1,1,4)
                            (1,2,3)
                            (2,1,3)
                            (1,3,2)
                            (2,2,2)
                            (3,1,2)
                            (1,4,1)
                            (2,3,1)
                            (3,2,1)
                            (4,1,1)]

            for (k, i) in enumerate(it)
                if k>length(expected_order)
                    break
                end
                @test Tuple(i) == expected_order[k]
            end
        end
    end

    @testset "Evaluation" begin

        @testset "2D" begin
            f2 = Fun(Chebyshev()^2, [1.0])
            @test f2(0.2, 0.4) == 1.0
        end

        @testset "3D" begin
            f3 = Fun(Chebyshev()^3, [1.0])
            @test f3(0.2, 0.4, 0.2) == 1.0
        end

        @testset "20D" begin
            f20 = Fun(Chebyshev()^20, [1.0])
            @test f20(rand(20)) == 1.0
        end
    end

    @testset "Square" begin
        S = Space(ChebyshevInterval()^2)
        @test @inferred(blocklengths(S)) ≡ oneto(∞)

        @test block(tensorizer(S), 1) == Block(1)

        @time for k=0:5,j=0:5
            ff=(x,y)->cos(k*acos(x))*cos(j*acos(y))
            f=Fun(ff,ChebyshevInterval()^2)
            @test f(0.1,0.2) ≈ ff(0.1,0.2)
        end

        @time for k=0:5,j=0:5
            ff = (x,y)->cos(k*acos(x/2))*cos(j*acos(y/2))
            f=Fun(ff,Interval(-2,2)^2)
            @test f(0.1,0.2) ≈ ff(0.1,0.2)
        end


        @time for k=0:5,j=0:5
            ff=(x,y)->cos(k*acos(x-1))*cos(j*acos(y-1))
            f=Fun(ff,Interval(0,2)^2)
            @test f(0.1,0.2) ≈ ff(0.1,0.2)
        end

        ## Try constructor variants

        ff=(x,y)->exp(-10(x+.2)^2-20(y-.1)^2)*cos(x*y)
        gg=x->exp(-10(x[1]+.2)^2-20(x[1]-.1)^2)*cos(x[1]*x[2])
        f=Fun(ff,ChebyshevInterval()^2,10000)
        @test f(0.,0.) ≈ ff(0.,0.)


        f=Fun(gg,ChebyshevInterval()^2,10000)
        @test f(0.,0.) ≈ ff(0.,0.)

        f=Fun(ff,ChebyshevInterval()^2)
        @test f(0.,0.) ≈ ff(0.,0.)
        f=Fun(gg,ChebyshevInterval()^2)
        @test f(0.,0.) ≈ ff(0.,0.)

        f=Fun(ff)
        @test f(0.,0.) ≈ ff(0.,0.)
        f=Fun(gg)
        @test f(0.,0.) ≈ ff(0.,0.)

    end

    @testset "Arithmetic" begin
        # Fun +-* constant
        f=Fun((x,y)->exp(x)*cos(y))

        @test f(0.1,0.2)+2 ≈ (f+2)(0.1,0.2)
        @test f(0.1,0.2)-2 ≈ (f-2)(0.1,0.2)
        @test f(0.1,0.2)*2 ≈ (f*2)(0.1,0.2)

        @testset "Addition" begin
            # coefficients
             c_1 = rand(20)
             c_2 = rand(30)

             added_coef = [c_2[1:20]+c_1;c_2[21:end]]

             # 2D
             f2_1 = Fun(Chebyshev()^2, c_1)
             f2_2 = Fun(Chebyshev()^2, c_2)
             @test coefficients(f2_1+f2_2) == added_coef

             @test (f2_1+f2_2)(0.3, 0.5)≈f2_1(0.3, 0.5)+f2_2(0.3, 0.5)

             # 3D
             f3_1 = Fun(Chebyshev()^3, c_1)
             f3_2 = Fun(Chebyshev()^3, c_2)
             @test coefficients(f3_1+f3_2) == added_coef

             @test (f3_1+f3_2)(0.3, 0.5, 0.6)≈f3_1(0.3, 0.5, 0.6)+f3_2(0.3, 0.5, 0.6)
         end

         @testset "Multiplication" begin
             # coefficients
             c_1 = rand(20)
             c_2 = rand(30)

             # 2D
             f2_1 = Fun(Chebyshev()^2, c_1)
             f2_2 = Fun(Chebyshev()^2, c_2)

             @test (f2_1 * f2_2)(0.4, 0.5) ≈ f2_1(0.4, 0.5) * f2_2(0.4, 0.5)

             # 3D: not implemented in code yet
             #f3_1 = Fun(Chebyshev()^3, c_1)
             #f3_2 = Fun(Chebyshev()^3, c_2)

             #@test (f3_1*f3_2)(0.4,0.5,0.6) ≈ f3_1(0.4,0.5,0.6)*f3_2(0.4,0.5,0.6)
         end
    end

    @testset "LowRankFun" begin
        @time F = @inferred LowRankFun((x,y)->besselj0(10(y-x)),Chebyshev(),Chebyshev())

        @test F(.123,.456) ≈ besselj0(10(.456-.123))

        @time G = @inferred LowRankFun((x,y)->besselj0(10(y-x));method=:Cholesky)

        @test G(.357,.246) ≈ besselj0(10(.246-.357))

        # test "fast" grid evaluation of LowRankFun
        f = @inferred LowRankFun((x,y) -> exp(x) * cos(y)); n = 1000
        x = range(-1, stop=1, length=n); y = range(-1, stop=1, length=n)
        X = x * fill(1.0,1,n); Y = fill(1.0, n) * y'
        @time v1 = f.(X, Y);
        @time v2 = f.(x, y');
        @test v1 ≈ v2

        # ensure that all coefficients are captured
        L = LowRankFun((x,y) -> x*y, Chebyshev() ⊗ Chebyshev())
        F = Fun(L)
        @test L(0.1, 0.2) ≈ F(0.1, 0.2)

        x = Fun(); y = x;
        D1 = Derivative(space(L), [1,0])
        D2 = Derivative(space(L), [0,1])
        @test (D2 * L)(0.1, 0.2) ≈ x(0.1)
        @test (D1 * L)(0.1, 0.2) ≈ y(0.2)
        @test ((L * D1) * L)(0.1, 0.2) ≈ x(0.1) * (y(0.2))^2
        @test ((L * D2) * L)(0.1, 0.2) ≈ (x(0.1))^2 * y(0.2)
        @test (D1[L] * L)(0.1, 0.2) ≈ 2x(0.1) * y(0.2)^2
        @test ((Derivative() ⊗ I) * L)(0.1, 0.2) ≈ y(0.2)
        @test ((I ⊗ Derivative()) * L)(0.1, 0.2) ≈ x(0.1)
    end


    @testset  "SVector segment" begin
        d = Segment(SVector(0.,0.) , SVector(1.,1.))
        x = Fun()
        @test (ApproxFunBase.complexlength(d)*x/2)(0.1)  ≈ (d.b - d.a)*0.1/2
        @test ApproxFunBase.fromcanonical(d,x)(0.1) ≈ (d.b+d.a)/2 + (d.b - d.a)*0.1/2

        x,y = Fun(Segment(SVector(0.,0.) , SVector(2.,1.)))
        @test x(0.2,0.1) ≈ 0.2
        @test y(0.2,0.1) ≈ 0.1

        d=Segment((0.,0.),(1.,1.))
        f=Fun(xy->exp(-xy[1]-2cos(xy[2])),d)
        @test f(0.5,0.5) ≈ exp(-0.5-2cos(0.5))
        @test f(SVector(0.5,0.5)) ≈ exp(-0.5-2cos(0.5))

        f=Fun(xy->exp(-xy[1]-2cos(xy[2])),d,20)
        @test f(0.5,0.5) ≈ exp(-0.5-2cos(0.5))

        f=Fun((x,y)->exp(-x-2cos(y)),d)
        @test f(0.5,0.5) ≈ exp(-0.5-2cos(0.5))

        f=Fun((x,y)->exp(-x-2cos(y)),d,20)
        @test f(0.5,0.5) ≈ exp(-0.5-2cos(0.5))
    end

    @testset "Multivariate calculus" begin
        ## Sum
        ff = (x,y) -> (x-y)^2*exp(-x^2/2-y^2/2)
        f=Fun(ff, (-4..4)^2)
        @test f(0.1,0.2) ≈ ff(0.1,0.2)

        @test sum(f,1)(0.1) ≈ 2.5162377980828357
        f=LowRankFun(f)
        @test evaluate(f.A,0.1) ≈ map(f->f(0.1),f.A)
    end

    @testset "KroneckerOperator" begin
        Mx = Multiplication(Fun(cos),Chebyshev())
        My = Multiplication(Fun(sin),Chebyshev())
        K = Mx⊗My

        @test BandedBlockBandedMatrix(view(K,1:10,1:10)) ≈ [K[k,j] for k=1:10,j=1:10]
        C = Conversion(Chebyshev()⊗Chebyshev(),Ultraspherical(1)⊗Ultraspherical(1))
        @test C[1:100,1:100] ≈ Float64[C[k,j] for k=1:100,j=1:100]
    end

    @testset "Partial derivative operators" begin
        d = Space(0..1) * Space(0..2)
        Dx = Derivative(d, [1,0])
        testbandedblockbandedoperator(Dx)
        f = Fun((x,y) -> sin(x) * cos(y), d)
        fx = Fun((x,y) -> cos(x) * cos(y), d)
        @test (Dx*f)(0.2,0.3) ≈ fx(0.2,0.3)
        Dy = Derivative(d, [0,1])
        testbandedblockbandedoperator(Dy)
        fy = Fun((x,y) -> -sin(x) * sin(y), d)
        @test (Dy*f)(0.2,0.3) ≈ fy(0.2,0.3)
        L = Dx + Dy
        testbandedblockbandedoperator(L)
        @test (L*f)(0.2,0.3) ≈ (fx(0.2,0.3)+fy(0.2,0.3))

        B=ldirichlet(factor(d,1))⊗ldirichlet(factor(d,2))
        @test abs(Number(B*f)-f(0.,0.)) ≤ 10eps()

        B=Evaluation(factor(d,1),0.1)⊗ldirichlet(factor(d,2))
        @test Number(B*f) ≈ f(0.1,0.)

        B=Evaluation(factor(d,1),0.1)⊗Evaluation(factor(d,2),0.3)
        @test Number(B*f) ≈ f(0.1,0.3)

        B=Evaluation(d,(0.1,0.3))
        @test Number(B*f) ≈ f(0.1,0.3)
    end

    @testset "x,y constructors" begin
        d=ChebyshevInterval()^2

        sp = ArraySpace(d,2)
        @test blocklengths(sp) == 2:2:∞
        @test block(ArraySpace(d,2),1) == Block(1)

        x,y=Fun(d)
        @test x(0.1,0.2) ≈ 0.1
        @test y(0.1,0.2) ≈ 0.2

        x,y=Fun(identity, d, 20)
        @test x(0.1,0.2) ≈ 0.1
        @test y(0.1,0.2) ≈ 0.2


        # Boundary

        x,y=Fun(identity, ∂(d), 20)
        @test x(0.1,1.0) ≈ 0.1
        @test y(1.0,0.2) ≈ 0.2


        x,y=Fun(identity, ∂(d))
        @test x(0.1,1.0) ≈ 0.1
        @test y(1.0,0.2) ≈ 0.2


        x,y=Fun(∂(d))
        @test x(0.1,1.0) ≈ 0.1
        @test y(1.0,0.2) ≈ 0.2
    end

    @testset "conversion between" begin
        dx = dy = ChebyshevInterval()
        d = dx × dy
        x,y=Fun(d)
        @test x(0.1,0.2) ≈ 0.1
        @test y(0.1,0.2) ≈ 0.2

        @test ∂(d) isa PiecewiseSegment
        x,y = Fun(∂(d))
        x,y = components(x),components(y)
        @test (x[1]-1im)(0.1,-1.0) ≈ 0.1-im
        g = [real(exp(x[1]-1im)); 0.0y[2]; real(exp(x[3]+1im)); real(exp(-1+1im*y[4]))]
        B = [ Operator(I,dx)⊗ldirichlet(dy);
             ldirichlet(dx)⊗Operator(I,dy);
             Operator(I,dx)⊗rdirichlet(dy);
             rneumann(dx)⊗Operator(I,dy)    ]

        @test Fun(g[1],rangespace(B)[1])(-0.1,-1.0) ≈ g[1](-0.1,-1.0)
        @test Fun(g[3],rangespace(B)[3])(-0.1,1.0)  ≈ g[3](-0.1,1.0)

        A = [B; Laplacian()]

        @test cfstype([g;0.0]) == Float64
        g2 = Fun([g;0.0],rangespace(A))
        @test cfstype(g2) == Float64

        @test g2[1](-0.1,-1.0) ≈ g[1](-0.1,-1.0)
        @test g2[3](-0.1,1.0)  ≈ g[3](-0.1,1.0)
    end

    @testset "Cheby * Interval" begin
        d = ChebyshevInterval()^2
        x,y = Fun(∂(d))

        @test ApproxFunBase.rangetype(Space(∂(d))) == Float64
        @test ApproxFunBase.rangetype(space(y)) == Float64

        @test (im*y)(1.0,0.1) ≈ 0.1im
        @test (x+im*y)(1.0,0.1) ≈ 1+0.1im

        @test exp(x+im*y)(1.0,0.1) ≈ exp(1.0+0.1im)
    end

    @testset "DefiniteIntegral" begin
        f = Fun((x,y) -> exp(-x*cos(y)))
        @test Number(DefiniteIntegral()*f) ≈ sum(f)
    end

    @testset "Piecewise Tensor" begin
        a = Fun(0..1) + Fun(2..3)
        f = a ⊗ a
        @test f(0.1,0.2) ≈ 0.1*0.2
        @test f(1.1,0.2) ≈ 0
        @test f(2.1,0.2) ≈ 2.1*0.2

        @test component(space(f),1,1) == Chebyshev(0..1)^2
        @test component(space(f),1,2) == Chebyshev(0..1)*Chebyshev(2..3)
        @test component(space(f),2,1) == Chebyshev(2..3)*Chebyshev(0..1)
        @test component(space(f),2,2) == Chebyshev(2..3)^2
    end

    @testset "Bug in chop of ProductFun" begin
        u = Fun(Chebyshev()^2,[0.0,0.0])
        @test coefficients(chop(ProductFun(u),10eps())) == zeros(0,1)


        d= (-1..1)^2
        B=[Dirichlet(factor(d,1))⊗I;I⊗ldirichlet(factor(d,2));I⊗rneumann(factor(d,2))]
        Δ=Laplacian(d)

        rs = rangespace([B;Δ])
        f = Fun((x,y)->exp(-x^2-y^2),d)
        @test_throws DimensionMismatch coefficients([0.0;0.0;0.0;0.0;f],rs)
    end

    @testset "off domain evaluate" begin
        g = Fun(1, Segment(SVector(0,-1) , SVector(π,-1)))
        @test g(0.1,-1) ≈ 1
        @test g(0.1,1) ≈ 0
    end

    @testset "Dirichlet" begin
        testblockbandedoperator(@inferred Dirichlet((0..1)^2))
        testblockbandedoperator(@inferred Dirichlet((0..1) × (0.0 .. 1)))
        testraggedbelowoperator(Dirichlet(Chebyshev()^2))
        testraggedbelowoperator(Dirichlet(Chebyshev(0..1) * Chebyshev(0.0..1)))
    end

    @testset "2d derivative (issue #346)" begin
        d = Chebyshev()^2
        f = Fun((x,y) -> sin(x) * cos(y), d)
        C=Conversion(Chebyshev()⊗Chebyshev(),Ultraspherical(1)⊗Ultraspherical(1))
        @test (C*f)(0.1,0.2) ≈ f(0.1,0.2)
        Dx = Derivative(d, [1,0])
        f = Fun((x,y) -> sin(x) * cos(y), d)
        fx = Fun((x,y) -> cos(x) * cos(y), d)
        @test (Dx*f)(0.2,0.3) ≈ fx(0.2,0.3)
        Dy = Derivative(d, [0,1])
        fy = Fun((x,y) -> -sin(x) * sin(y), d)
        @test (Dy*f)(0.2,0.3) ≈ fy(0.2,0.3)
        L=Dx+Dy
        testbandedblockbandedoperator(L)

        @test (L*f)(0.2,0.3) ≈ (fx(0.2,0.3)+fy(0.2,0.3))

        B=ldirichlet(factor(d,1))⊗ldirichlet(factor(d,2))
        @test Number(B*f) ≈ f(-1.,-1.)

        B=Evaluation(factor(d,1),0.1)⊗ldirichlet(factor(d,2))
        @test Number(B*f) ≈ f(0.1,-1.)

        B=Evaluation(factor(d,1),0.1)⊗Evaluation(factor(d,2),0.3)
        @test Number(B*f) ≈ f(0.1,0.3)

        B=Evaluation(d,(0.1,0.3))
        @test Number(B*f) ≈ f(0.1,0.3)
    end

    @testset "ProductFun" begin
        u0   = ProductFun((x,y)->cos(x)+sin(y) +exp(-50x.^2-40(y-0.1)^2)+.5exp(-30(x+0.5)^2-40(y+0.2)^2))


        @test values(u0)-values(u0|>LowRankFun)|>norm < 1000eps()
        @test chebyshevtransform(values(u0))-coefficients(u0)|>norm < 100eps()

        ##TODO: need to do adaptive to get better accuracy
        @test sin(u0)(.1,.2)-sin(u0(.1,.2))|>abs < 10e-4


        F = LowRankFun((x,y)->hankelh1(0,10abs(y-x)),Chebyshev(1.0..2.0),Chebyshev(Segment(1.0im,2.0im)))

        @test F(1.5,1.5im) ≈ hankelh1(0,10abs(1.5im-1.5))

        P = ProductFun((x,y)->x^2*y^3, Chebyshev() ⊗ Chebyshev())
        @test (Derivative() * P)(0.1, 0.2) ≈ ProductFun((x,y)->2x*y^3)(0.1, 0.2)
        @test (P * Derivative())(0.1, 0.2) ≈ ProductFun((x,y)->x^2*3y^2)(0.1, 0.2)

        P = ProductFun((x,y)->x*y, Chebyshev() ⊗ Chebyshev())
        xf = Fun(); yf = xf;
        xi, yi = 0.1, 0.2
        x, y = xf(xi), yf(yi)
        @test Evaluation(1) * P == xf
        @test Evaluation(-1) * P == -xf
        D1 = Derivative(Chebyshev() ⊗ Chebyshev(), [1,0])
        D2 = Derivative(Chebyshev() ⊗ Chebyshev(), [0,1])
        @test (D2 * P)(xi, yi) ≈ x
        @test (D1 * P)(xi, yi) ≈ y
        @test ((P * D1) * P)(xi, yi) ≈ x * y^2
        @test ((P * D2) * P)(xi, yi) ≈ x^2 * y
        @test ((I ⊗ Derivative()) * P)(xi, yi) ≈ x
        @test ((Derivative() ⊗ I) * P)(xi, yi) ≈ y

        # distribute over plus operator
        @test ((D1 + Multiplication(xf)⊗I) * P)(xi, yi) ≈ y +  x^2 * y
        @test ((P * (D1 + Multiplication(xf)⊗I)) * P)(xi, yi) ≈ x * y^2 +  x^3 * y^2

        A = (Multiplication(xf)⊗I) * Derivative(Chebyshev()^2, [0,1])
        @test (A * P)(xi, yi) ≈ x^2

        # MultivariateFun methods
        f = invoke(*, Tuple{KroneckerOperator, ApproxFunBase.MultivariateFun},
                Derivative() ⊗ I, P)
        @test f(xi, yi) ≈ y
        O = invoke(*, Tuple{ApproxFunBase.MultivariateFun, KroneckerOperator},
                P, Derivative() ⊗ I)
        @test (O * Fun(P))(xi, yi) ≈ x * y^2

        @testset "chopping" begin
            local P
            M = [0 0 0; 0 1 0; 0 0 1]
            P = ProductFun(M, Chebyshev() ⊗ Chebyshev(), chopping = true)
            @test coefficients(P) == M
            M = [0 0 0; 0 1 0; 0 0 1e-100]
            P = ProductFun(M, Chebyshev() ⊗ Chebyshev(), chopping = true)
            @test coefficients(P) == @view M[1:2, 1:2]
            M = [0 0 0; 0 1 0; 0 0 0]
            P = ProductFun(M, Chebyshev() ⊗ Chebyshev(), chopping = true)
            @test coefficients(P) == @view M[1:2, 1:2]
            M = zeros(3,3)
            P = ProductFun(M, Chebyshev() ⊗ Chebyshev(), chopping = true)
            @test all(iszero, coefficients(P))
        end

        @testset "KroneckerOperator" begin
            local P
            P = ProductFun((x,y)->x^2*y^3, Chebyshev() ⊗ Chebyshev())
            A = Multiplication(xf)⊗I
            a = (KroneckerOperator(A) * P)(xi, yi)
            b = (A * P)(xi, yi)
            @test a ≈ b
            a = ((P * KroneckerOperator(A)) * P)(xi, yi)
            b = ((P * A) * P)(xi, yi)
            @test a ≈ b
            # distribute a KroneckerOperator over a times operator
            A = (Multiplication(xf)⊗I) * Derivative(Chebyshev()^2, [0,1])
            a = ((P * A) * P)(xi, yi)
            b = ((P * KroneckerOperator(A)) * P)(xi, yi)
            @test a ≈ b
        end
    end

    @testset "Functional*Fun" begin
        d=ChebyshevInterval()
        B=ldirichlet(d)
        f=ProductFun((x,y)->cos(cos(x)*sin(y)),d^2)

        @test norm(B*f-Fun(y->cos(cos(-1)*sin(y)),d))<20000eps()
        @test norm(f*B-Fun(x->cos(cos(x)*sin(-1)),d))<20000eps()
    end

    @testset "matrix" begin
        f=Fun((x,y)->[exp(x*cos(y));cos(x*sin(y));2],ChebyshevInterval()^2)
        @test f(0.1,0.2) ≈ [exp(0.1*cos(0.2));cos(0.1*sin(0.2));2]

        f=Fun((x,y)->[exp(x*cos(y)) cos(x*sin(y)); 2 1],ChebyshevInterval()^2)
        @test f(0.1,0.2) ≈ [exp(0.1*cos(0.2)) cos(0.1*sin(0.2));2 1]
    end

    @testset "grad" begin
        f = Fun((x,y)->x^2*y^3, Chebyshev()^2)
        g = grad(f)
        @test g == grad(space(f))*f == grad(domain(f)) * f
        x = 0.1; y = 0.2
        @test g[1](x, y) ≈ 2x*y^3
        @test g[2](x, y) ≈ x^2*3y^2
    end
end
