using ApproxFunOrthogonalPolynomials
using LinearAlgebra
using Test

function test_transform!(v, v2, S)
    v .= rand.(eltype(v))
    v2 .= v
    @test itransform(S, transform(S, v)) ≈ v
    @test transform(S, itransform(S, v)) ≈ v
    transform!(S, v)
    @test transform(S, v2) ≈ v
    itransform!(S, v)
    @test v2 ≈ v
end

macro verbose(ex)
    head = ex.head
    args = ex.args
    @assert args[1] == Symbol("@testset")
    name = args[3] isa String ? args[3] : nothing
    if VERSION >= v"1.8"
        insert!(args, 3, Expr(:(=), :verbose, true))
    end
    Expr(head, args...)
end

