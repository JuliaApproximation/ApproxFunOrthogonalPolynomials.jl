export ultraconversion!,ultraint!

## Start of support for UFun

# diff from T -> U
function ultradiff(v::AbstractVector{T}) where T<:Number
    Base.require_one_based_indexing(v)
    #polynomial is p(x) = sum ( v[i] * x^(i-1) )
    if length(v)≤1
        w = zeros(T,1)
    else
        w = Array{T}(undef, length(v)-1)
        for k in eachindex(w)
            @inbounds w[k] = k*v[k+1]
        end
    end

    w
end

#int from U ->T

#TODO: what about missing truncation?
function ultraint!(v::AbstractMatrix{T}) where T<:Number
    for j in axes(v,2)
        for k in reverse(axes(v,1)[firstindex(v,1)+1:end])
            @inbounds v[k,j] = v[k-1,j]/(k-1)
        end
    end

    @simd for j in axes(v,2)
        @inbounds v[1,j] = zero(T)
    end

    v
end

function ultraint!(v::AbstractVector{T}) where T<:Number
    resize!(v,length(v)+1)
    @simd for k in reverse(eachindex(v)[firstindex(v)+1:end])
        @inbounds v[k] = v[k-1]/(k-1)
    end

    @inbounds v[firstindex(v)] = zero(T)

    v
end

# Convert from U -> T
function ultraiconversion(v::AbstractVector{T}) where T<:Number
    Base.require_one_based_indexing(v)
    n = length(v)
    w = Array{T}(undef, n)

    if n == 1
        w[1] = v[1]
    elseif n == 2
        w[1] = v[1]
        w[2] = 2v[2]
    elseif n ≥ 3
        @inbounds w[end] = 2v[end]
        @inbounds w[end-1] = 2v[end-1]

        for k = n-2:-1:2
            @inbounds w[k] = 2*(v[k] + .5w[k+2])
        end

        @inbounds w[1] = v[1] + .5w[3]
    end

    w
end


# Convert T -> U
function ultraconversion(v::AbstractVector{<:Number})
    ultraconversion!(float.(v))
end

function ultraconversion!(v::AbstractVector{<:Number})
    Base.require_one_based_indexing(v)
    n = length(v) #number of coefficients

    if n == 2
        @inbounds v[2] /= 2
    elseif n > 2
        @inbounds v[1] -= v[3]/2

        for j=2:n-2
            @inbounds v[j] = (v[j] - v[j+2])/2
        end
        @inbounds v[n-1] /= 2
        @inbounds v[n] /= 2
    end

    v
end

function ultraconversion!(v::AbstractMatrix{T}) where T<:Number
    Base.require_one_based_indexing(v)
    n = size(v)[1] #number of coefficients
    m = size(v)[2] #number of funs


    if n ≤ 1
        #do nothing
    elseif n == 2
        @simd for k=1:m
            @inbounds v[2,k] /= 2
        end
    else
        for k=1:m
            @inbounds v[1,k] -= v[3,k]/2

            for j=2:n-2
                @inbounds v[j,k] = (v[j,k] - v[j+2,k])/2
            end
            @inbounds v[n-1,k] /= 2
            v[n,k] /= 2
        end
    end

    v
end


#ultraiconversion and ultraconversion are linear, so it is possible to define them on Complex numbers as so
#ultraiconversion(v::AbstractVector{Complex{Float64}})=ultraiconversion(real(v)) + ultraiconversion(imag(v))*1.0im
#ultraconversion(v::AbstractVector{Complex{Float64}})=ultraconversion(real(v)) + ultraconversion(imag(v))*1.0im

#using DualNumbers
#ultraconversion{T<:Number}(v::AbstractVector{Dual{T}})=dual(ultraconversion(real(v)), ultraconversion(epsilon(v)) )
