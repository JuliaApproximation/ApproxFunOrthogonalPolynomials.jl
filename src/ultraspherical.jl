export ultraconversion!,ultraint!

## Start of support for UFun

# diff from T -> U
function ultradiff(v::AbstractVector{<:Number})
    ultradiff!(copy(v))
end

function ultradiff!(v::AbstractVector{<:Number})
    Base.require_one_based_indexing(v)
    #polynomial is p(x) = sum ( v[i] * x^(i-1) )
    if length(v) <= 1
        fill!(v, zero(eltype(v)))
        return v
    end
    for k in eachindex(v)[1:end-1]
        v[k] = k*v[k+1]
    end
    resize!(v, length(v)-1)
    return v
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
function ultraiconversion(v::AbstractVector{<:Number})
    ultraiconversion!(copy(v))
end

function ultraiconversion!(v::AbstractVector{<:Number})
    Base.require_one_based_indexing(v)
    n = length(v)

    if n == 2
        @inbounds v[2] *= 2
    elseif n ≥ 3
        @inbounds v[end] *= 2
        @inbounds v[end-1] *= 2

        for k = n-2:-1:2
            @inbounds v[k] = 2v[k] + v[k+2]
        end

        @inbounds v[1] += 0.5 * v[3]
    end

    return v
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
