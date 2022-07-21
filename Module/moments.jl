import StatsBase: moment

##############################################################
## Univariate moments
##############################################################

function StatsBase.moment(d::Distribution,n::Int)
    if n == 1
        return 0
    elseif n == 2
        return var(d)
    elseif n == 3
        return skewness(d) * std(d)^3
    elseif n == 4
        return (kurtosis(d) + 3) * std(d)^4
    end
end

##############################################################
## Product distribution
##############################################################

function 𝕍(d::Product)
    Diagonal([moment(d.v[i],2) for i = 1:length(d)])
end

function 𝕊(d::Product)
    m = zeros(fill(length(d),3)...)
    for i = 1:length(d)
        m[i,i,i] = moment(d.v[i],3)
    end
    return m
end

function countunique(x)
    u = unique(x)
    c = [count(==(element),x) for element in u]
    return u, c
end

function 𝕂(d::Product)
    dim = length(d)
    m = zeros(fill(dim,4)...)
    for i = 1:dim, j = 1:dim, k = 1:dim, l = 1:dim
        u,c = countunique([i,j,k,l])
        if minimum(c) > 1
            m[i,j,k,l] = prod(moment.(d.v[u],c))
        end
    end
    return m
end

##############################################################
## MvNormal distribution
##############################################################

function 𝕊(Σ)
    d = size(Σ,1)
    zeros(d,d,d)
end
function 𝕂(Σ)
    d = size(Σ,1)
    K = zeros(d,d,d,d)
    for x = 1:d, y = 1:d, z = 1:d, w = 1:d
        K[x,y,z,w] = Σ[x,y]*Σ[z,w] + Σ[x,z]*Σ[y,w] + Σ[x,w]*Σ[y,z]
    end
    return K
end


##############################################################
## General
##############################################################

𝔼(args...) = mean(args...)
𝕍(args...) = cov(args...)
𝕊(d::Number) = zeros(d,d,d)
𝕊(θ::Distribution) = 𝕊(cov(θ))
𝕂(θ::Distribution) = 𝕂(cov(θ))

function 𝔼(d::Product)
    return mean.(d.v)
end