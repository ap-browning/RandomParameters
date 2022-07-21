#=
    approximations.jl

Approximate distributions using second order Taylor expansions.

=#


##############################################################
## Equations for the moments
##############################################################

Ef¹(f̂,∇,H,Σ,S,K) = f̂ + 1/2 * H ⊙ Σ
Ef²(f̂,∇,H,Σ,S,K) = f̂^2 + 
                    Σ ⊙ (∇ ⊗ ∇ + f̂ * H) + 
                    S ⊙ (H ⊗ ∇) + 
                    K ⊙ (1/4 * H ⊗ H)
Ef³(f̂,∇,H,Σ,S,K) = f̂^3 +
                    Σ ⊙ (3f̂ * ∇ ⊗ ∇ + 1.5 * f̂^2 * H) +
                    S ⊙ (∇ ⊗ ∇ ⊗ ∇ + 3f̂ * H ⊗ ∇) +
                    K ⊙ (3/4 * f̂ * H ⊗ H + 3/2 * ∇ ⊗ ∇ ⊗ H)
Efᵢⱼ(f̂ᵢ,∇ᵢ,Hᵢ,f̂ⱼ,∇ⱼ,Hⱼ,Σ,S,K) = f̂ᵢ * f̂ⱼ + 
                    Σ ⊙ (1/2 * (f̂ᵢ * Hⱼ + f̂ⱼ * Hᵢ + 2∇ᵢ ⊗ ∇ⱼ)) +
                    S ⊙ (∇ᵢ ⊗ Hⱼ + ∇ⱼ ⊗ Hᵢ) + 
                    K ⊙ (1/4 * (Hᵢ ⊗ Hⱼ))


##############################################################
## Equations for the moments
##############################################################

"""
    approximate_moments(f,θ;order=2)

Calculate approximate mean, variance and (if `order == 3`) skewness of univariate function `f` of parameters `θ`
"""
function approximate_moments(f::Function,θ::Distribution;order=2)
    # Get moments and derivatives
    θ̂,V,S,K = 𝔼(θ),𝕍(θ),𝕊(θ),𝕂(θ)
    f̂,∇,H = get_derivatives(f,θ̂)
    # Calculate mean and variance
    μ  = Ef¹(f̂,∇,H,V,S,K)
    σ² = Ef²(f̂,∇,H,V,S,K) - μ^2
    # If after a second order (mean and variance) then stop
    if order == 2
        return μ,σ²
    end
    # Calculate and return skewness
    ω  = (Ef³(f̂,∇,H,V,S,K) - 3σ²*μ - μ^3) / σ²^(3/2)
    return μ,σ²,ω
end

"""
    approximate_moments(f,θ,n;order=2,independent=false)

Calculate approximate mean, variance and (if `order == 3`) skewness of vector-valued function `f` of parameters `θ`.

`f` must be in ℝⁿ. By default, `independent=false` and so the covariance matrix is returned as a diagonal. To get the full covariance matrix, set `independent=false`.
"""
function approximate_moments(f::Function,θ::Distribution,n::Number;independent=false,order=2)
    # Get moments and derivatives
    θ̂,V,S,K = 𝔼(θ),𝕍(θ),𝕊(θ),𝕂(θ)
    f̂,∇,H = get_derivatives(f,θ̂,n)
    # Calculate mean and variance
    μ = [Ef¹(f̂[i],∇[i,:],H[i,:,:],V,S,K) for i = 1:n]
    Σ = independent == false ? zeros(n,n) : Diagonal(zeros(n))
    for i = 1:n, j = (independent == false ? (i:n) : (i:i))
        Σ[i,j] = Σ[j,i] = Efᵢⱼ(f̂[i],∇[i,:],H[i,:,:],f̂[j],∇[j,:],H[j,:,:],V,S,K) - μ[i] * μ[j]
    end
    # If after a second order (mean and variance) then stop
    if order == 2
        return μ,Symmetric(Σ)
    end
    # Calculate and return skewness'
    ω = similar(μ)
    for i = 1:n
        ω[i] = (Ef³(f̂[i],∇[i,:],H[i,:,:],V,S,K) - 3Σ[i,i]*μ[i] - μ[i]^3) / Σ[i,i]^(3/2)
    end
    return μ,Symmetric(Σ),ω
end

"""
    approximate_moments(f,θ,n,m;order=2,independent=true)

Calculate approximate mean, variance and (if `order == 3`) skewness of matrix-valued function `f` of parameters `θ`.

`f` must be in ℝ(n×m). The rows of `n` are assumed to be dependent observations and a vector of length `n` of means, covariance matrices, and skewness' associated with each row of `f` is returned.
"""
function approximate_moments(f::Function,θ::Distribution,n::Number,m::Number;order=2)
    # Remap f to a vector. 
    fv = x -> (f(x)')[:]
    # Index mapping
    idx(i,j) = (i-1)*m + j
    # Get moments and derivatives
    θ̂,V,S,K = 𝔼(θ),𝕍(θ),𝕊(θ),𝕂(θ)
    f̂,∇,H = get_derivatives(fv,θ̂,n*m)
    # Initialise outputs
    μ = [zeros(m) for _ = 1:n]
    Σ = [zeros(m,m) for _ = 1:n]
    ω = order == 3 ? [zeros(m) for _ = 1:n] : nothing
    # Loop through independent observations
    for i = 1:n
        # Means
        for j = 1:m
            ix = idx(i,j)
            μ[i][j] = Ef¹(f̂[ix],∇[ix,:],H[ix,:,:],V,S,K)
        end
        # Variances
        for j = 1:m, k = j:m
            ix1 = idx(i,j)
            ix2 = idx(i,k)
            Σ[i][j,k] = Σ[i][k,j] = Efᵢⱼ(f̂[ix1],∇[ix1,:],H[ix1,:,:],f̂[ix2],∇[ix2,:],H[ix2,:,:],V,S,K) - μ[i][j] * μ[i][k]
        end
        # Skewness' (if third order)
        if order == 3
            for j = 1:m
                ix = idx(i,j)
                ω[i][j] = (Ef³(f̂[ix],∇[ix,:],H[ix,:,:],V,S,K) - 3Σ[i][j,j]*μ[i][j] - μ[i][j]^3) / Σ[i][j,j]^(3/2)
            end
        end
    end
    if order == 2
        return μ,Σ
    else
        return μ,Σ,ω
    end
end


##############################################################
## Approximate distributions
##############################################################

"""
    approximate_transformed_distribution(f,θ,args...;order=2,independent=true)

Construct an approximate distribution for the transformation `f` of random variables `θ`.
"""
function approximate_transformed_distribution(args...;order=2,kwargs...)
    if order == 2
        μ,Σ = approximate_moments(args...;order=2,kwargs...)
        if length(args) == 2
            return Normal(μ,√Σ)
        elseif length(args) == 3
            return MvNormal(μ,Σ)
        else
            return MvNormal.(μ,Σ)
        end
    else
        μ,Σ,ω = approximate_moments(args...;order=3,kwargs...)
        if length(args) == 2
            return GammaAlt(μ,√Σ,ω)
        elseif length(args) == 3
            return MvGamma(μ,Σ,ω)
        else
            return MvGamma.(μ,Σ,ω)
        end
    end
end



##############################################################
## Univariate skewed approximations 
##############################################################

function approximate_mean_variance_skewness(f::Function,θ::Distribution)
    θ̂,Σ,S,K = 𝔼(θ),𝕍(θ),𝕊(θ),𝕂(θ)
    f̂,∇,H = f(θ̂), ∂(f,θ̂), ∂²(f,θ̂)
    Ef = f̂ + 1/2 * H ⊙ Σ
    Ef² = f̂^2 + ∇ ⊗ ∇ ⊙ Σ + f̂ * H ⊙ Σ + 1/4 * K ⊙ (H ⊗ H) + ∇ ⊗ H ⊙ S
    Ef³ = f̂^3 + 3f̂ * ∇ * ∇' ⊙ Σ + 
        1.5 * f̂^2 * H ⊙ Σ + 
        ∇ ⊗ ∇ ⊗ ∇ ⊙ S +
        0.75 * f̂ * H ⊗ H ⊙ K + 
        3f̂ * ∇ ⊗ H ⊙ S +
        1.5 * ∇ * ∇' ⊗ H ⊙ K
    return Ef,Ef² - Ef^2, (Ef³ - 3Ef² * Ef + 2Ef^3) / (Ef² - Ef^2)^(3/2)
end
function approximate_transformed_distribution_skewed(f::Function,θ::Distribution)
    μ,σ²,ω = approximate_mean_variance_skewness(f,θ)
    return GammaAlt(μ,√σ²,ω)
end


function approximate_mean_variance_skewness(f::Function,θ::Distribution,n::Int)
    θ̂,Σ,S,K = 𝔼(θ),𝕍(θ),𝕊(θ),𝕂(θ)
    f̂v,∇v,Hv = get_derivatives(f,θ̂,n)
    μ = zeros(n); σ² = zeros(n); ω = zeros(n);
    for i = 1:n # Could probably make this threads...
        f̂,∇,H = f̂v[i],∇v[i,:],Hv[i,:,:]
        Ef = f̂ + 1/2 * H ⊙ Σ
        Ef² = f̂^2 + ∇ ⊗ ∇ ⊙ Σ + f̂ * H ⊙ Σ + 1/4 * K ⊙ (H ⊗ H) + ∇ ⊗ H ⊙ S
        Ef³ = f̂^3 + 3f̂ * ∇ * ∇' ⊙ Σ + 
            1.5 * f̂^2 * H ⊙ Σ + 
            ∇ ⊗ ∇ ⊗ ∇ ⊙ S +
            0.75 * f̂ * H ⊗ H ⊙ K + 
            3f̂ * ∇ ⊗ H ⊙ S +
            1.5 * ∇ * ∇' ⊗ H ⊙ K
        μ[i] = Ef
        σ²[i] = Ef² - Ef^2
        ω[i] = (Ef³ - 3Ef² * Ef + 2Ef^3) / (Ef² - Ef^2)^(3/2)
    end
    return μ,σ²,ω
end




function approximate_transformed_distribution_skewed(f::Function,θ::Distribution,n::Int)
    μ,σ²,ω = approximate_mean_variance_skewness(f,θ,n)
    return GammaAlt.(μ,sqrt.(σ²),ω)
end
