import Base: rand, minimum, maximum
import Distributions: pdf, cdf, kurtosis, logpdf, loglikelihood
import Statistics: mean, var, std, quantile
import StatsBase: skewness, sample, kurtosis

##############################################################
## Alternative parameterisation of the gamma distribution
##############################################################
# Due to numerically stability, returns a normal distribution if ω ≤ 1e-4
const ω_threshold = 1e-4
struct GammaAlt{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    ω::T
    θ::NamedTuple
    d::Gamma
end
GammaAlt(μ,σ,ω) = abs(ω) > ω_threshold ? GammaAlt(μ,σ,ω,gamma_moments_inv([μ,σ,ω]),Gamma(4 / ω^2)) : Normal(μ,σ)

function gamma_moments_inv(m)
    μ,σ,ω = m
    k = 4 / ω^2
    θ = σ * ω / 2
    s = μ - 2σ / ω
    return (k = k,θ = θ,s = s)
end

#### Evaluation
rand(rng::AbstractRNG, d::GammaAlt) = d.θ.θ * rand(rng,d.d) + d.θ.s
pdf(d::GammaAlt,x::Real) = pdf(d.d,(x - d.θ.s) / d.θ.θ) / abs(d.θ.θ)
logpdf(d::GammaAlt,x::Real) = logpdf(d.d,(x - d.θ.s) / d.θ.θ) - log(abs(d.θ.θ))
cdf(d::GammaAlt,x::Real) = d.θ.θ > 0.0 ? 
    cdf(d.d,(x - d.θ.s) / d.θ.θ) : 
    1 - cdf(d.d,(x - d.θ.s) / d.θ.θ)
quantile(d::GammaAlt,p::AbstractArray) = d.θ.θ > 0.0 ? 
    d.θ.θ * quantile(d.d,p) .+ d.θ.s : 
    d.θ.θ * quantile(d.d,1 .- p) .+ d.θ.s
quantile(d::GammaAlt,p::Number) = d.θ.θ > 0.0 ? 
    d.θ.θ * quantile(d.d,p) .+ d.θ.s : 
    d.θ.θ * quantile(d.d,1 - p) .+ d.θ.s
minimum(d::GammaAlt) = quantile(d,0.0)
maximum(d::GammaAlt) = quantile(d,1.0)

mean(d::Union{GammaAlt}) = d.μ
std(d::Union{GammaAlt}) = d.σ
var(d::Union{GammaAlt}) = std(d)^2
skewness(d::Union{GammaAlt}) = d.ω
kurtosis(d::Union{GammaAlt}) = 6 / d.θ.k



##############################################################
## Custom implementation of point-mass density (i.e., dirac) (i.e., for constant parameters)
##############################################################

# Due to numerically stability, returns a normal distribution if ω ≤ 1e-4
struct DiracContinuous{T<:Real} <: ContinuousUnivariateDistribution
    value::T
end

minimum(d::DiracContinuous) = d.value
maximum(d::DiracContinuous) = d.value
mean(d::DiracContinuous) = d.value
var(d::DiracContinuous) = 0.0
std(d::DiracContinuous) = 0.0
skewness(d::DiracContinuous) = 0.0  # Based on normal where σ → 0.0
kurtosis(d::DiracContinuous) = 0.0  # Based on normal where σ → 0.0
pdf(d::DiracContinuous, x::Real) = insupport(d, x) ? 1.0 : 0.0
logpdf(d::DiracContinuous, x::Real) = insupport(d, x) ? 0.0 : -Inf
insupport(d::DiracContinuous, x::Real) = x == d.value
rand(rng::AbstractRNG, d::DiracContinuous) = d.value

##############################################################
## Gaussian copula
##############################################################
"""
    GaussianCopula(P)

Construct a Gaussian copula with correlation matrix `P`.
"""
abstract type Copula end
struct GaussianCopula{N} <: Copula
    P::Matrix           # Covariance matrix
    L::LowerTriangular  # Cholesky decomposition of `P`
    D::Number           # Determinant of P
end
GaussianCopula(P::Matrix) = GaussianCopula{size(P,1)}(P,cholesky(P).U',det(P))
GaussianCopula(p) = (P = vec_to_cor(p); GaussianCopula{size(P,1)}(P,cholesky(P).U',det(P)))

rand(rng::AbstractRNG,C::GaussianCopula{N}) where {N} = normcdf.(C.L * randn(rng,N))
rand(rng::AbstractRNG,C::GaussianCopula{N},n::Int) where {N} = normcdf.(C.L * randn(rng,N,n))
rand(C::GaussianCopula{N},n::Int) where {N} = normcdf.(C.L * randn(N,n))
sample(C::GaussianCopula{N},n::Int=1) where {N} = normcdf.(C.L * randn(N,n))

pdf(C::GaussianCopula{N},u::Vector) where {N} = (x = norminvcdf.(u); 1 / sqrt(C.D) * exp.(-0.5 * x' * (inv(C.P) - I) * x))
logpdf(C::GaussianCopula{N},u::Vector) where {N} = (x = norminvcdf.(u); -0.5log(C.D) - 0.5 * x' * (inv(C.P) - I) * x)


marginalize(C::GaussianCopula{N};dims=[1,2]) where {N} = GaussianCopula(C.P[dims,dims])


##############################################################
## Copula-MVDistribution
##############################################################
"""
    MvDependent(C,v)

Construct dependent multivariate distribution where the marginals are described by
elements of `v` (a vector), and the dependence structure is described a copula `C`.
"""
struct MvDependent{N}
    C::Copula
    v::Vector
    q::Vector{Bool}
end
MvDependent(C::Copula,v::Vector;q=falses(length(v))) = MvDependent{length(v)}(C,v,q)

function sample(d::MvDependent{N},n=1) where {N}
    U = rand(d.C,n)
    for i = 1:N
        U[i,:] = d.q[i] ? quantile_interpolated(d.v[i],U[i,:]) : quantile(d.v[i],U[i,:])
    end
    return U
end
rand(d::MvDependent,n::Int=1) = n==1 ? sample(d,n)[:] : sample(d,n)

marginalize(d::MvDependent;dims=[1,2]) = MvDependent(
    marginalize(d.C;dims),
    d.v[dims],
    q=d.q[dims]
)

nantoinf(x) = isnan(x) ? -Inf : x

pdf(d::MvDependent,x::Vector) = pdf(d.C,max.(0.0,cdf.(d.v,x))) * prod(pdf.(d.v,x))
logpdf(d::MvDependent,x::Vector) = nantoinf(logpdf(d.C,max.(0.0,cdf.(d.v,x)))) + sum(logpdf.(d.v,x))
loglikelihood(d::MvDependent,x::Matrix) = sum(logpdf(d,x[:,i]) for i = 1:size(x,2))

"""
    vec_to_cor(p)

Convert number of vector `p` to a correlation matrix.

Example:

    vec_to_cor([0.1,0.2,0.3])
    3×3 Matrix{Float64}:
     1.0  0.1  0.2
     0.1  1.0  0.3
     0.2  0.3  1.0
"""
function vec_to_cor(p::Vector)
    n = 0.5 * (1 + sqrt(1 + 8length(p)))
    isinteger(n) || error("Check length of input.")
    n = Int(n)
    P = zeros(n,n)
    P[(triu(ones(n,n)) - I) .== 1.0] = p
    P += P' + I
end
vec_to_cor(p::Real) = vec_to_cor([p])


##############################################################
## Create correlated gamma
##############################################################

# Load correlation data from file
data = load("Module/correlation_data.jld2","W₁","W₂","P̃","P")

# Construct forward interpolation
itp = LinearInterpolation(data[1:3],data[4],extrapolation_bc=Interpolations.Flat())

# Function to get input correlation from output
function get_input_correlation(ω₁,ω₂,ρ)
    # If correlation is small, return output correlation as input
    if abs(ρ) < 1e-3
        return ρ
    end
    ω1 = min(abs(ω₁),2.0)   # Only works for ω < 2
    ω2 = min(abs(ω₂),2.0)
    ρ1 = min(abs(ρ),0.99)   # Only works for ρ < 0.98
    # Check that the correlation is achievable, else return the closest
    ρ₁ = itp(ω1,ω2,0.0) # Output correlation at 0.0
    ρ₂ = itp(ω1,ω2,0.0) # Output correlation at 0.99
    return sign(ρ₁ - ρ1) == sign(ρ₂ - ρ1) ? ρ : min(max(0.0,find_zero(ρ̃ -> itp(ω1,ω2,ρ̃) - ρ1,(0.0,0.99))),0.99)
end

# Function to get MvGamma
function MvGamma(μ,σ,ω,ρ)
    length(μ) == 2 && length(σ) == 2 && length(ω) == 2 || error("Only valid for bivariate distributions.")
    d = GammaAlt.(μ,σ,ω)
    ρ̃ = get_input_correlation(ω...,ρ)
    return MvDependent(GaussianCopula(ρ̃),d)
end
function MvGamma(μ,Σ::Union{Matrix,Symmetric},ω)
    σ = sqrt.(diag(Σ))
    ρ = Σ[1,2] / prod(σ)
    return MvGamma(μ,σ,ω,ρ)
end