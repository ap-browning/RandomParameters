#=
    Timings

    Simulation results for non-linear two-pool model with dependent observations at a single time point.

=#

using Revise
using RandomParameters
using Plots
using StatsPlots
using LinearAlgebra
using Distributions
using Optim
using DifferentialEquations
using AdaptiveMCMC
using .Threads
using KernelDensity
using Random
using BenchmarkTools

using ForwardDiff
using FiniteDiff

include("figure_defaults.jl")

#################################################
## MODEL

function twopool_nonlin_rhs!(dx,x,p,t)
    k₂₁,V₂₁,k₀₁,k₀₂ = p
    x₁,x₂ = x
    k(x₁) = k₂₁ * x₁ / (V₂₁ + x₁)
    dx[1] = -(k₀₁ + k(x₁)) * x₁
    dx[2] = k(x₁) * x₁ - k₀₂ * x₂
end

# Model 1: observe only at single time
t_single = 10.0
function twopool_nonlin_single(θ)
    k₂₁,V₂₁,k₀₁,k₀₂,ε₁,ε₂ = θ
    x₀ = [15.0,0.0]
    sol = solve(ODEProblem(twopool_nonlin_rhs!,x₀,(0.0,t_single),[k₂₁,V₂₁,k₀₁,k₀₂]),saveat=[t_single])
    x = sol.u[1]
    x[1] *= ε₁
    x[2] += ε₂
    return x
end

#################################################
## "TRUE" PARAMETER DISTRIBUTION

    # Means
    μ₂₁,μv₂₁,μ₀₁,μ₀₂ = μ = [0.6,5.0,0.1,0.4]

    # Variances (only V₂₁, k₂₁ and ε are variable)
    σ₂₁,σv₂₁,σ₁,σ₂ = 0.1,1.0,0.01,0.01

    # "True" distributions
    k₀₁ = DiracContinuous(μ₀₁)
    k₂₁ = Normal(μ₂₁,σ₂₁)
    V₂₁ = Normal(μv₂₁,σv₂₁)
    k₀₂ = DiracContinuous(μ₀₂)
    ε₁ = Normal(1.0,σ₁)
    ε₂ = Normal(0.0,σ₂)

    θ = Product([k₂₁,V₂₁,k₀₁,k₀₂,ε₁,ε₂])

#################################################
## Single observation point

    n_single = 100
    x_single = hcat([twopool_nonlin_single(θᵢ) for θᵢ = eachcol(rand(MersenneTwister(1),θ,n_single))]...)

    # Get transformation
    d = approximate_transformed_distribution(twopool_nonlin_single,θ,2;order=3,independent=false)

    # Time standard approach (fixed parameters)
    Σ = cov(x_single')  # Sample variance
    @btime begin
        μ = twopool_nonlin_single(mean(θ))
        d = MvNormal(μ,Σ)
        l = loglikelihood(d,x_single)
    end                     # 67 μs

    # Time our standard approach (random parameters)
    @btime begin
        d = approximate_transformed_distribution(twopool_nonlin_single,θ,2;order=2,independent=false)
        l = loglikelihood(d,x_single)
    end                     # 846 μs

    # Time kurotis calculation (most expensive bit)
    @btime K = 𝕂(θ);        # 546 μs

    # Time kurtosis calculation (sparse structure)
    function 𝕂indep(d)
        dim = length(d)
        m = zeros(dim,dim)
        for i = 1:dim, j = 1:dim
            u,c = RandomParameters.countunique([i,i,j,j])
            if minimum(c) > 1
                m[i,j] = prod(moment.(d.v[u],c))
            end
        end
        return m
    end
    @btime K = 𝕂indep(θ)    # 46 μs