#=
    Figure 9

    Simulation results for non-linear two-pool model with dependent observations.

    Observations are taken at (b-c) a single time point, and (a,d) multiple time points.

=#

using Revise
using RandomParameters
using Plots
using StatsPlots
using LinearAlgebra
using Distributions
using DifferentialEquations
using AdaptiveMCMC
using .Threads
using KernelDensity
using Random

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

    ξ = log.([μ₂₁,μv₂₁,μ₀₁,μ₀₂,σ₂₁,σv₂₁,σ₁,σ₂])

    # Bounds
    bounds = fill([-15.0,0.0],length(ξ)); bounds[2] = [-15.0,3.0]; bounds[6] = [-15.0,2.0]

#################################################
## Single observation point: data and likelihood

    n_single = 100
    x_single = hcat([twopool_nonlin_single(θᵢ) for θᵢ = eachcol(rand(MersenneTwister(1),θ,n_single))]...)

    # Likelihood
    function loglike_single(ξ)
        μ₂₁,μv₂₁,μ₀₁,μ₀₂,σ₂₁,σv₂₁,σ₁,σ₂ = exp.(ξ)
        k₀₁ = DiracContinuous(μ₀₁)
        k₂₁ = Normal(μ₂₁,σ₂₁)
        V₂₁ = Normal(μv₂₁,σv₂₁)
        k₀₂ = DiracContinuous(μ₀₂)
        ε₁ = Normal(1.0,σ₁)
        ε₂ = Normal(0.0,σ₂)
        θ = Product([k₂₁,V₂₁,k₀₁,k₀₂,ε₁,ε₂])
        d = approximate_transformed_distribution(twopool_nonlin_single,θ,2;order=3,independent=false)
        loglikelihood(d,x_single)
    end

    # Find MLE
    @time ξ̂ = optimise(loglike_single,ξ;bounds,obj=:maximise,alg=:LN_NELDERMEAD)[2]

#################################################
## F(ξ) -> [moments]

function moments(ξ)
    μ₂₁,μv₂₁,μ₀₁,μ₀₂,σ₂₁,σv₂₁,σ₁,σ₂ = exp.(ξ)
    k₀₁ = DiracContinuous(μ₀₁)
    k₂₁ = Normal(μ₂₁,σ₂₁)
    V₂₁ = Normal(μv₂₁,σv₂₁)
    k₀₂ = DiracContinuous(μ₀₂)
    ε₁ = Normal(1.0,σ₁)
    ε₂ = Normal(0.0,σ₂)
    θ = Product([k₂₁,V₂₁,k₀₁,k₀₂,ε₁,ε₂])
    μ,Σ,ω = approximate_moments(twopool_nonlin_single,θ,2;order=3,independent=false)
    [μ;sqrt.(diag(Σ));ω; Σ[1,2] / prod(sqrt.(diag(Σ)))]
end

# Get Jacobian
J = FiniteDiff.finite_difference_jacobian(moments,ξ̂)

# Get FIM
S = J' * J