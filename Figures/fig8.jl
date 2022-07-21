#=
    Figure 8

    Simulation results for non-linear two-pool model with dependent observations.

    Observations are taken at (b - c) a single time point, and (a,d) multiple time points.

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

# Model 2: observe both pools at various times
T = 2.0:2.0:10.0
function twopool_nonlin(θ)
    k₂₁,V₂₁,k₀₁,k₀₂,ε₁,ε₂ = θ
    x₀ = [15.0,0.0]
    sol = solve(ODEProblem(twopool_nonlin_rhs!,x₀,(0.0,maximum(T)),[k₂₁,V₂₁,k₀₁,k₀₂]),saveat=T)
    x = hcat(sol.(T)...)'
    x[:,1] *= ε₁
    x[:,2] .+= ε₂
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
## (a) Multiple observation points

    n = 20
    Θ = rand(MersenneTwister(2),θ,n*length(T)); Θ = reshape(Θ,length(θ),n,length(T))
    x = [hcat([twopool_nonlin(Θ[:,j,i])[i,:] for j = 1:n]...) for i = 1:length(T)]

    # Likelihood
    function loglike(ξ)
        μ₂₁,μv₂₁,μ₀₁,μ₀₂,σ₂₁,σv₂₁,σ₁,σ₂ = exp.(ξ)
        k₀₁ = DiracContinuous(μ₀₁)
        k₂₁ = Normal(μ₂₁,σ₂₁)
        V₂₁ = Normal(μv₂₁,σv₂₁)
        k₀₂ = DiracContinuous(μ₀₂)
        ε₁ = Normal(1.0,σ₁)
        ε₂ = Normal(0.0,σ₂)
        θ = Product([k₂₁,V₂₁,k₀₁,k₀₂,ε₁,ε₂])
        d = approximate_transformed_distribution(twopool_nonlin,θ,length(T),2;order=3)
        sum(loglikelihood.(d,x))
    end

    # Find MLE
    @time ξ̂ = optimise(loglike,ξ;bounds,obj=:maximise,alg=:LN_NELDERMEAD)[2]

    # Profile
    plims = [[-1.0,0.5],[0.0,3.0],[-2.5,-2.2],[-1.0,-0.8],[-3.0,-1.0],[-1.0,2.0],[-8.0,-3.0],[-8.0,-3.0]]
    @time pvec,prof,argm = profile(loglike,ξ,plims;bounds,obj=:maximise,alg=:LN_NELDERMEAD,normalise=true,npt=50)

    # Benchmark against the fixed-parameter problem
    function loglike_fixedparam(ξ)
        μ₂₁,μv₂₁,μ₀₁,μ₀₂,σ₂₁,σv₂₁,σ₁,σ₂ = exp.(ξ)
        k₀₁ = DiracContinuous(μ₀₁)
        k₂₁ = Normal(μ₂₁,σ₂₁)
        V₂₁ = Normal(μv₂₁,σv₂₁)
        k₀₂ = DiracContinuous(μ₀₂)
        ε₁ = Normal(1.0,σ₁)
        ε₂ = Normal(0.0,σ₂)
        θ = Product([k₂₁,V₂₁,k₀₁,k₀₂,ε₁,ε₂])
        d = approximate_transformed_distribution(twopool_nonlin,θ,length(T),2;order=3)
        sum(loglikelihood.(d,x))
    end

#################################################
## (b) Single observation point

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
    @time ξ̂_single = optimise(loglike_single,ξ;bounds,obj=:maximise,alg=:LN_NELDERMEAD)[2]

    # Profile
    plims = [[-2.0,2.0],[-4.0,3.0],[-4.0,-2.0],[-1.0,-0.5],[-5.0,-1.0],[-5.0,2.0],[-8.0,-1.0],[-8.0,-3.0]]
    @time pvec_single,prof_single,argm = profile(loglike_single,ξ,plims;bounds,obj=:maximise,alg=:LN_NELDERMEAD,normalise=true,npt=50)

#################################################
## Figure 8

param_names = ["μ21","μv21","μ01","μ02","σ21","σv21","σ1","σ2"]

fig8 = plot(layout=grid(2,4))
for i = 1:8
    plot!(fig8,subplot=i,pvec_single[i],prof_single[i],c=col_skew,label="Single")
    plot!(fig8,subplot=i,pvec[i],prof[i],c=col_norm,label="Multiple")
    hline!(fig8,subplot=i,[-1.92],c=:black,ls=:dash,label="95% CI")
    vline!(fig8,subplot=i,[ξ[i]],c=:black,ls=:dot,label="True value")
    plot!(fig8,subplot=i,xlabel=param_names[i])
    if i < 8
        plot!(fig8,subplot=i,legend=:none)
    end
end
plot!(ylim=(-5.0,0.0),widen=true,size=(700,350))
add_plot_labels!(fig8)

savefig(fig8,"$(@__DIR__)/fig8.svg")
