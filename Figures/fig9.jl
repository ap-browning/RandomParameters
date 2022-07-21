#=
    Figure 9

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
    @time ξ̂_single = optimise(loglike_single,ξ;bounds,obj=:maximise,alg=:LN_NELDERMEAD)[2]

#################################################
## MCMC

    # Setup prior and posterior
    lb = [bound[1] for bound in bounds]
    ub = [bound[2] for bound in bounds]
    logprior = Product(Uniform.(lb,ub))
    logpost = ξ -> insupport(prior,ξ) ? logpdf(prior,ξ) + loglike_single(ξ) : -Inf

    # Find twelve random starting locations with posterior support
    ξ₀ = Array{Vector}(undef,12); ξ₀[1] = ξ
    for i = 2:12
        ξprop = rand(prior); ξpost = logpost(ξprop)
        while isinf(ξpost)
            ξprop = rand(prior); ξpost = logpost(ξprop)
        end
        ξ₀[i] = ξprop
    end

    # Sample six chains using adaptive MCMC
    X = Array{Any}(undef,12)
    @time @threads for i = 1:12
        X[i] = adaptive_rwm(ξ₀[i],logpost,100_000;thin=100,algorithm=:aswam)
    end

#################################################
## FIND MAP IN ZERO V STANDARD DEVIATION REGION

    # Likelihood
    function loglike_deltaV(ξ)
        μ₂₁,μv₂₁,μ₀₁,μ₀₂,σ₂₁,σv₂₁,σ₁,σ₂ = exp.(ξ)
        k₀₁ = DiracContinuous(μ₀₁)
        k₂₁ = Normal(μ₂₁,σ₂₁)
        V₂₁ = DiracContinuous(μv₂₁)
        k₀₂ = DiracContinuous(μ₀₂)
        ε₁ = Normal(1.0,σ₁)
        ε₂ = Normal(0.0,σ₂)
        θ = Product([k₂₁,V₂₁,k₀₁,k₀₂,ε₁,ε₂])
        d = approximate_transformed_distribution(twopool_nonlin_single,θ,2;order=3,independent=false)
        loglikelihood(d,x_single)
    end

    # Find MLE in region near chain 2
    @time ξ̂_deltaV = optimise(loglike_deltaV,X[2].X[:,end];bounds,obj=:maximise,alg=:LN_NELDERMEAD)[2]

    # Perform likelihood ratio test: λ ~ Chisq(dim(ξ))
    λ = 2(loglike_single(ξ̂_single) - loglike_deltaV(ξ̂_deltaV))
    p = 1 - cdf(Chisq(length(ξ)),λ)

#################################################
## PLOTS

    # Plot density
    fig9a = plot()
    for i = 1:12
        plot!(fig9a,X[i].D[1],label="")
    end
    plot!(fig9a,ylim=(190.0,280.0),legend=:bottomright)
    hline!(fig9a,[logpost(ξ̂_single)],c=:black,ls=:dash,label="MAP")
    hline!(fig9a,[logpost(ξ)],c=:black,ls=:dot,label="True value")
    hline!(fig9a,[logpost(ξ̂_single) - quantile(Chisq(length(ξ)),0.95) / 2],c=:black,ls=:solid,label="95%")
    vline!(fig9a,[400.0],c=:black,ls=:dashdot,label="Burnin")

    # Chains to consider
    threshold = logpost(ξ̂_single) - quantile(Chisq(length(ξ)),0.95) / 2
    idx = [mean(X[i].D[1][401:end]) .> threshold for i = 1:length(X)]

    # Produce posterior densities
    param_names = ["μ21","μv21","μ01","μ02","σ21","σv21","σ1","σ2"]
    fig9c = plot(layout=grid(2,4))
    for i = 1:8
        for j = 1:length(X)
            if idx[j]
                density!(fig9c,subplot=i,X[j].X[i,401:end])
            else
                plot!(fig9c,subplot=i,[],[])
            end
        end
        vline!(fig9c,subplot=i,[ξ[i]],c=:black,ls=:dot)
        plot!(fig9c,subplot=i,xlabel=param_names[i])
    end
    plot!(fig9c,legend=:none)

    # Plot model output at each MAP
    d = approximate_transformed_distribution(twopool_nonlin_single,θ,2;order=3,independent=false)

    θ̂_single = Product([
        Normal(exp(ξ̂_single[1]),exp(ξ̂_single[5])),
        Normal(exp(ξ̂_single[2]),exp(ξ̂_single[6])),
        DiracContinuous(exp(ξ̂_single[3])),
        DiracContinuous(exp(ξ̂_single[4])),
        Normal(1.0,exp(ξ̂_single[7])),
        Normal(0.0,exp(ξ̂_single[8]))
        ])
    d̂_single = approximate_transformed_distribution(twopool_nonlin_single,θ̂_single,2;order=3,independent=false)

    θ̂_deltaV = Product([
        Normal(exp(ξ̂_deltaV[1]),exp(ξ̂_deltaV[5])),
        DiracContinuous(exp(ξ̂_deltaV[2])),
        DiracContinuous(exp(ξ̂_deltaV[3])),
        DiracContinuous(exp(ξ̂_deltaV[4])),
        Normal(1.0,exp(ξ̂_deltaV[7])),
        Normal(0.0,exp(ξ̂_deltaV[8]))
        ])
    d̂_deltaV = approximate_transformed_distribution(twopool_nonlin_single,θ̂_deltaV,2;order=3,independent=false)

    xv = range(0.2,1.5,200)
    yv = range(0.5,0.75,201)
    dv = [pdf(d,[xx,yy]) for xx in xv, yy in yv]
    dv_single = [pdf(d̂_single,[xx,yy]) for xx in xv, yy in yv]
    dv_deltaV = [pdf(d̂_deltaV,[xx,yy]) for xx in xv, yy in yv]

    fig9b = contourf(xv,yv,dv',color=cgrad(:bone,rev=true),fill=true,lw=0.0,levels=5,legend=:none)
    contour!(xv,yv,dv_single',c=col_alt,levels=5,legend=:none)
    contour!(xv,yv,dv_deltaV',c=col_skew,levels=5,ls=:dash,lw=2.0)
    scatter!(fig9b,eachrow(x_single)...,c=col_blue)

#################################################
## FIGURE 9

fig9 = plot(fig9a,fig9b,fig9c,layout=@layout([a b; c{0.6h}]),size=(800,600))

add_plot_labels!(fig9)
savefig(fig9,"$(@__DIR__)/fig9.svg")