#=

    Figure 6 and Figure S2

    Random parameter linear two-pool model.

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

using ForwardDiff
using FiniteDiff

include("figure_defaults.jl")

#################################################
## MODEL

function twopool_rhs!(dx,x,p,t)
    k₁₂,k₀₁,k₂₁,k₀₂ = p
    x₁,x₂ = x
    dx[1] = k₁₂ * x₂ - (k₀₁ + k₂₁) * x₁
    dx[2] = k₂₁ * x₁ - (k₀₂ + k₁₂) * x₂
end

# Model: observe pool 2 at t = 1:0.5:5
T = [0.5:1.0:3.5;5.0:2.0:7.0]
function twopool(θ)
    k₁₂,k₀₁,k₂₁,k₀₂,ε = θ
    x₀ = [15.0,0.0]
    sol = solve(ODEProblem(twopool_rhs!,x₀,(0.0,maximum(T)),[k₁₂,k₀₁,k₂₁,k₀₂]),saveat=T)
    [x[2] for x in sol.(T)] * ε
end

#################################################
## "TRUE" PARAMETER DISTRIBUTION

    # Means
    μ₁₂,μ₀₁,μ₂₁,μ₀₂ = μ = [0.0,0.7,0.6,0.4]

    # Variances (only k₂₁ and ε are variable)
    σ₂₁,σ = 0.1,0.01

    # "True" distributions
    k₁₂ = DiracContinuous(μ₁₂)
    k₀₁ = DiracContinuous(μ₀₁)
    k₂₁ = Normal(μ₂₁,σ₂₁)
    k₀₂ = DiracContinuous(μ₀₂)
    ε = Normal(1.0,σ)

    θ = Product([k₁₂,k₀₁,k₂₁,k₀₂,ε])

#################################################
## SYNTHETIC DATA

    n = 20
    x = [[twopool(rand(θ))[i] for _ = 1:n] for i = 1:length(T)]

    # Plot at parameter mean
    sol = solve(ODEProblem(twopool_rhs!,[15.0,0.0],(0.0,10.0),μ))
    plot(sol,vars=2,c=:black,label="Model")

    # Plot data
    [scatter!(fill(T[i],n),x[i],c=col_skew,label=i==1 ? "Data" : "") for i = 1:length(T)]

    plot!()

#################################################
## LIKELIHOOD

    ξ = [μ[2:4];log(σ₂₁);log(σ)]

    # Output (vectorised to save computation)
    f = θ -> twopool(θ)

    function loglike(ξ)
        μ₀₁,μ₂₁,μ₀₂,lσ₂₁,lσ = ξ
        k₁₂ = DiracContinuous(0.0)
        k₀₁ = DiracContinuous(μ₀₁)
        k₂₁ = Normal(μ₂₁,exp(lσ₂₁))
        k₀₂ = DiracContinuous(μ₀₂)
        ε = Normal(1.0,exp(lσ))
        θ = Product([k₁₂,k₀₁,k₂₁,k₀₂,ε])
        #d = approximate_transformed_distribution_skewed(f,θ,length(T))
        d = approximate_transformed_distribution(f,θ,length(T);order=3,independent=true)
        sum(loglikelihood.(d,x))
    end

#################################################
## FIND MLE/MAP

    bounds = [[0.0,2.0],[0.0,2.0],[0.0,2.0],[-10.0,0.0],[-10.0,0.0]]
    @time ξ̂ = optimise(loglike,ξ;bounds,obj=:maximise,alg=:LN_NELDERMEAD)[2]

    k̂₁₂ = DiracContinuous(0.0)
    k̂₀₁ = DiracContinuous(ξ̂[1])
    k̂₂₁ = Normal(ξ̂[2],exp(ξ̂[4]))
    k̂₀₂ = DiracContinuous(ξ̂[3])
    ε̂ = Normal(1.0,exp(ξ̂[5]))
    θ̂ = Product([k̂₁₂,k̂₀₁,k̂₂₁,k̂₀₂,ε̂])

#################################################
## BENCHMARK AGAINST FIXED-PARAMETER MODEL

    function loglike_std(θ)
        k₀₁,k₂₁,k₀₂,lσ = θ
        μ = f([k₁₂,k₀₁,k₂₁,k₀₂,1.0])
        d = Normal.(μ,exp(σ))
        sum(loglikelihood.(d,x))
    end
    @time ξ̂_std = optimise(loglike_std,[ξ[1:3];ξ[end]];bounds=[bounds[1:3];[bounds[end]]],obj=:maximise,alg=:LN_NELDERMEAD)[2]

    @btime loglike(ξ̂)
    @btime loglike_std(ξ̂_std)

#################################################
## PROFILE

    plims = [[0.5,0.8],[0.4,0.7],[0.35,0.45],[-3.0,-2.0],[-10.0,-2.0]]
    pvec,prof,argm = profile(loglike,ξ,plims,obj=:maximise,normalise=true,npt=50)

#################################################
## MCMC TO GET CONFIDENCE INTERVAL FOR DISTRIBUTION

    prior = Product([Uniform(bound...) for bound in bounds])
    post = ξ -> insupport(prior,ξ) ? loglike(ξ) : -Inf

    X = Array{Any}(undef,6)
    @time @threads for i = 1:6
        X[i] = adaptive_rwm(ξ,post,10_000;thin=100,algorithm=:aswam).X
    end

    ξmcmc = hcat(X...)

#################################################
## Figure 6

    # Profiles
    row1 = plot(plot.(pvec,prof,c=col_skew,lw=2.0)...,link=:y,ylim=(-5.0,0.0),widen=true,label="",layout=grid(1,5))
    for i = 1:length(pvec)
        hline!(row1,subplot=i,[-1.92],c=:black,ls=:dash,label="")
        vline!(row1,subplot=i,[ξ[i]],c=:black,ls=:dot,label="")
        plot!(row1,subplot=i,xlabel=["μ₀₁","μ₂₁","μ₀₂","log(σ₂₁)","lσ"][i])
    end

    # Distribution uncertainty
    xv = range(0.2,1.0,200)
    yv = hcat([pdf.(Normal(x[2],exp(x[4])),xv) for x in eachcol(ξmcmc)]...)
    l = [quantile(y,0.025) for y in eachrow(yv)]
    u = [quantile(y,0.975) for y in eachrow(yv)]
    m = pdf.(k̂₂₁,xv)

    fig6f = plot(xv,m,ribbon=(m-l,u-m),c=:black,lw=2.0,fα=0.1,label="",xlabel="k₂₁",ylabel="Density")

    tplot = range(0.0,10.0,50)
    function fplot(θ)
        k₁₂,k₀₁,k₂₁,k₀₂,ε = θ
        x₀ = [15.0,0.0]
        sol = solve(ODEProblem(twopool_rhs!,x₀,(0.0,maximum(tplot)),[k₁₂,k₀₁,k₂₁,k₀₂]),saveat=tplot)
        [x[2] for x in sol.u] * ε
    end
    d̂ = approximate_transformed_distribution_skewed(fplot,θ̂,length(tplot))
    m_pred = mean.(d̂)
    l_pred = quantile.(d̂,0.025)
    u_pred = quantile.(d̂,0.975)

    fig6g = plot(tplot,m_pred,ribbon=(m_pred-l_pred,u_pred-m_pred),c=:black,fα=0.1,lw=2.0,label="MAP Prediction")
    [scatter!(fill(T[i],n),x[i],c=col_skew,label=i==1 ? "Data" : "") for i = 1:length(T)]
    plot!(fig6g,xlabel="Time [min]",ylabel="Concentration",ylim=(0.0,5.0),widen=true)

    row2 = plot(fig6f,fig6g)

    fig6 = plot(row1,row2,layout=@layout([a{0.3h}; b]),size=(800,400))
    add_plot_labels!(fig6)

    savefig(fig6,"$(@__DIR__)/fig6.svg")

#################################################
## Figure S2

    figS2 = plot(layout=grid(2,3))

    # Surrogate densities at each observation time
    d = approximate_transformed_distribution_skewed(f,θ,length(T))

    # Limits
    limits = [[1.0,5.0],[2.0,6.0],[2.0,4.0],[1.5,3.0],[0.8,1.75],[0.25,0.75]]

    # Compare to synthetic data
    n = 10_000
    for i = 1:6
        y = [f(rand(θ))[i] for _ = 1:n]
        density!(figS2,subplot=i,y,c=:black,lw=2.0,label="")
        plot!(figS2,subplot=i,x -> pdf(d[i],x),c=col_skew,lw=2.0,xlim=limits[i],label="")
        plot!(figS2,subplot=i,xlabel="x₂($(T[i]))")
    end

    add_plot_labels!(figS2)

    savefig(figS2,"$(@__DIR__)/figS2.svg")
