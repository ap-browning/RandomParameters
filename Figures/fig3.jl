#=

    Figure 3

    Profile likelihood analysis of random parameter logistic model.

=#

using Revise
using RandomParameters
using Plots
using StatsPlots
using LinearAlgebra
using Distributions
using Optim
using BenchmarkTools

include("figure_defaults.jl")

#################################################
## Deterministic model

    function logistic(t,θ)
        λ,R,r₀,ε = θ
        R / (1 + (R / r₀ - 1)*exp(-λ/3*t)) + ε
    end

#################################################
## Synthetic data

    # Target distribution
    ξ = [[1.0,300.0,50.0];log.([0.05,20.0,3.0,4.0])]
    θ = Product(Normal.([ξ[1:3];0.0],exp.(ξ[4:end])))

    # Number of observations
    n = 10

    # Observation times
    T = 0.0:2.0:15.0

    # Data
    x = [[logistic(t,rand(θ)) for i = 1:n] for t in T]

    # Output functions
    F = [θ -> logistic(t,θ) for t in T]

    # temp
    f = θ -> [logistic(t,θ) for t in T]

#################################################
## Setup plots

    row2 = plot(layout=grid(1,7))
    row3 = plot(layout=grid(1,7))
    row4 = plot(layout=grid(1,7))

    # Fix problem plotting profiles
    fix_prof(x) = (x[x .< -10.0] .= -10.0; x)

#################################################
## Settings...

    plims = [[0.8,1.2],[250.0,350.0],[40.0,60.0],[-7.0,-2.0],[2.0,4.0],[-3.0,3.0],[-2.0,4.0]]
    bounds = [[0.8,5.0],[100.0,1000.0],[10.0,100.0],[-10.0,-1.0],[-10.0,5.0],[-10.0,3.0],[-10.0,10.0]]
    plims_std = [plims[[1,2,3]]; [[1.0,4.0]]]

#################################################
## Profile standard additive noise model

    function loglike_std(ξ)
        θ = ξ[1:3]
        σ = exp(ξ[end])
        sum(loglikelihood(
            Normal(logistic(T[i],[θ;0.0]),σ),
        x[i]) for i = 1:length(T))
    end

    ξ̂_std = optimise(loglike_std,[ξ[1:3];ξ[end]],obj=:maximise)[2]

    # Profile all parameters
    pvec,prof,argm = profile(loglike_std,[ξ[1:3];ξ[end]],plims_std,obj=:maximise,normalise=true)
    for i = 1:4
        j = i < 4 ? i : 7
        plot!(row2,subplot=j,pvec[i],prof[i],c=col_std)
    end

    # Benchmark standard model likelihood evaluation
    @btime loglike_std(ξ̂_std)

#################################################
## Profile surrogate model (Normal)

    function loglike(ξ)
        θ = Product(Normal.([ξ[1:3];0.0],exp.(ξ[4:end])))
        sum(loglikelihood(
            approximate_transformed_distribution(F[i],θ),
        x[i]) for i = 1:length(T))
    end

    # Profile all parameters
    pvec,prof,argm = profile(loglike,ξ,plims;bounds,npt=30,obj=:maximise,normalise=true)
    for i = 1:7
        plot!(row2,subplot=i,pvec[i],fix_prof(prof[i]),c=col_norm,ls=:dash)
    end

    # Profile with known noise
    pvec,prof,argm = profile(x -> loglike([x; ξ[end]]),ξ[1:6],plims[1:6];bounds=bounds[1:6],npt=30,obj=:maximise,normalise=true)
    for i = 1:6
        plot!(row3,subplot=i,pvec[i],fix_prof(prof[i]),c=col_norm,ls=:dash)
    end

    # Profile with zero noise
    pvec,prof,argm = profile(x -> loglike([x; -Inf]),ξ[1:6],plims[1:6];bounds=bounds[1:6],npt=30,obj=:maximise,normalise=true)
    for i = 1:6
        plot!(row4,subplot=i,pvec[i],fix_prof(prof[i]),c=col_norm,ls=:dash)
    end

    # Benchmark standard model likelihood evaluation
    @btime loglike_std(ξ̂_std)

#################################################
## Profile surrogate skewed model

    function loglike_skewed(ξ)
        θ = Product(Normal.([ξ[1:3];0.0],exp.(ξ[4:end])))
        sum(loglikelihood(
            approximate_transformed_distribution(F[i],θ;order=3),
        x[i]) for i = 1:length(T))
    end

    ξ̂_skewed = optimise(loglike_skewed,ξ;obj=:maximise,bounds)[2]
    θ̂_skewed = Product(Normal.([ξ̂_skewed[1:3];0.0],exp.(ξ̂_skewed[4:end])))

    # Profile all parameters
    pvec,prof,argm = profile(loglike_skewed,ξ,plims;bounds,npt=30,obj=:maximise,normalise=true)
    for i = 1:7
        plot!(row2,subplot=i,pvec[i],fix_prof(prof[i]),c=col_skew,ls=:dot)
    end

    # Profile with known noise
    pvec,prof,argm = profile(x -> loglike_skewed([x; ξ[end]]),ξ[1:6],plims[1:6];bounds=bounds[1:6],npt=30,obj=:maximise,normalise=true)
    for i = 1:6
        plot!(row3,subplot=i,pvec[i],fix_prof(prof[i]),c=col_skew,ls=:dot)
    end

    # Profile with zero noise
    pvec,prof,argm = profile(x -> loglike_skewed([x; -Inf]),ξ[1:6],plims[1:6];bounds=bounds[1:6],npt=30,obj=:maximise,normalise=true)
    for i = 1:6
        plot!(row4,subplot=i,pvec[i],fix_prof(prof[i]),c=col_skew,ls=:dot)
    end

    # Benchmark random-parameter model likelihood evaluation
    @btime loglike_skewed(ξ̂_skewed)

#################################################
## Plots...

    # (a) Standard model
    fig3a = plot(t -> logistic(t,[ξ̂_std[1:3];0.0]),ribbon=1.96*exp(ξ̂_std[end]),xlim=(0.0,15.0),c=:black,fα=0.1)
    scatter!(fig3a,T,hcat(x...)',c=col_blue,legend=:none,widen=true,xlabel="Time [d]",ylabel="Radius [µm]")

    # (b) RODE model
    tplot = range(0.0,15.0,200)
    m = similar(tplot); l = similar(tplot); u = similar(tplot)
    for i = 1:length(tplot)
        d = approximate_transformed_distribution(θ -> logistic(tplot[i],θ),θ̂_skewed;order=3);
        m[i] = mean(d)
        l[i] = quantile(d,0.025)
        u[i] = quantile(d,0.975)
    end
    fig3b = plot(tplot,m,ribbon=(m-l,u-m),c=:black,fα=0.1)
    scatter!(T,hcat(x...)',c=col_blue,legend=:none,xlabel="Time [d]",ylabel="Radius [µm]")

    # (c), (d), (e)
    plot!(row2,link=:y,ylim=(-5.0,0.0),widen=true,legend=:none,xlabel="p")
    plot!(row3,link=:y,ylim=(-5.0,0.0),widen=true,legend=:none,xlabel="p")
    plot!(row4,link=:y,ylim=(-5.0,0.0),widen=true,legend=:none,xlabel="p")
    for row in [row2,row3,row4], i = 1:7
        hline!(row,subplot=i,[-1.92],ls=:dash,c=:black)
        vline!(row,subplot=i,[ξ[i]],c=:black)
    end


    # (a,b)
    fig3 = plot(fig3a,fig3b,row2,row3,row4,layout=@layout([grid(1,2){0.5h}; b; c; d]),size=(1200,800))
    add_plot_labels!(fig3)
    savefig(fig3,"$(@__DIR__)/fig3.svg")