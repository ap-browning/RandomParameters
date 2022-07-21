#=

    Figure 4

    Random parameter logistic model with unknown correlation and skewness.

=#

using Revise
using RandomParameters
using Plots
using StatsPlots
using LinearAlgebra
using Distributions
using Optim

include("figure_defaults.jl")

#################################################
## Deterministic model

    function logistic(t,θ)
        λ,R,r₀,ε = θ
        R / (1 + (R / r₀ - 1)*exp(-λ/3*t)) + ε
    end

#################################################
## PART 1: CORRELATION

    #################################################
    ## Synthetic data

        # Target distribution
        ρ = 0.6
        μ = [1.0,300.0,50.0,0.0]
        σ = [0.05,20.0,3.0,4.0]
        ξ = [ρ;μ[1:3];log.(σ)]

        Σ = diagm(σ.^2); Σ[1,2] = Σ[2,1] = ρ * prod(σ[1:2])
        θ = MvNormal(μ,Σ)

        # Number of observations
        N = [10,100,1000]

        # Observation times
        T = 0.0:2.0:15.0

        # Data
        x_all = [[logistic(t,rand(θ)) for i = 1:maximum(N)] for t in T]
        x = [[x_all[i][1:n] for i = 1:length(T)] for n in N]

        # Output functions
        F = [θ -> logistic(t,θ) for t in T]


    #################################################
    ## Settings

        bounds = [[-0.9,0.9],[0.8,5.0],[100.0,1000.0],[10.0,100.0],[-10.0,-1.0],[-10.0,5.0],[-10.0,3.0],[-10.0,10.0]]
        pvec = range(-0.9,0.9,20)

        # Fix problem plotting profiles
        fix_prof(x) = (x[x .< -10.0] .= -10.0; x)

    #################################################
    ## Profile using normal approximation

    function loglike(ξ,x)
        ρ = ξ[1]
        μ = [ξ[2:4];0.0]
        σ = exp.(ξ[5:end])
        Σ = diagm(σ.^2); Σ[1,2] = Σ[2,1] = ρ * prod(σ[1:2])
        θ = MvNormal(μ,Σ)
        sum(loglikelihood(
            approximate_transformed_distribution(F[i],θ),
        x[i]) for i = 1:length(T))
    end

    prof_normal = [profile(ξ -> loglike(ξ,xᵢ),ξ,pvec,1;bounds,obj=:maximise,normalise=true,npt=50)[2] for xᵢ in x]

    #################################################
    ## Profile using skewed approximation

    function loglike_skewed(ξ,x)
        ρ = ξ[1]
        μ = [ξ[2:4];0.0]
        σ = exp.(ξ[5:end])
        Σ = diagm(σ.^2); Σ[1,2] = Σ[2,1] = ρ * prod(σ[1:2])
        θ = MvNormal(μ,Σ)
        sum(loglikelihood(
            approximate_transformed_distribution_skewed(F[i],θ),
        x[i]) for i = 1:length(T))
    end

    prof_skewed = [profile(ξ -> loglike_skewed(ξ,xᵢ),ξ,pvec,1;bounds,obj=:maximise,normalise=true,npt=50)[2] for xᵢ in x]

    #################################################
    ## Fig 4

    fig4a = plot(pvec,fix_prof.(prof_normal),ylim=(-5.0,0.0),c=col_norm,ls=[:solid :dash :dot],label=["10 (Normal)" "100" "1000"])
    plot!(pvec,fix_prof.(prof_skewed),ylim=(-5.0,0.0),c=col_skew,ls=[:solid :dash :dot],label=["10 (Gamma)" "100" "1000"])
    hline!([-1.92],c=:black,ls=:dash,label="")
    vline!([ρ],c=:red,label="")
    plot!(xlim=(-0.9,0.9),xticks=-0.9:0.3:0.9,xlabel="ρλR",ylabel="PLL",widen=true,legend=:bottomright,size=(400,300))

#################################################
## PART 2: SKEWNESS

    #################################################
    ## Synthetic data

        # Target distribution
        ω = [-1.5,0.0,0.0,0.0]
        μ = [1.0,300.0,50.0,0.0]
        σ = [0.05,20.0,3.0,4.0]
        ξ = [ω[1];μ[1:3];log.(σ)]

        θ = Product(GammaAlt.(μ,σ,ω))

        # Number of observations
        N = [10,100,1000]

        # Observation times
        T = 0.0:2.0:15.0

        # Data
        x_all = [[logistic(t,rand(θ)) for i = 1:maximum(N)] for t in T]
        x = [[x_all[i][1:n] for i = 1:length(T)] for n in N]

        # Output functions
        F = [θ -> logistic(t,θ) for t in T]


    #################################################
    ## Settings

        bounds = [[-2.0,2.0],[0.8,5.0],[100.0,1000.0],[10.0,100.0],[-10.0,-1.0],[-10.0,5.0],[-10.0,3.0],[-10.0,10.0]]
        pvec = range(-2.0,1.0,20)

        # Fix problem plotting profiles
        fix_prof(x) = (x[x .< -10.0] .= -10.0; x)

    #################################################
    ## Profile using normal approximation

    function loglike(ξ,x)
        ω = [ξ[1],0.0,0.0,0.0]
        μ = [ξ[2:4];0.0]
        σ = exp.(ξ[5:end])
        θ = Product(GammaAlt.(μ,σ,ω))
        sum(loglikelihood(
            approximate_transformed_distribution(F[i],θ),
        x[i]) for i = 1:length(T))
    end

    prof_normal = [profile(ξ -> loglike(ξ,xᵢ),ξ,pvec,1;bounds,obj=:maximise,normalise=true,npt=50)[2] for xᵢ in x]

    #################################################
    ## Profile using skewed approximation

    function loglike_skewed(ξ,x)
        ω = [ξ[1],0.0,0.0,0.0]
        μ = [ξ[2:4];0.0]
        σ = exp.(ξ[5:end])
        θ = Product(GammaAlt.(μ,σ,ω))
        sum(loglikelihood(
            approximate_transformed_distribution_skewed(F[i],θ),
        x[i]) for i = 1:length(T))
    end

    prof_skewed = [profile(ξ -> loglike_skewed(ξ,xᵢ),ξ,pvec,1;bounds,obj=:maximise,normalise=true,npt=50)[2] for xᵢ in x]

    #################################################
    ## Fig 4b

    fig4b = plot(pvec,fix_prof.(prof_normal),ylim=(-5.0,0.0),c=col_norm,ls=[:solid :dash :dot],label=["10 (Normal)" "100" "1000"])
    plot!(pvec,fix_prof.(prof_skewed),ylim=(-5.0,0.0),c=col_skew,ls=[:solid :dash :dot],label=["10 (Gamma)" "100" "1000"])
    hline!([-1.92],c=:black,ls=:dash,label="")
    vline!([ω[1]],c=:red,label="")
    plot!(xlim=(-2.0,1.0),xticks=-2.0:0.5:1.0,xlabel="ωλ",ylabel="PLL",widen=true,legend=:bottomright,size=(400,300))

    
#################################################
## Fig 4

    fig4 = plot(fig4a,fig4b,size=(800,300))
    add_plot_labels!(fig4)

    savefig(fig4,"$(@__DIR__)/fig4.svg")