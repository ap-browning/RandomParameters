#=

    Figure 2

    Approximations for the univariate random parameter logistic model.

=#

using Revise
using RandomParameters
using Plots
using StatsPlots
using LinearAlgebra
using Distributions

include("figure_defaults.jl")

#################################################
## CREATE FIGURE

fig2 = plot(layout=grid(1,3))
fig2a,fig2b,fig2c = [fig2[i] for i = 1:3]

#################################################
## Deterministic model

    function logistic(t,θ)
        λ,R,r₀ = θ
        R / (1 + (R / r₀ - 1)*exp(-λ/3*t))
    end
    f(θ) = logistic(20.0,θ)

#################################################
## Mean and standard deviation of inputs

    μ = [0.5,300.0,10.0]
    σ = [0.05,50.0,1.0]

#################################################
## Fig 2(a)
#  - Independent inputs

    θ₁ = Product(Normal.(μ,σ))
    x₁ = [f(rand(θ₁)) for i = 1:100_000]

    d₁₁ = approximate_transformed_distribution(f,θ₁)
    d₁₂ = approximate_transformed_distribution(f,θ₁;order=3)

    density!(fig2a,x₁,c=:black)
    plot!(fig2a,d₁₁,c=:blue,ls=:dash)
    plot!(fig2a,d₁₂,c=:red,ls=:dot)

#################################################
## Fig 2(b)
#  - λ and R are correlated inputs

    ρ = 0.8
    Σ = diagm(σ.^2); Σ[1,2] = Σ[2,1] = ρ * prod(σ[1:2])
    θ₂ = MvNormal(μ,Σ)
    x₂ = [f(rand(θ₂)) for i = 1:100_000]

    d₂₁ = approximate_transformed_distribution(f,θ₂)
    d₂₂ = approximate_transformed_distribution(f,θ₂;order=3)

    density!(fig2b,x₂,c=:black,strokewidth=2)
    plot!(fig2b,d₂₁,c=:blue,ls=:dash)
    plot!(fig2b,d₂₂,c=:red,ls=:dot)

#################################################
## Fig 2(c)
#  - λ, R and ω are skewed

    ω = [1.0,-1.0,0.2]
    θ₃ = Product(GammaAlt.(μ,σ,ω))
    x₃ = [f(rand(θ₃)) for i = 1:100_000]

    d₃₁ = approximate_transformed_distribution(f,θ₃)
    d₃₂ = approximate_transformed_distribution(f,θ₃;order=3)

    density!(fig2c,x₃,c=:black,label="Simulated")
    plot!(fig2c,d₃₁,c=:blue,ls=:dash,label="Normal")
    plot!(fig2c,d₃₂,c=:red,ls=:dot,label="Gamma")


#################################################
## Styling

    [plot!(fig2,subplot=i,legend=:none) for i = 1:2]
    [plot!(fig2,subplot=i,xlabel="f(θ)",ylabel="Density",xlim=(40,280)) for i = 1:3]
    add_plot_labels!(fig2)
    plot!(fig2,size=(800,200),link=:all)

    savefig(fig2,"$(@__DIR__)/fig2.svg")