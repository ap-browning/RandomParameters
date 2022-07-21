#=

    Figure 1

    Spheroid data and logistic model fit.

=#

using Revise
using RandomParameters
using Plots
using StatsPlots
using LinearAlgebra
using CSV, DataFrames, DataFramesMeta
using Random

include("figure_defaults.jl")


#################################################
## Download and process data

    data = CSV.read(download("https://github.com/ap-browning/Spheroids/raw/main/Data/ConfocalData.csv"),DataFrame)
    data = @subset(data,:CellLine .== "983b", :InitialCondition .== 5000)

    days = [0.0,2.0,4.0,7.0,9.0,11.0,13.0,15.0] .+ 3.0
    data = @subset(data,[day ∈ days for day in data.Day])

#################################################
## Deterministic model

    function logistic(t,θ)
        λ,R,r₀ = θ
        R / (1 + (R / r₀ - 1)*exp(-λ/3*(t-3)))
    end

#################################################
## Fit logistic model

    function loglike(θ)
        lσ = θ[end]
        μ = [logistic(t,θ) for t in data.Day]
        loglikelihood(Normal(0.0,exp(lσ)),μ - data.R)
    end

    θ̂ = optimise(loglike,[1.0,300.0,150.0,0.0],obj=:maximise)[2]

#################################################
## Produce plots

    fig1b = plot(t -> logistic(t,θ̂),ribbon=1.96*exp(θ̂[end]),xlim=(3.0,18.0),c=:black,fα=0.1)
    @df data boxplot!(:Day,:R,c=col_skew,lc=col_skew,bar_width=0.8,outliers=false,whisker_width=:match,α=1.0)
    @df data scatter!(fig1b,:Day,:R,c=col_norm,msw=0.0,ms=4.0,legend=:none,α=1.0)
    plot!(legend=:none,widen=true,xticks=3:5:18,xlabel="Time [d]",ylabel="Radius [µm]",size=(400,300))
    add_plot_labels!(fig1b,offset=1)

    savefig(fig1b,"$(@__DIR__)/fig1b.svg")
    fig1b