#=

    Figure S1

    Multivariate transformations of random parameter logistic model.

=#

using Revise
using RandomParameters
using Plots
using StatsPlots
using LinearAlgebra
using Distributions
using ColorSchemes

include("figure_defaults.jl")

#################################################
## Deterministic model

    function logistic(t,θ)
        λ,K,r₀ = θ
        K / (1 + (K / r₀ - 1)*exp(-λ/3*t))
    end
    T = [20.0,30.0,40.0]
    f(θ) = [logistic(t,θ) for t in T]

#################################################
## Input and output distributions

    μ = [0.5,300.0,10.0]
    σ = [0.05,50.0,1.0]
    ρ = 0.8
    Σ = diagm(σ.^2); Σ[1,2] = Σ[2,1] = ρ * prod(σ[1:2])
    θ = MvNormal(μ,Σ)

    x = [f(rand(θ)) for i = 1:100_000]
    x = hcat(x...)

    d = approximate_transformed_distribution(f,θ,length(T))
    _,_,ω = approximate_mean_variance_skewness(f,θ,length(T))

#################################################
## Produce plots...

    limits = [[50.0,250.0],[100.0,400.0],[100.0,450.0]]
    ticks = [50.0:50.0:250,100.0:100.0:400.0,100.0:100.0:400.0]

    figS1 = plot(layout=grid(3,3))

    # Marginals
    for i = 1:3
        # Plot index
        idx = 4i - 3
        # Plot data
        density!(figS1,subplot=idx,x[i,:],lw=1.5,c=:black)
        # Plot surrogate
        plot!(figS1,subplot=idx,Normal(d.μ[i],sqrt(d.Σ[i,i])),c=col_norm,ls=:dash,lw=2.0)
        # Plot skewed surrogate
        plot!(figS1,subplot=idx,GammaAlt(d.μ[i],sqrt(d.Σ[i,i]),ω[i]),c=col_skew,ls=:dot,lw=2.0)
        # Settings
        plot!(figS1,subplot=idx,
            legend=:none,
            xlimits=limits[i],
            xticks=ticks[i],
            widen=false,
            axis=:x,yticks=[],box=:off
        )
    end

    # 2D Marginals
    for i = 1:2,j = i+1:3
        # Plot index
        idx = 3(j-1)+i
        # Plot data
        r = density2d!(figS1,subplot=idx,x[i,:],x[j,:],color=cgrad(:bone,rev=true),fill=true,lw=0.0,levels=5)
        # Plot surrogate
        d_marginal = MvNormal(d.μ[[i,j]],d.Σ[[i,j],[i,j]])
        xgrid = range(limits[i]...,500)
        ygrid = range(limits[j]...,500)
        pdf_marginal = [pdf(d_marginal,[a,b]) for a in xgrid, b in ygrid]
        contour!(figS1,subplot=idx,xgrid,ygrid,pdf_marginal',xlim=limits[i],ylim=limits[j],levels=5,c=col_norm,ls=:dash)
        # Plot skewed surrogate
        d_marginal = MvGamma(d.μ[[i,j]],d.Σ[[i,j],[i,j]],ω[[i,j]])
        xgrid = range(limits[i]...,500)
        ygrid = range(limits[j]...,500)
        pdf_marginal = [pdf(d_marginal,[a,b]) for a in xgrid, b in ygrid]
        contour!(figS1,subplot=idx,xgrid,ygrid,pdf_marginal',xlim=limits[i],ylim=limits[j],levels=5,c=col_skew,ls=:dot)
        # Settings
        plot!(figS1,subplot=idx,
            legend=:none,
            xticks=ticks[i],
            yticks=ticks[j],
            widen=false,
        )
        plot!(figS1,subplot=3(i-1)+j,box=:none,grid=:none)
    end

    # Styling
    plot!(figS1,subplot=1,xticks=(ticks[1],""))
    plot!(figS1,subplot=4,xticks=(ticks[1],""))
    plot!(figS1,subplot=5,xticks=(ticks[2],""))
    plot!(figS1,subplot=8,yticks=(ticks[3],""))
    plot!(figS1,top_margin=-1Plots.mm,left_margin=-3.0Plots.mm)
    plot!(figS1,subplot=7,xlabel="f₁(θ) [$(Int(T[1])) min]")
    plot!(figS1,subplot=8,xlabel="f₂(θ) [$(Int(T[2])) min]")
    plot!(figS1,subplot=9,xlabel="f₃(θ) [$(Int(T[3])) min]")
    plot!(figS1,subplot=4,ylabel="f₂(θ) [$(Int(T[2])) min]")
    plot!(figS1,subplot=7,ylabel="f₃(θ) [$(Int(T[3])) min]")

    plot!(figS1,size=(500,500))

    savefig(figS1,"$(@__DIR__)/figS1.svg")
