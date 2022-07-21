#=
    Figure 7 and Figure S3

    Simulation results for non-linear two-pool model with dependent observations.

    Observations are taken at (b - c) a single time point, and (a,d) multiple time points.

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

#################################################
## (a) Multiple observation points

    n = 20
    Θ = rand(MersenneTwister(2),θ,n*length(T)); Θ = reshape(Θ,length(θ),n,length(T))
    x = [hcat([twopool_nonlin(Θ[:,j,i])[i,:] for j = 1:n]...) for i = 1:length(T)]
    
    # Plot 95% confidence interval based on transformation
    tplot = range(0.0,10.0,100)
    function twopool_continuous(θ)
        k₂₁,V₂₁,k₀₁,k₀₂,ε₁,ε₂ = θ
        x₀ = [15.0,0.0]
        sol = solve(ODEProblem(twopool_nonlin_rhs!,x₀,(0.0,maximum(T)),[k₂₁,V₂₁,k₀₁,k₀₂]),saveat=tplot)
        x = hcat(sol.(tplot)...)'
        x[:,1] *= ε₁
        x[:,2] .+= ε₂
        return x
    end
    d = approximate_transformed_distribution(twopool_continuous,θ,length(tplot),2;order=3)

    # Get mean to plot
    μ₁ = [mean(dᵢ.v[1]) for dᵢ in d]
    μ₂ = [mean(dᵢ.v[2]) for dᵢ in d]
    l₁ = [quantile(dᵢ.v[1],0.025) for dᵢ in d]
    l₂ = [quantile(dᵢ.v[2],0.025) for dᵢ in d]
    u₁ = [quantile(dᵢ.v[1],0.975) for dᵢ in d]
    u₂ = [quantile(dᵢ.v[2],0.975) for dᵢ in d]
    
    fig7a = plot(sol,c=[col_blue col_orng],label=["x1(t)" "x2(t)"],lw=2.0)

    fig7a = plot(tplot,μ₁,c=col_blue,lw=2.0,label="x1(t)",ribbon=(μ₁-l₁,u₁-μ₁))
    plot!(tplot,μ₂,c=col_orng,lw=2.0,label="x2(t)",ribbon=(μ₂-l₂,u₂-μ₂))
    for i = 1:length(T)
        scatter!(fig7a,T[i]*ones(n),x[i][1,:],c=col_blue,label=i==1 ? "f1" : "")
        scatter!(fig7a,T[i]*ones(n),x[i][2,:],c=col_orng,label=i==1 ? "f2" : "")
    end
    plot!(fig7a,ylim=(0.0,10.0),xlabel="Time [min]",ylabel="Concentration",widen=true)
    plot!(fig7a,size=(500,400))

    savefig(fig7a,"$(@__DIR__)/fig7a.svg")

#################################################
## (b) Single observation point

    n_fine = 100_000
    x_fine = hcat([twopool_nonlin_single(rand(θ)) for i = 1:n_fine]...)

    n_single = 100
    x_single = hcat([twopool_nonlin_single(θᵢ) for θᵢ = eachcol(rand(MersenneTwister(1),θ,n_single))]...)

    # Get transformation
    d = approximate_transformed_distribution(twopool_nonlin_single,θ,2;order=3,independent=false)

    # Plot data
    plt_biv = density2d(eachrow(x_fine)...,color=cgrad(:bone,rev=true),fill=true,lw=0.0,levels=5)
    plot!(xlim=(0.25,1.25),ylim=(0.5,0.71))
    plot!(aspect_ratio=(-(xlims(plt_biv)...)) / (-(ylims(plt_biv)...)))

    # Plot distribution
    xv = range(xlims(plt_biv)...,200)
    yv = range(ylims(plt_biv)...,201)
    dv = [pdf(d,[xx,yy]) for xx in xv, yy in yv]
    contour!(plt_biv,xv,yv,dv',levels=5,c=col_skew,lw=2.0)
    plot!(plt_biv,xticks=(xticks(plt_biv)[1][1],[]),yticks=yticks(plt_biv)[1][1],ylabel="f1")

    # Plot data
    scatter!(eachrow(x_single)...,c=col_blue)

    # Marginals (second needs to be rotated...)
    plt_x1 = density(x_fine[1,:],c=:black,lw=2.0,xlim=xlims(plt_biv))
    plot!(plt_x1,d.v[1],c=col_skew,lw=2.0,ls=:dash)
    plot!(plt_x1,ylim=(0.0,ylims(plt_x1)[2]),xticks=xticks(plt_biv)[1][1],xlabel="f2")

    plt_x2 = plot(pdf(kde(x_fine[2,:]),yv),yv,c=:black,lw=2.0,ylim=ylims(plt_biv))
    plot!(plt_x2,pdf.(d.v[2],yv),yv,c=col_skew,lw=2.0,ls=:dash)
    plot!(plt_x2,xlim=(0.0,xlims(plt_x2)[2]),yticks=(yticks(plt_biv)[1][1],[]))

    plot!(plt_x1,aspect_ratio=1/3*(-(xlims(plt_x1)...)) / (-(ylims(plt_x1)...)))
    plot!(plt_x2,aspect_ratio=3*(-(xlims(plt_x2)...)) / (-(ylims(plt_x2)...)))

    [plot!(plt_x1,x_single[1,j]*[1.0,1.0],[0.0,0.1*ylims(plt_x1)[2]],label="",c=col_blue) for j = 1:n_single]
    [plot!(plt_x2,[0.0,0.1*xlims(plt_x2)[2]],x_single[2,j]*[1.0,1.0],label="",c=col_blue) for j = 1:n_single]

    # Put together
    layout = @layout [biv{0.6w,0.6h} x2
                      x1  _ ]
    fig7b = plot(plt_biv,plt_x2,plt_x1;layout,size=(500,500),margin=0Plots.mm,legend=:none)

    savefig(fig7b,"$(@__DIR__)/fig7b.svg")

    # Extra plot (boxplot)
    fig7_inset = boxplot(x_single',c=[col_norm col_skew],lc=[col_norm col_skew],bar_width=0.7,xticks=[1,2],xlim=(0.5,2.5),ymirror=true,legend=:none,ylim=(0.0,1.3))
    savefig(fig7_inset,"$(@__DIR__)/fig7_inset.svg")

#################################################
## Figure S3

figS3 = plot(layout=grid(2,3))

    # Fine data
    n_fine = 100_000
    #x_fine = [hcat([twopool_nonlin(rand(θ))[i,:] for j = 1:n_fine]...) for i = 1:length(T)]

    # Get distributions
    d = approximate_transformed_distribution(twopool_nonlin,θ,length(T),2;order=3)

    # Limits to plot
    xlimits = [[4,7],[1.5,4],[0.5,3],[0.5,2.0],[0.0,1.5]]
    ylimits = [[3.5,5.6],[3,4],[1.9,2.2],[1,1.2],[0.5,0.7]]

    # Loop through observation times
    for i = 1:length(T)
        density2d!(figS3,subplot=i,eachrow(x_fine[i])...,color=cgrad(:bone,rev=true),fill=true,lw=0.0,levels=5)
        plot!(figS3,subplot=i,xlims=xlimits[i],ylims=ylimits[i])
        xv = range(xlimits[i]...,200)
        yv = range(ylimits[i]...,201)
        dv = [pdf(d[i],[xx,yy]) for xx in xv, yy in yv]
        contour!(figS3,subplot=i,xv,yv,dv',levels=5,c=col_skew,lw=2.0)
        plot!(figS3,subplot=i,xlabel="f1",ylabel="f2",title="t = $(Int(T[i]))")
        scatter!(figS3,subplot=i,eachrow(x[i])...,c=col_blue)
    end

    plot!(figS3,subplot=6,axis=:none,box=:none)
    #add_plot_labels!(figS3)

    savefig(figS3,"$(@__DIR__)/figS3.svg")