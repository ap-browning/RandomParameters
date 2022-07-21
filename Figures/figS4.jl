#=

    Figure S4

    Misspecified random parameter logistic model.

=#

using Revise
using RandomParameters
using Plots
using StatsPlots
using LinearAlgebra
using Distributions
using Optim
using AdaptiveMCMC
using .Threads
using JLD2

include("figure_defaults.jl")

#################################################
## Deterministic model

    function logistic(t,θ)
        λ,K,r₀,ε = θ
        K / (1 + (K / r₀ - 1)*exp(-λ/3*t)) + ε
    end

#################################################
## BIMODAL

    #################################################
    ## Bimodal distribution for λ

        # Target distribution
        w = 0.4 # Weights
        μ = [0.7,1.3,300.0,50.0,0.0]
        σ = [0.05,0.05,20.0,3.0,4.0]
        ξ = [logit(w);μ[1:4];log.(σ)]

        # Function to get target from inputs
        function get_θ(ξ)
            w = logis(ξ[1])
            μ = [ξ[2:5];0.0]
            σ = exp.(ξ[6:end])
            θ₁ = MvNormal([μ[1];μ[3:5]],[σ[1];σ[3:5]])
            θ₂ = MvNormal([μ[2];μ[3:5]],[σ[2];σ[3:5]])
            MixtureModel(MvNormal[θ₁,θ₂],[w,1-w])
        end
        θ = get_θ(ξ)

        # Just get λ (useful for plotting...)
        function get_λ(ξ)
            w = logis(ξ[1])
            μ = [ξ[2:5];0.0]
            σ = exp.(ξ[6:end])
            λ₁ = Normal(μ[1],σ[1])
            λ₂ = Normal(μ[2],σ[2])
            MixtureModel(Normal[λ₁,λ₂],[w,1-w])
        end
        λ = get_λ(ξ)

        # Number of observations
        n = 1000

        # Observation times
        T = 0.0:2.0:15.0

        # Data
        x = [[logistic(t,rand(θ)) for i = 1:n] for t in T]

        # Output functions
        F = [θ -> logistic(t,θ) for t in T]

        # Bounds (for MAP only)
        bounds = [[-10.0,10.0],[0.1,5.0],[0.1,5.0],[100.0,1000.0],[10.0,100.0],[-10.0,-1.0],[-10.0,-1.0],[-10.0,5.0],[-10.0,3.0],[-10.0,10.0]]

    #################################################
    ## MCMC of bimodal model

        function loglike(ξ)
            θ = get_θ(ξ)
            w = logis(ξ[1])
            sum(loglikelihood(
                MixtureModel([
                    approximate_transformed_distribution_skewed(F[i],θ.components[1]),
                    approximate_transformed_distribution_skewed(F[i],θ.components[2]),
                ],[w,1-w]),
            x[i]) for i = 1:length(T))
        end

        lb = [bound[1] for bound in bounds]
        ub = [bound[2] for bound in bounds]
        post = ξ -> all(lb .< ξ .< ub) ? loglike(ξ) : -Inf

        # Get MAP
        _,ξ̂_bimo = optimise(loglike,ξ;obj=:maximise,bounds)

        # MAP distribution estimate
        θ̂_bimo = get_θ(ξ̂_bimo)
        λ̂_bimo = get_λ(ξ̂_bimo)

    #################################################
    ## MCMC where we assume normality

    function loglike_normal(ξ)
        μ = [ξ[1:3];0.0]
        σ = exp.(ξ[4:end])
        θ = Product(Normal.(μ,σ))
        sum(loglikelihood(
            approximate_transformed_distribution_skewed(F[i],θ),
        x[i]) for i = 1:length(T))
    end

    # Get MAP
    ξ_norm0 = [0.5*(ξ[2]+ξ[3]);ξ[4:5];0.5*(ξ[6] + ξ[7]);ξ[8:end]]
    bounds_norm = [bounds[[2]];bounds[4:5];bounds[[6]];bounds[8:end]]
    lb_norm = [bound[1] for bound in bounds_norm]
    ub_norm = [bound[2] for bound in bounds_norm]
    _,ξ̂_norm = optimise(loglike_normal,ξ_norm0;obj=:maximise,bounds=bounds_norm)

    # MAP distribution estimate]
    θ̂_norm = Product(Normal.([ξ̂_norm[1:3];0.0],exp.(ξ̂_norm[4:end])))
    λ̂_norm = θ̂_norm.v[1]

#################################################
## PLOT

# (a) MAP of distributions
plt_a = plot(x -> pdf(λ̂_bimo,x),xlim=(0.5,1.5),c=col_skew,lw=2.0,label="MAP (Bimodal)")
plot!(x -> pdf(λ̂_norm,x),xlim=(0.5,1.5),c=col_norm,lw=2.0,label="MAP (Normal)")
plot!(x -> pdf(λ,x),xlim=(0.5,1.5),c=:black,lw=2.0,label="True")

# (b) MAP to get predictions - 95% PI
tplot = range(0.0,15.0,200)
m_pred_bimo = similar(tplot); l_pred_bimo = similar(tplot); u_pred_bimo = similar(tplot)
for i = 1:length(tplot)
    d = MixtureModel([
            approximate_transformed_distribution_skewed(θ -> logistic(tplot[i],θ),θ̂_bimo.components[1])
            approximate_transformed_distribution_skewed(θ -> logistic(tplot[i],θ),θ̂_bimo.components[2])
        ],[logis(ξ̂_bimo[1]),1-logis(ξ̂_bimo[1])])
    m_pred_bimo[i] = mean(d)
    l_pred_bimo[i] = quantile(d,0.025)
    u_pred_bimo[i] = quantile(d,0.975)
end

m_pred_norm = similar(tplot); l_pred_norm = similar(tplot); u_pred_norm = similar(tplot)
for i = 1:length(tplot)
    d = approximate_transformed_distribution_skewed(θ -> logistic(tplot[i],θ),θ̂_norm);
    m_pred_norm[i] = mean(d)
    l_pred_norm[i] = quantile(d,0.025)
    u_pred_norm[i] = quantile(d,0.975)
end

plt_b = plot(tplot,m_pred_bimo,ribbon=(m_pred_bimo-l_pred_bimo,u_pred_bimo-m_pred_bimo),c=col_skew,fα=0.2,lw=2.0,label="MAP Prediction (Bimodal)")
plot!(plt_b,tplot,m_pred_norm,c=col_norm,fα=0.05,ls=:dash,label="MAP Prediction (Normal)",lw=2.0)
plot!(plt_b,tplot,l_pred_norm,c=col_norm,fα=0.1,ls=:dot,label="",lw=2.0)
plot!(plt_b,tplot,u_pred_norm,c=col_norm,fα=0.1,ls=:dot,label="",lw=2.0)
violin!(plt_b,[T[1]],x[1],lw=0.0,c=:black,α=0.5,label="Data")
[violin!(plt_b,[T[i]],x[i],lw=0.0,c=:black,α=0.5,label="") for i = 2:length(T)]

# (c) MAP to get predictions - density
xw = range(75,275,200)
dw_bimo = d = MixtureModel([
    approximate_transformed_distribution_skewed(θ -> logistic(T[4],θ),θ̂_bimo.components[1])
    approximate_transformed_distribution_skewed(θ -> logistic(T[4],θ),θ̂_bimo.components[2])
],[logis(ξ̂_bimo[1]),1-logis(ξ̂_bimo[1])])
mw_bimo = pdf.(dw_bimo,xw)
mw_norm = pdf.(approximate_transformed_distribution(θ -> logistic(T[4],θ),θ̂_norm;order=3),xw)

plt_c = plot(xw,mw_bimo,c=col_skew,lw=2.0,fα=0.2,label="MAP (Bimodal)")
plot!(xw,mw_norm,c=col_norm,lw=2.0,fα=0.2,label="MAP (Normal)")
density!(x[4],c=:black,label="Data",xlim=extrema(xw),widen=true,xlabel="R(4) [µm]",ylabel="Density")

# Fig S4
figS4 = plot(plt_a,plt_b,plt_c,layout=grid(1,3),size=(1200,300))
add_plot_labels!(figS4)
savefig(figS4,"$(@__DIR__)/figS4.svg")