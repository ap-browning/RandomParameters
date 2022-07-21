#=

    Figure 5

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
        λ,R,r₀,ε = θ
        R / (1 + (R / r₀ - 1)*exp(-λ/3*t)) + ε
    end

#################################################
## PART 1: SKEWED

    #################################################
    ## Skewed distribution for λ

        # Target distribution
        ω = [-1.5,0.0,0.0,0.0]
        μ = [1.0,300.0,50.0,0.0]
        σ = [0.05,20.0,3.0,4.0]
        ξ = [ω[1];μ[1:3];log.(σ)]

        θ = Product(GammaAlt.(μ,σ,ω))

        # Number of observations
        n = 1000

        # Observation times
        T = 0.0:2.0:15.0

        # Data
        x = [[logistic(t,rand(θ)) for i = 1:n] for t in T]

        # Output functions
        F = [θ -> logistic(t,θ) for t in T]

        # Bounds (for MAP only)
        bounds = [[-2.0,2.0],[0.8,5.0],[100.0,1000.0],[10.0,100.0],[-10.0,-1.0],[-10.0,5.0],[-10.0,3.0],[-10.0,10.0]]

    #################################################
    ## MCMC of skewed model

        function loglike(ξ)
            ω = [ξ[1],0.0,0.0,0.0]
            μ = [ξ[2:4];0.0]
            σ = exp.(ξ[5:end])
            θ = Product(GammaAlt.(μ,σ,ω))
            sum(loglikelihood(
                approximate_transformed_distribution_skewed(F[i],θ),
            x[i]) for i = 1:length(T))
        end

        # Get MAP
        _,ξ̂_skew = optimise(loglike,ξ;obj=:maximise,bounds)

        # MAP distribution estimate
        θ̂_skew = Product(GammaAlt.([ξ̂_skew[2:4];0.0],exp.(ξ̂_skew[5:end]),[ξ̂_skew[1],0.0,0.0,0.0]))

        # Run 6 chains (one per core, each of length 10_000)
        X_skew = Array{Any}(undef,6)
        @time @threads for i = 1:6
            X_skew[i] = adaptive_rwm(ξ,loglike,100_000;thin=100,algorithm=:aswam).X
        end

        # Pool samples
        ξmcmc_skew = hcat(X_skew...)

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
        _,ξ̂_norm = optimise(loglike_normal,ξ[2:end];obj=:maximise,bounds=bounds[2:end])

        # MAP distribution estimate]
        θ̂_norm = Product(Normal.([ξ̂_norm[1:3];0.0],exp.(ξ̂_norm[4:end])))

        # Run 6 chains (one per core, each of length 10_000)
        X_norm = Array{Any}(undef,6)
        @time @threads for i = 1:6
            X_norm[i] = adaptive_rwm(ξ[2:end],loglike_normal,100_000;thin=100,algorithm=:aswam).X
        end

        # Pool samples
        ξmcmc_norm = hcat(X_norm...)

        # Save
        @save "$(@__DIR__)/fig5_results_skewed.jld2" ξ̂_skew θ̂_skew ξmcmc_skew ξ̂_norm θ̂_norm ξmcmc_norm

    #################################################
    ## Plot

        @load "Figures/fig5_results_skewed.jld2"

        xv = range(0.8,1.1,200)
        yv = hcat([pdf.(GammaAlt(x[2],exp(x[5]),x[1]),xv) for x in eachcol(ξmcmc_skew[:,1:10:end])]...)
        l = [quantile(y,0.025) for y in eachrow(yv)]
        u = [quantile(y,0.975) for y in eachrow(yv)]
        λ̂_skew = GammaAlt(ξ̂_skew[2],exp(ξ̂_skew[5]),ξ̂_skew[1])
        m = pdf.(λ̂_skew,xv)

        fig5a = plot(xv,m,ribbon=(m-l,u-m),c=col_skew,lw=2.0,fα=0.1,label="MAP (Gamma)")
        plot!(xv,x->pdf(θ.v[1],x),c=:black,lw=2.0,label="True (Gamma)")

        xv = range(0.8,1.2,200)
        yv = hcat([pdf.(GammaAlt(x[1],exp(x[4]),0.0),xv) for x in eachcol(ξmcmc_norm[:,1:10:end])]...)
        l = [quantile(y,0.025) for y in eachrow(yv)]
        u = [quantile(y,0.975) for y in eachrow(yv)]
        λ̂_norm = GammaAlt(ξ̂_norm[1],exp(ξ̂_norm[4]),0.0)
        m = pdf.(λ̂_norm,xv)

        fig5b = plot(xv,m,ribbon=(m-l,u-m),c=col_norm,lw=2.0,fα=0.1,label="MAP (Normal)")
        plot!(xv,x->pdf(θ.v[1],x),c=:black,lw=2.0,label="True (Gamma)")

        ## (c) Predictions
        tplot = range(0.0,15.0,200)
        m_pred_skew = similar(tplot); l_pred_skew = similar(tplot); u_pred_skew = similar(tplot)
        for i = 1:length(tplot)
            d = approximate_transformed_distribution_skewed(θ -> logistic(tplot[i],θ),θ̂_skew);
            m_pred_skew[i] = mean(d)
            l_pred_skew[i] = quantile(d,0.025)
            u_pred_skew[i] = quantile(d,0.975)
        end

        m_pred_norm = similar(tplot); l_pred_norm = similar(tplot); u_pred_norm = similar(tplot)
        for i = 1:length(tplot)
            d = approximate_transformed_distribution_skewed(θ -> logistic(tplot[i],θ),θ̂_norm);
            m_pred_norm[i] = mean(d)
            l_pred_norm[i] = quantile(d,0.025)
            u_pred_norm[i] = quantile(d,0.975)
        end

        fig5c = plot(tplot,m_pred_skew,ribbon=(m_pred_skew-l_pred_skew,u_pred_skew-m_pred_skew),c=col_skew,fα=0.2,lw=2.0,label="MAP Prediction (Gamma)")
        plot!(tplot,m_pred_norm,c=col_norm,fα=0.1,ls=:dash,label="MAP Prediction (Normal)")
        plot!(tplot,l_pred_norm,c=col_norm,fα=0.1,ls=:dot,label="")
        plot!(tplot,u_pred_norm,c=col_norm,fα=0.1,ls=:dot,label="")
        violin!(fig5c,[T[1]],x[1],lw=0.0,c=:black,α=0.5,label="Data")
        [violin!(fig5c,[T[i]],x[i],lw=0.0,c=:black,α=0.5,label="") for i = 2:length(T)]
        
        # t = 6 d
        xw = range(125,225,200)
        ξw_skew = ξmcmc_skew[:,1:10:end]
        yw_skew = zeros(length(xw),size(ξw_skew,2))
        ξw_norm = ξmcmc_norm[:,1:10:end]
        yw_norm = zeros(length(xw),size(ξw_norm,2))
        for i = 1:size(ξw_skew,2)
            θw_skew = Product(GammaAlt.([ξw_skew[2:4,i];0.0],exp.(ξw_skew[5:end,i]),[ξw_skew[1,i],0.0,0.0,0.0]))
            dw_skew = approximate_transformed_distribution(θ -> logistic(T[4],θ),θw_skew;order=3)
            yw_skew[:,i] = pdf.(dw_skew,xw) 
            θw_norm = Product(Normal.([ξw_norm[1:3,i];0.0],exp.(ξw_norm[4:end,i])))
            dw_norm = approximate_transformed_distribution(θ -> logistic(T[4],θ),θw_norm;order=3)
            yw_norm[:,i] = pdf.(dw_norm,xw) 
        end
        lw_skew = [quantile(y,0.025) for y in eachrow(yw_skew)]
        uw_skew = [quantile(y,0.975) for y in eachrow(yw_skew)]
        lw_norm = [quantile(y,0.025) for y in eachrow(yw_norm)]
        uw_norm = [quantile(y,0.975) for y in eachrow(yw_norm)]
        mw_skew = pdf.(approximate_transformed_distribution(θ -> logistic(T[4],θ),θ̂_skew;order=3),xw)
        mw_norm = pdf.(approximate_transformed_distribution(θ -> logistic(T[4],θ),θ̂_norm;order=3),xw)

        fig5d = plot(xw,mw_skew,ribbon=(mw_skew-lw_skew,uw_skew-mw_skew),c=col_skew,lw=2.0,fα=0.2,label="MAP (Gamma)")
        plot!(xw,mw_norm,ribbon=(mw_norm-lw_norm,uw_norm-mw_norm),c=col_norm,lw=2.0,fα=0.2,label="MAP (Normal)")
        density!(x[4],c=:black,label="Data",xlim=extrema(xw),widen=true,xlabel="R(4) [µm]",ylabel="Density")

        row1 = plot(fig5a,fig5b,fig5c,fig5d,layout=grid(1,4))
        plot!(row1,subplot=1,xlabel="λ",ylabel="Density",xlim=(0.8,1.15),ylim=(0.0,15.0),widen=true,legend=:topleft)
        plot!(row1,subplot=2,xlabel="λ",ylabel="Density",xlim=(0.8,1.15),ylim=(0.0,15.0),widen=true,legend=:topleft)
        plot!(row1,subplot=3,xlabel="Time [d]",ylabel="Radius [µm]",widen=true,legend=:bottomright)

#################################################
## PART 2: BIMODAL

    #################################################
    ## Bimodal distribution for λ

        # Target distribution
        w = 0.4 # Weights
        μ = [0.9,1.1,300.0,50.0,0.0]
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

        # Number of observations
        n = 1000

        # Observation times
        T = 0.0:2.0:15.0

        # Data
        x = [[logistic(t,rand(θ)) for i = 1:n] for t in T]

        # Output functions
        F = [θ -> logistic(t,θ) for t in T]

        # Bounds (for MAP only)
        bounds = [[-10.0,10.0],[0.8,5.0],[0.8,5.0],[100.0,1000.0],[10.0,100.0],[-10.0,-1.0],[-10.0,-1.0],[-10.0,5.0],[-10.0,3.0],[-10.0,10.0]]

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

        # Run 6 chains (one per core, each of length 100_000)
        X_bimo = Array{Any}(undef,6)
        @time @threads for i = 1:6
            X_bimo[i] = adaptive_rwm(ξ,post,100_000;thin=100,algorithm=:aswam).X
        end

        # Pool samples
        ξmcmc_bimo = hcat(X_bimo...)


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

    ξ_norm0 = [0.5*(ξ[2]+ξ[3]);ξ[4:5];0.5*(ξ[6] + ξ[7]);ξ[8:end]]
    bounds_norm = [bounds[[2]];bounds[4:5];bounds[[6]];bounds[8:end]]
    lb_norm = [bound[1] for bound in bounds_norm]
    ub_norm = [bound[2] for bound in bounds_norm]
    post_norm = ξ -> all(lb_norm .< ξ .< ub_norm) ? loglike_normal(ξ) : -Inf

    # Get MAP
    _,ξ̂_norm = optimise(loglike_normal,ξ_norm0;obj=:maximise,bounds=bounds_norm)

    # MAP distribution estimate]
    θ̂_norm = Product(Normal.([ξ̂_norm[1:3];0.0],exp.(ξ̂_norm[4:end])))

    # Run 6 chains (one per core, each of length 10_000)
    X_norm = Array{Any}(undef,6)
    @time @threads for i = 1:6
        X_norm[i] = adaptive_rwm(ξ̂_norm,post_norm,100_000;thin=100,algorithm=:aswam).X
    end

    # Pool samples
    ξmcmc_norm = hcat(X_norm...)

    #################################################
    ## Plot

    # (e) Posterior for λ using binomial model
    xv = range(0.7,1.3,200)
    yv = hcat([pdf.(get_λ(x),xv) for x in eachcol(ξmcmc_bimo[:,1:10:end])]...)
    l = [quantile(y,0.025) for y in eachrow(yv)]
    u = [quantile(y,0.975) for y in eachrow(yv)]
    m = pdf.(λ̂_bimo,xv)

    fig5e = plot(xv,m,ribbon=(m-l,u-m),c=col_skew,lw=2.0,fα=0.2,label="MAP (Bimodal)")
    plot!(xv,x->pdf(get_λ(ξ),x),c=:black,lw=2.0,label="True (Bimodal)",ylim=(0.0,7.0))

    # (g) Posterior for λ using normal model
    yv = hcat([pdf.(Normal(x[1],exp(x[4])),xv) for x in eachcol(ξmcmc_norm[:,1:10:end])]...)
    l = [quantile(y,0.025) for y in eachrow(yv)]
    u = [quantile(y,0.975) for y in eachrow(yv)]
    λ̂_norm = Normal(ξ̂_norm[1],exp(ξ̂_norm[4]))
    m = pdf.(λ̂_norm,xv)

    fig5f = plot(xv,m,ribbon=(m-l,u-m),c=col_norm,lw=2.0,fα=0.2,label="MAP (Normal)")
    plot!(xv,x->pdf(get_λ(ξ),x),c=:black,lw=2.0,label="True (Bimodal)")

    ## (g) Predictions
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

    fig5g = plot(tplot,m_pred_bimo,ribbon=(m_pred_bimo-l_pred_bimo,u_pred_bimo-m_pred_bimo),c=col_skew,fα=0.2,lw=2.0,label="MAP Prediction (Bimodal)")
    plot!(tplot,m_pred_norm,c=col_norm,fα=0.05,ls=:dash,label="MAP Prediction (Normal)")
    plot!(tplot,l_pred_norm,c=col_norm,fα=0.1,ls=:dot,label="")
    plot!(tplot,u_pred_norm,c=col_norm,fα=0.1,ls=:dot,label="")
    violin!(fig5g,[T[1]],x[1],lw=0.0,c=:black,α=0.5,label="Data")
    [violin!(fig5g,[T[i]],x[i],lw=0.0,c=:black,α=0.5,label="") for i = 2:length(T)]

    # (h) Predictions at t = 6 d
    xw = range(125,225,200)
    ξw_bimo = ξmcmc_bimo[:,1:10:end]
    yw_bimo = zeros(length(xw),size(ξw_bimo,2))
    ξw_norm = ξmcmc_norm[:,1:10:end]
    yw_norm = zeros(length(xw),size(ξw_norm,2))
    for i = 1:size(ξw_bimo,2)
        θw_bimo = get_θ(ξw_bimo[:,i])
        dw_bimo = MixtureModel([
            approximate_transformed_distribution_skewed(θ -> logistic(T[4],θ),θw_bimo.components[1]),
            approximate_transformed_distribution_skewed(θ -> logistic(T[4],θ),θw_bimo.components[2]),
        ],[logis(ξw_bimo[1,i]),1-logis(ξw_bimo[1,i])])
        yw_bimo[:,i] = pdf.(dw_bimo,xw) 
        θw_norm = Product(Normal.([ξw_norm[1:3,i];0.0],exp.(ξw_norm[4:end,i])))
        dw_norm = approximate_transformed_distribution(θ -> logistic(T[4],θ),θw_norm;order=3)
        yw_norm[:,i] = pdf.(dw_norm,xw) 
    end
    lw_bimo = [quantile(y,0.025) for y in eachrow(yw_bimo)]
    uw_bimo = [quantile(y,0.975) for y in eachrow(yw_bimo)]
    lw_norm = [quantile(y,0.025) for y in eachrow(yw_norm)]
    uw_norm = [quantile(y,0.975) for y in eachrow(yw_norm)]
    dw_bimo = d = MixtureModel([
        approximate_transformed_distribution_skewed(θ -> logistic(T[4],θ),θ̂_bimo.components[1])
        approximate_transformed_distribution_skewed(θ -> logistic(T[4],θ),θ̂_bimo.components[2])
    ],[logis(ξ̂_bimo[1]),1-logis(ξ̂_bimo[1])])
    mw_bimo = pdf.(dw_bimo,xw)
    mw_norm = pdf.(approximate_transformed_distribution(θ -> logistic(T[4],θ),θ̂_norm;order=3),xw)

    fig5h = plot(xw,mw_bimo,ribbon=(mw_bimo-lw_bimo,uw_bimo-mw_bimo),c=col_skew,lw=2.0,fα=0.2,label="MAP (Gamma)")
    plot!(xw,mw_norm,ribbon=(mw_norm-lw_norm,uw_norm-mw_norm),c=col_norm,lw=2.0,fα=0.2,label="MAP (Normal)")
    density!(x[4],c=:black,label="Data",xlim=extrema(xw),widen=true,xlabel="R(4) [µm]",ylabel="Density")
    row2 = plot(fig5d,fig5e,fig5f,layout=grid(1,3))
    plot!(row2,subplot=1,xlabel="λ",ylabel="Density",xlim=(0.7,1.3),ylim=(0.0,5.0),widen=true,legend=:topleft)
    plot!(row2,subplot=2,xlabel="λ",ylabel="Density",xlim=(0.7,1.3),ylim=(0.0,5.0),widen=true,legend=:topleft)
    plot!(row2,subplot=3,xlabel="Time [d]",ylabel="Radius [µm]",widen=true,legend=:bottomright)

#################################################
## Figure 5

    fig5 = plot(fig5a,fig5b,fig5c,fig5d,fig5e,fig5f,fig5g,fig5h,size=(1200,500),layout=grid(2,4))

    plot!(fig5,subplot=1,xlabel="λ",ylabel="Density",xlim=(0.8,1.1),ylim=(0.0,15.0),widen=true,legend=:topleft)
    plot!(fig5,subplot=2,xlabel="λ",ylabel="Density",xlim=(0.8,1.15),ylim=(0.0,15.0),widen=true,legend=:topleft)
    plot!(fig5,subplot=3,xlabel="Time [d]",ylabel="Radius [µm]",widen=true,legend=:bottomright)
    plot!(fig5,subplot=4,xlabel="R(4) [µm]",ylabel="Density",widen=true,legend=:none)
    plot!(fig5,subplot=5,xlabel="λ",ylabel="Density",xlim=(0.7,1.3),ylim=(0.0,6.0),widen=true,legend=:topleft)
    plot!(fig5,subplot=6,xlabel="λ",ylabel="Density",xlim=(0.7,1.3),ylim=(0.0,6.0),widen=true,legend=:topleft)
    plot!(fig5,subplot=7,xlabel="Time [d]",ylabel="Radius [µm]",widen=true,legend=:bottomright)
    plot!(fig5,subplot=8,xlabel="R(4) [µm]",ylabel="Density",widen=true,legend=:none)

    add_plot_labels!(fig5)
    savefig(fig5,"$(@__DIR__)/fig5.svg")