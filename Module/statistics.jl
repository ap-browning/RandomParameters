function optimise(f,x₀;bounds=fill([-Inf,Inf],length(x₀)),obj=:minimise,alg=:LN_NELDERMEAD,autodiff=false,ftol_abs=1e-8,maxtime=30)
    function func(x::Vector,dx::Vector)
        length(dx) > 0 && autodiff == :forward && copyto!(dx,ForwardDiff.gradient(f,x))
        return f(x)
    end
    opt = NLopt.Opt(alg,length(x₀))
    if obj == :minimise
        opt.min_objective = func
    else
        opt.max_objective = func
    end
    opt.maxtime = maxtime
    opt.ftol_abs = ftol_abs
    opt.lower_bounds = [bound[1] for bound in bounds]
    opt.upper_bounds = [bound[2] for bound in bounds]
    (minf,minx,ret) = NLopt.optimize(opt,x₀)
    return minf,minx
end

nanmaximum(x) = maximum(filter(!isnan,x))

function profile(f,x₀,xlims;bounds=fill([-Inf,Inf],length(x₀)),npt=20,normalise=false,obj=:minimise,kwargs...)

    # Dimensions
    n = length(x₀)

    # Find optimum
    optf,optx = optimise(f,x₀;bounds,obj=obj,kwargs...)

    # Grid to profile, insert optimum
    pvec = [collect(range(xlims[i]...,length=npt)) for i = 1:n]
    prof = similar.(pvec)
    argm = [[zeros(size(x₀)) for i = 1:length(pvec[i])] for i = 1:length(pvec)]

    # Loop through dimensions (in parallel...) 
    @threads for i = 1:n
        # Where is the optimum?
        idx = findfirst(pvec[i] .> optx[i])
        guess = copy(optx)[setdiff(1:n,i)]
        # Start above the optima and move up
        for j = idx+1:length(pvec[i])
            fᵢⱼ = x̄ -> f([x̄;pvec[i][j]][invperm([setdiff(1:n,i);i])])
            pᵢⱼ,λᵢⱼ = optimise(fᵢⱼ,guess;bounds=bounds[setdiff(1:n,i)],obj=obj,kwargs...)
            prof[i][j] = pᵢⱼ
            argm[i][j] = λᵢⱼ
            isnan(pᵢⱼ) || (guess = λᵢⱼ)
        end
        guess = copy(optx)[setdiff(1:n,i)]
        for j = idx:-1:1
            fᵢⱼ = x̄ -> f([x̄;pvec[i][j]][invperm([setdiff(1:n,i);i])])
            pᵢⱼ,λᵢⱼ = optimise(fᵢⱼ,guess;bounds=bounds[setdiff(1:n,i)],obj=obj,kwargs...)
            prof[i][j] = pᵢⱼ
            argm[i][j] = λᵢⱼ
            isnan(pᵢⱼ) || (guess = λᵢⱼ) 
        end
        if normalise
            prof[i] .-= nanmaximum(prof[i])
        end
    end

    return pvec,prof,argm

end

function profile(f,x₀,pvec,param;bounds=fill([-Inf,Inf],length(x₀)),npt=20,normalise=false,obj=:minimise,kwargs...)

    # Dimensions
    n = length(x₀)

    # Find optimum
    optf,optx = optimise(f,x₀;bounds,obj=obj,kwargs...)

    # Grid to profile
    prof = similar(pvec)
    argm = [zeros(size(x₀)) for _ = 1:length(pvec)]

    # Where is the optimum?
    idx = findfirst(pvec .> optx[param])
    idx = idx === nothing ? length(pvec) : idx
    guess = copy(optx)[setdiff(1:n,param)]
    # Start above the optima and move up
    for j = idx+1:length(pvec)
        fᵢⱼ = x̄ -> f([x̄;pvec[j]][invperm([setdiff(1:n,param);param])])
        prof[j],guess = optimise(fᵢⱼ,guess;bounds=bounds[setdiff(1:n,param)],obj=obj,kwargs...)
        argm[j] = guess
    end
    guess = copy(optx)[setdiff(1:n,param)]
    for j = idx:-1:1
        fᵢⱼ = x̄ -> f([x̄;pvec[j]][invperm([setdiff(1:n,param);param])])
        prof[j],guess = optimise(fᵢⱼ,guess;bounds=bounds[setdiff(1:n,param)],obj=obj,kwargs...)
        argm[j] = guess
    end
    if normalise
        prof .-= maximum(prof)
    end

    return pvec,prof,argm

end




##############################################################
## USEFUL FUNCTIONS
##############################################################

# Logistic and logit to Transform
logis = x -> exp(x) / (1 + exp(x))
logit = p -> log( p / (1 - p))