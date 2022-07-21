#=
    RandomParameters.jl

    A Julia module containing code to solve random parameter problems.

    Author:     Alexander P. Browning
                ======================
                School of Mathematical Sciences
                Queensland University of Technology
                ======================
                ap.browning@icloud.com
                alexbrowning.me

=#
module RandomParameters

    using Distributions
    using ForwardDiff
    using Combinatorics
    using StatsBase, Statistics
    using Distributions
    using LinearAlgebra
    using NLopt
    using .Threads
    using Random
    using Plots
    using KernelDensity
    using StatsFuns
    using JLD2
    using Interpolations
    using Roots

    export GammaAlt, GammaAltNegative, DiracContinuous, Copula, GaussianCopula, marginalize, MvDependent, MvGamma
    export 𝔼, 𝕍, 𝕊, 𝕂
    export kron_product, ⊗, ⊙
    export  approximate_moments, 
            approximate_mean_variance_skewness, 
            approximate_transformed_distribution,
            approximate_transformed_distribution_skewed
    export density2d
    export optimise,profile,logis,logit

    include("algebra.jl")
    include("calculus.jl")
    include("approximations.jl")
    include("distributions.jl")
    include("moments.jl")
    include("plotting.jl")
    include("statistics.jl")

end