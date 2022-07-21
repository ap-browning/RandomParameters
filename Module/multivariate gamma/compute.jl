
push!(LOAD_PATH,"/Users/Alex/Code/SpheroidsRODE/Module")

using RandomParameters
using Distributions
using LinearAlgebra
using .Threads
using Interpolations
using Roots
using JLD2

## Get correlations from inputs (skewnesses)
function get_output_correlation(ω₁,ω₂,ρ̃;n=100,ε=1e-5)
    # Marginals
    d₁ = GammaAlt(0.0,1.0,ω₁)
    d₂ = GammaAlt(0.0,1.0,ω₂)
    # Joint (with copula)
    dmv = MvDependent(GaussianCopula(ρ̃),[d₁,d₂])
    # Integrate
    x = range(isa(d₁,Normal) ? quantile(d₁,ε) : minimum(d₁),quantile.(d₁,1-ε),n)
    y = range(isa(d₂,Normal) ? quantile(d₂,ε) : minimum(d₂),quantile.(d₂,1-ε),n)
    # Quadrature to get E(XY)
    Exy = sum(filter(!isnan,
        [xx * yy * pdf(dmv,[xx,yy]) for xx in x, yy in y])) *
        diff(x)[1] * diff(y)[1]
    # Correlation
    (Exy - mean(d₁) * mean(d₂)) / std(d₁) / std(d₂)
end

## Compute over a grid
W₁ = range(0.0,2.0,50)
W₂ = range(0.0,2.0,50)
P̃ = range(0.0,0.99,50)
P = zeros(length(W₁),length(W₂),length(P̃))
@time @threads for i = 1:length(W₁)
    for j = 1:length(W₂), k = 1:length(P̃)
        P[i,j,k] = get_output_correlation(W₁[i],W₂[j],P̃[k])
    end
    display("Done $i")
end

# Save
jldsave("Module/correlation_data.jld2",W₁=W₁,W₂=W₂,P̃=P̃,P=P)
