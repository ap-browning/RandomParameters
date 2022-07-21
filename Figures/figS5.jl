#=

    Figure S5

    Failure of methods where outcomes are discontinuous (Allee effect)

=#

using Revise
using RandomParameters
using Plots
using StatsPlots
using LinearAlgebra
using Distributions
using DifferentialEquations

#################################################
## MODEL

λ,R,A = 3.0,300.0,50.0
function logistic_allee(x,p,t)
    λ / 3 * x * (x / A - 1) * (1 - x / R)
end

x₀ = Product([Normal(51.0,1.0)])

#################################################
## FIG S3

# Plot n = 100 trajectories
figS5a = plot()
for i = 1:50
    sol = solve(ODEProblem(logistic_allee,rand(x₀)[1],(0.0,10.0)))
    plot!(sol,c=col_norm,label="")
end
plot!(xlabel="Time [d]",ylabel="Radius [µm]")

# Output is value at t = 5
function f(x₀)
    sol = solve(ODEProblem(logistic_allee,x₀[1],(0.0,5.0)))
    sol(5.0)
end

# Approximate distribution
d = approximate_transformed_distribution(f,x₀;order=3)
x = [f(rand(x₀)) for i = 1:10_000]

figS5b = density(x,lw=2.0,c=:black,label="Simulated")
plot!(d,lw=2.0,c=col_skew,label="Approximate (Gamma)")
plot!(xlim=(-500.0,500.0),legend=:topleft,xlabel="R(6) [µm]",ylabel="Density")

# Fig S3
figS5 = plot(figS5a,figS5b,size=(600,200),widen=true)
add_plot_labels!(figS5)
savefig(figS5,"$(@__DIR__)/figS5.svg")
