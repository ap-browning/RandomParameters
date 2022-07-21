##############################################################
## Derivatives
##############################################################

function tressian(f,x)
    H = x -> ForwardDiff.hessian(f,x)[:]
    reshape(ForwardDiff.jacobian(H,x),fill(length(x),3)...)
end
∂ = ForwardDiff.gradient
∂² = ForwardDiff.hessian
∂³ = tressian


"""
    get_derivatives(f,x)

Obtain value, gradient and Hessian of scalar function f at point x.
"""
function get_derivatives(f,x)
    result = DiffResults.HessianResult(x)
    result = ForwardDiff.hessian!(result,f,x)
    f̂ = DiffResults.value(result)
    ∇ = DiffResults.jacobian(result)
    H = DiffResults.hessian(result)
    return f̂,∇,H
end

"""
    get_derivatives(f,x)

Obtain value, gradients (Jacobian) and Hessian tensor of vector function f at point x.
"""
function get_derivatives(f,x,n)
    d = length(x)
    result = DiffResults.DiffResult(zeros(n,d),zeros(n*d,d))
    result = ForwardDiff.jacobian!(result,x -> ForwardDiff.jacobian(f, x),x)
    ∇ = DiffResults.value(result)   
    H = reshape(DiffResults.jacobian(result),n,d,d)
    f̂ = f(x)
    return f̂,∇,H
end