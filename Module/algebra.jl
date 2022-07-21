

function kron_product(a,b)
    # Check sizes
    (unique(size(a)) == unique(size(b))) && 
    (length(unique(size(a))) == length(unique(size(b)))) ||
    error("Check sizes!")
    # Size
    n = size(a)[1]
    # Dimensionality of output
    d = length(size(a)) + length(size(b))
    # Output
    vec = kron(b[:],a[:])
    # Reshape
    reshape(vec,fill(n,d)...)
end
⊗(a,b) = kron_product(a,b)
⊙ = ⋅

# Speed-ups
LinearAlgebra.dot(a::Diagonal,b) = (n = size(b,1); sum(a[i,i] * b[i,i] for i = 1:n))
LinearAlgebra.dot(a,b::Diagonal) = dot(b,a)

dot2(a::Diagonal,b) = (n = size(b,1); sum(a[i,i] * b[i,i] for i = 1:n))
dot2(a,b::Diagonal) = dot2(b,a)