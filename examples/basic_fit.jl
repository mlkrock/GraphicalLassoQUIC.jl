##
using RCall, LinearAlgebra
n=50

S = rand(n,n)
S = S'*S

λ = ones(n,n)

initial_guess = Matrix{Float64}(I(n))

R"""
X = glasso::glasso($S,$λ,thr=1e-13)$wi
"""

@rget X

jX = QUIC(S,λ,tol=1e-13,msg=0)


