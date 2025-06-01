
"""
QUIC(S,λ)
Solving the graphical lasso problem. Minimize the penalized negative log-likelihood
```math
-logdet(X) + tr(S*X) + sum(λ .* abs.(X))
```
over positive semidefinite matrices X. Note that λ is a symmetric matrix with nonnegative diagonal elements
and strictly positive off-diagonal elements. Please specify λ as a matrix (not scalar).

This package solves the graphical lasso problem using the QUadratic approximation of Inverse Covariance (QUIC) algorithm (Hsieh et. at. 2014).
See Algorithm 1 on page 2917 for an overview of QUIC, and Algorithm 2 on page 2928 for a more detailed description.
The original QUIC algorithm written in C. This is a pure Julia implementation of their algorithm. 

Basic usage of the function is `QUIC(S,λ)` where S is a positive semidefinite matrix (usually a sample covariance matrix) and λ is a penalty matrix. 
Some options for the user include `tol` (default: 1e-4), `msg` to control output  (default 0, possible values are 0 through 4 with 4 being the most verbose),
maximum number of iterations `maxIter` (default: 1000), and `Xinit` and `Winit` (`Winit` is the inverse of `Xinit`, please provide both if you want to specify 
an initial guess). If `Xinit` and `Winit` are not specified, then the initial guess is the identity matrix, which gives a fast update for the first iteration. 
but the code/algorithm is much more complicated. If S is very large, you will need to look into this new linear-time method.

External links
* [Hsieh, C.-J., Sustik, M. A., Dhillon, I. S. and Ravikumar, P. (2014) QUIC: quadratic approximation for sparse inverse covariance estimation. J. Mach. Learn. Res., 15, 2911– 2947.](https://jmlr.org/papers/volume15/hsieh14a/hsieh14a.pdf)
"""
function QUIC(S,λ; tol =1e-4, msg=0, Xinit=nothing, Winit=nothing, maxIter = 1000, rng=Random.GLOBAL_RNG)

    #Minimize -logdet(X) + tr(S*X) + sum(λ .* abs.(X)) over positive definite X.
    #Note that W = X⁻¹ is approximately the sample covariance S.
    #D denotes the Newton direction solved by  coordiniate descent.
    #U = D*W stored for more efficient inner products of the form w_i' D w_j.

    QUIC_MSG_NO       = 0
    QUIC_MSG_MIN      = 1
    QUIC_MSG_NEWTON   = 2
    QUIC_MSG_CD       = 3
    QUIC_MSG_LINE     = 4

    p = size(S,1)

    if(isnothing(Xinit) && isnothing(Winit))
        X = Matrix{Float64}(I(p))
        W = Matrix{Float64}(I(p))
    else
        X = Matrix(Symmetric(Xinit))
        W = Matrix(Symmetric(Winit))
    end

    maxlineiter = 20
    cdSweepTol = 0.05    
    σ = 0.001
    β = 0.5
    ϵ = eps(Float64)
    fX1 = Inf
    fXprev = Inf
    fX = Inf
    logdetX = 0.0
    trSX = 0.0
    l1normX = 0.0
    D = zeros(p,p)
    U = zeros(p,p)

    for NewtonIter in 1:maxIter
        
        normD = 0.0
        diffD = 0.0
        μ_counter = 1
        μ_flag = 1
        activeSet_i = Int64[]
        activeSet_j = Int64[] 
        subgrad = Inf

        #Two main steps to the algorithm: find Newton direction, then line search for step in the Newton direction.
        #Main step 1: Finding the best Newton direction D via coordinate descent.
        if (isdiag(X) && isone(NewtonIter))
            #Fast solution to coordinate descent problem if the initial guess is diagonal.

            #Update off diagonal elements of D.
            @inbounds for j in 1:p
                @inbounds for i in 1:(j-1)
                    ainv = 1.0/(W[i,i] * W[j,j])
                    b = S[i,j]
                    μ = softthresh(-b*ainv,λ[i,j]*ainv)
                    D[i,j] = μ
                    D[j,i] = μ 
                end
            end

            #Update diagonal elements of D.
            @inbounds for i in 1:p
                ainv = 1.0/(W[i,i]^2)
                b = S[i,i] - W[i,i]
                c = X[i,i]
                μ = -c + softthresh(c-b*ainv,λ[i,i]*ainv)
                D[i,i] = μ
            end
            
            #Compute objective at initial guess.
            logdetX = sum(log.(diag(X)))
            l1normX = sum(diag(λ) .* abs.(diag(X)))
            trSX = dot(diag(X),diag(S))
            fX = -logdetX + trSX + l1normX

            if (msg ≥ QUIC_MSG_NEWTON) 
                println("Newton iteration 1.")
                println("  X is a diagonal matrix.")
                println("  Initial f value $fX.")
            end

        else
            #Standard update for Newton direction D.

            #Compute objective for an initial guess that is non-diagonal.
            if(isone(NewtonIter))
                logdetX = 2.0*sum(log.(diag(cholesky(X).U)))
                l1normX = sum(λ .* abs.(X))
                trSX = sum(S .* X)
                fX = -logdetX + trSX + l1normX                    
            end

            #Compute the active set and the minimum norm subgradient.
            subgrad = 0.0
            numActive = 0
            @inbounds for j in 1:p
                @inbounds for i in 1:j
                    g = S[i,j]-W[i,j]
                    if (X[i,j] != 0.0 || (abs(g) > λ[i,j]))
                        push!(activeSet_i, i)
                        push!(activeSet_j, j)
                        numActive += 1
                        if X[i,j] > 0
                            g += λ[i,j]
                        elseif X[i,j] < 0
                            g -= λ[i,j]
                        else
                            g = abs(g) - λ[i,j]
                        end
                        subgrad += abs(g)
                    end
                end
            end

            if(msg ≥ QUIC_MSG_NEWTON) 
                println("Newton iteration $NewtonIter.")
                if(isone(NewtonIter))
                    println("  Initial f value $fX.")
                end
                println("  Active set size = $numActive.")
                println("  sub-gradient = $subgrad, l1-norm of X = $l1normX.")
            end

            #Coordinate Descent to find best Newton direction D.
            #Only needs to be performed on activeSet.
            D .= 0.0
            U .= 0.0

            indexshuffle = collect(1:numActive)

            for cdSweep in 1:Int64(floor(1.0 + NewtonIter/3.0))

                diffD = 0.0
                shuffle!(rng,indexshuffle)
                activeSet_i = activeSet_i[indexshuffle]
                activeSet_j = activeSet_j[indexshuffle]

                @inbounds for l in 1:numActive
                    i = activeSet_i[l]
                    j = activeSet_j[l]

                    if i==j
                        #Update diagonal elements of D.
                        ainv = 1.0/(W[i,i]^2)
                        b = S[i,i] - W[i,i] +  dot(W[:,i],U[:,i])
                        c = X[i,i] + D[i,i]
                        normD -= abs(D[i,i])
                        μ = -c + softthresh(c - b*ainv,λ[i,i]*ainv)
                        D[i,i] += μ
                        U[i,:] .+= μ*W[i,:]	
                        diffD += abs(μ)
                        normD += abs(D[i,i])
                    else
                        #Update off diagonal elements of D.
                        ainv = 1.0/(W[i,j]^2 + W[i,i] * W[j,j])       
                        b = S[i,j] - W[i,j] + dot(W[:,i],U[:,j])
                        c = X[i,j] + D[i,j]
                        normD -= 2.0*abs(D[i,j])
                        μ = -c + softthresh(c - b*ainv,λ[i,j]*ainv) 
                        D[i,j] += μ
                        D[j,i] += μ
                        U[i,:] .+= μ * W[j,:]
                        U[j,:] .+= μ * W[i,:]
                        diffD += 2.0*abs(μ)
                        normD += 2.0*abs(D[i,j]) 
                    end
                end

                if (msg ≥ QUIC_MSG_CD) 
                    println("  Coordinate descent sweep $cdSweep. norm of D = $normD, change in D = $diffD.")
                end

                if diffD ≤ normD*cdSweepTol
                    break
                end
            end
        end

        #Newton direction D has been calculated.
        #Main step 2: Armijo step linesearch in the direction D.
        #Check if guess is positive definite and for sufficient decrease in objective.

        l1normXD = sum(λ .* abs.(X+D))
        sumSD = sum(S .* D)
        tr∇g = sumSD - sum(W.*D)
        fX1prev = Inf
        α = 1.0
        δ = 0.0 

        for lineiter in 0:maxlineiter

            #Use W to store X + α*D and its cholesky factor (if it exists).
            W = X + α*D
            l1normX1 = sum(λ .* abs.(W))
            cholerror = false
            try
                W = cholesky(W)
            catch err
                if isa(err,PosDefException)
                    cholerror = true
                end
            end

            if(cholerror)
                if (msg ≥ QUIC_MSG_LINE)
                    println("    Line search step size $α.  Lack of positive definiteness.")
                end
                α *= β
                continue
            end
            
            logdetX1 = -2.0*sum(log.(diag(W.U)))
            trSX1 = trSX + α * sumSD
            fX1 = logdetX1 + trSX1 + l1normX1
            δ = tr∇g + l1normXD - l1normX

            #Sufficient decrease of objective?
            if((fX1 ≤ fX + α*σ*δ) || iszero(normD))
                if (msg ≥ QUIC_MSG_LINE)
                    println("    Line search step size chosen: $α.")
                end
                fXprev = fX
                fX = fX1
                l1normX = l1normX1
                logdetX = logdetX1
                trSX = trSX1
                break
            end

            if (msg ≥ QUIC_MSG_LINE) 
                println("    Line search step size $α.")
                println("      Objective value would not decrease sufficiently: $(fX1 - fX).")
            end

            if (fX1prev < fX1) 
                fXprev = fX
                l1normX = l1normX1
                logdetX = logdetX1
                trSX = trSX1
                break
            end
            
            fX1prev = fX1
            α *= β
        end

        #Armijo linesearch complete.
        #Update guess for X and check for convergence.

        if (msg ≥ QUIC_MSG_NEWTON) 
            println("  New f value $fX, objective value decreased by $(fXprev - fX).")
        end

        #Update guess for X. 
        #Note that W is storing the cholesky of X + αD.
        #So calling inv on W gives back (X + αD)⁻¹
        X .+= α * D
        W = inv(W)

        #Converged?
        if ((subgrad*α ≥ l1normX*tol) && (abs((fX - fXprev)/fX) ≥ ϵ))
            continue
        end

        break
    end

    return X
end

softthresh(z,r) = sign(z)*max(abs(z)-r,0.0)