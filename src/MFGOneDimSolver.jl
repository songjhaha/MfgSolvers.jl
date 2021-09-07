using LinearAlgebra, SparseArrays

solve_mfg(Problem::MFGOneDim; method=:PI1, node=50, N=100, maxit=80, verbose=true) = solve_mfg_1d(Problem, Val(method), node, N, maxit, verbose)

function solve_mfg_1d(Problem::MFGOneDim, ::Val{:PI1}, node::Int64, N::Int64, maxit::Int64, verbose::Bool)
    xmin, xmax, T, ε, m0, uT, cal_V, F1, F2, update_Q = Problem.xmin, Problem.xmax, Problem.T, Problem.ε, Problem.m0, Problem.uT, Problem.V, Problem.F1, Problem.F2, Problem.update_Q
    begin
        hs = (xmax-xmin)/node
        ht = T/N
        hist_q = Float64[]
        hist_m = Float64[]
        hist_u = Float64[]
        resFP = 0.0
        resHJB = 0.0
        converge = false
    end

    # function L_r_norm(u,r::Int=2;hs=hs,ht=ht)
    #     norm = sum(abs.(u).^r) * (hs * ht)
    #     norm = norm^(1/r)
    #     return norm
    # end

    function L_r_norm(u)
        norm = maximum(abs.(u))
        return norm
    end

    # initial
    begin
        sgrid = Vector(0:hs:1-hs)
        tgrid = Vector(0:ht:T)
        M0 = m0.(sgrid)
        C = hs * sum(M0)
        M0 = M0 ./ C
        M = ones(node,N+1)
        M[:,1] = M0  # initial distribution
        U = zeros(node,N+1)
        U[:,end] = uT.(sgrid)
        V = cal_V.(sgrid)
        QL = zeros(node,N+1)  # initial guess control QL=QR=0
        QR = zeros(node,N+1)
        QL_new = copy(QL)
        QR_new = copy(QR)
        M_old = copy(M)
        U_old = copy(U)
    end

    # Linear operators
    begin 
        ## periodic boundary
        A, DR, DL = build_Linear_operator(node,hs)
    end

    # Start Policy Iteration
    @inbounds for iter in 1:maxit
        # solve FP equation with control
        for ti in 2:N+1
            lhs =  spdiagm(0=>ones(node)) - ht .* (ε .* A + 
                    (DR*spdiagm(0=>QL[:,ti]) + DL*spdiagm(0=>QR[:,ti]))
                    )
            M[:,ti] = lhs \ M[:,ti-1]
        end

        # solve HJB equation with control and M
        for ti in N:-1:1  
            lhs = spdiagm(0=>ones(node)) - ht .* (ε .* A -
                    (spdiagm(0=>QL[:,ti])*DL + spdiagm(0=>QR[:,ti])*DR)
                    )
            rhs = U[:,ti+1] + ht .* (0.5 .* F1.(M[:,ti+1]) .*(QL[:,ti+1].^2 + QR[:,ti+1].^2) + V +  F2.(M[:,ti+1]))
            U[:,ti] = lhs \ rhs
        end

        # update control Q from U and M
        for ti in 1:N+1
            QL_new[:,ti] = update_Q.(max.(DL*U[:,ti],0), M[:,ti])
            QR_new[:,ti] = update_Q.(min.(DR*U[:,ti],0), M[:,ti])
        end
          
        QL_new, QL = QL, QL_new
        QR_new, QR = QR, QR_new

        # record history
        L_dist_M = L_r_norm(M-M_old)
        L_dist_U = L_r_norm(U-U_old)
        L_dist_Q = L_r_norm([QL-QL_new; QR-QR_new])
        append!(hist_m, L_dist_M)
        append!(hist_u, L_dist_U)
        append!(hist_q, L_dist_Q)


        verbose && println("iteraton $(iter), ||Q_{k+1} - Q_{k}|| = $(L_dist_Q)")

        M_old = copy(M)
        U_old = copy(U)

        # If converge, compute residual
        if L_dist_Q < 1e-8
            converge = true
            verbose && println("converge!Iteration $iter")

            for ti in 2:N+1
                lhs =  spdiagm(0=>ones(node)) - ht .* (ε .* A + 
                        (DR*spdiagm(0=>QL[:,ti])+DL*spdiagm(0=>QR[:,ti]))
                        )
                rhs = M[:,ti-1]
                resFP += sum(abs2.(lhs*M[:,ti]-rhs))
            end
            resFP = sqrt(hs*ht*resFP)
            verbose && println("M L2 residual $resFP")

            for ti in N:-1:1  
                lhs = spdiagm(0=>ones(node)) - ht .* (ε .* A -
                        (spdiagm(0=>QL[:,ti])*DL+spdiagm(0=>QR[:,ti])*DR)
                        )
                rhs = U[:,ti+1] + ht .* (0.5 .* F1.(M[:,ti+1]) .*(QL[:,ti+1].^2 + QR[:,ti+1].^2) + V +  F2.(M[:,ti+1]))
                resHJB += sum(abs2.(lhs*U[:,ti]-rhs))
            end
            resHJB = sqrt(hs*ht*resHJB)
            verbose && println("U L2 residual $resHJB")
            break            
        end
        if iter == maxit
            println("error! not converge!")
        end
    end
    history = Solver_history(hist_q,hist_m,hist_u,resFP,resHJB)
    result = MFGOneDim_result(converge,M,U,QL,QR,sgrid,tgrid,length(hist_q),history)

    return result
end

function solve_mfg_1d(Problem::MFGOneDim, ::Val{:PI2}, node::Int64, N::Int64, maxit::Int64, verbose::Bool)
    xmin, xmax, T, ε, m0, uT, cal_V, F1, F2, update_Q = Problem.xmin, Problem.xmax, Problem.T, Problem.ε, Problem.m0, Problem.uT, Problem.V, Problem.F1, Problem.F2, Problem.update_Q
    begin
        hs = (xmax-xmin)/node
        ht = T/N
        hist_q = Float64[]
        hist_m = Float64[]
        hist_u = Float64[]
        resFP = 0.0
        resHJB = 0.0
        converge = false
    end

    # function L_r_norm(u,r::Int=2;hs=hs,ht=ht)
    #     norm = sum(abs.(u).^r) * (hs * ht)
    #     norm = norm^(1/r)
    #     return norm
    # end

    function L_r_norm(u)
        norm = maximum(abs.(u))
        return norm
    end

    # initial
    begin
        sgrid = Vector(0:hs:1-hs)
        tgrid = Vector(0:ht:T)
        M0 = m0.(sgrid)
        C = hs * sum(M0)
        M0 = M0 ./ C
        M = ones(node,N+1)
        M[:,1] = M0  # initial distribution
        U = zeros(node,N+1)
        U[:,end] = uT.(sgrid)
        V = cal_V.(sgrid)
        QL = zeros(node,N+1)  # initial guess control QL=QR=0
        QR = zeros(node,N+1)
        QL_new = copy(QL)
        QR_new = copy(QR)
        M_old = copy(M)
        U_old = copy(U)
    end

    # Linear operators
    begin 
        ## periodic boundary
        A, DR, DL = build_Linear_operator(node,hs)
    end

    # Start Policy Iteration
    @inbounds for iter in 1:maxit
        # solve FP equation with control
        for ti in 2:N+1
            lhs =  spdiagm(0=>ones(node)) - ht .* (ε .* A + 
                    (DR*spdiagm(0=>QL[:,ti]) + DL*spdiagm(0=>QR[:,ti]))
                    )
            M[:,ti] = lhs \ M[:,ti-1]
        end

        # solve HJB equation with control and M
        for ti in N:-1:1  
            lhs = spdiagm(0=>ones(node)) - ht .* (ε .* A -
                    (spdiagm(0=>QL[:,ti])*DL + spdiagm(0=>QR[:,ti])*DR)
                    )
            rhs = U[:,ti+1] + ht .* (0.5 .* F1.(M[:,ti+1]) .*(QL[:,ti+1].^2 + QR[:,ti+1].^2) + V +  F2.(M[:,ti+1]))
            U[:,ti] = lhs \ rhs
        end

        # update control Q from U and M
        for ti in 1:N+1
            QL_new[:,ti] = update_Q.(max.(DL*U[:,ti],0), M[:,ti])
            QR_new[:,ti] = update_Q.(min.(DR*U[:,ti],0), M[:,ti])
        end
          
        QL_new, QL = QL, QL_new
        QR_new, QR = QR, QR_new

        # record history
        L_dist_M = L_r_norm(M-M_old)
        L_dist_U = L_r_norm(U-U_old)
        L_dist_Q = L_r_norm([QL-QL_new; QR-QR_new])
        append!(hist_m, L_dist_M)
        append!(hist_u, L_dist_U)
        append!(hist_q, L_dist_Q)


        verbose && println("iteraton $(iter), ||Q_{k+1} - Q_{k}|| = $(L_dist_Q)")

        M_old = copy(M)
        U_old = copy(U)

        # If converge, compute residual
        if L_dist_Q < 1e-8
            converge = true
            verbose && println("converge!Iteration $iter")

            for ti in 2:N+1
                lhs =  spdiagm(0=>ones(node)) - ht .* (ε .* A + 
                        (DR*spdiagm(0=>QL[:,ti])+DL*spdiagm(0=>QR[:,ti]))
                        )
                rhs = M[:,ti-1]
                resFP += sum(abs2.(lhs*M[:,ti]-rhs))
            end
            resFP = sqrt(hs*ht*resFP)
            verbose && println("M L2 residual $resFP")

            for ti in N:-1:1  
                lhs = spdiagm(0=>ones(node)) - ht .* (ε .* A -
                        (spdiagm(0=>QL[:,ti])*DL+spdiagm(0=>QR[:,ti])*DR)
                        )
                rhs = U[:,ti+1] + ht .* (0.5 .* F1.(M[:,ti+1]) .*(QL[:,ti+1].^2 + QR[:,ti+1].^2) + V +  F2.(M[:,ti+1]))
                resHJB += sum(abs2.(lhs*U[:,ti]-rhs))
            end
            resHJB = sqrt(hs*ht*resHJB)
            verbose && println("U L2 residual $resHJB")
            break            
        end
        if iter == maxit
            println("error! not converge!")
        end
    end
    history = Solver_history(hist_q,hist_m,hist_u,resFP,resHJB)
    result = MFGOneDim_result(converge,M,U,QL,QR,sgrid,tgrid,length(hist_q),history)

    return result
end