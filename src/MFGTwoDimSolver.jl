using LinearAlgebra, SparseArrays


function solve_mfg(Problem::MFGTwoDim; node1=50, node2=50, N=100, maxit=80, verbose=true)
    xmin1, xmax1, xmin2, xmax2, T, ε, m0, uT, cal_V, F1, F2, update_Q = Problem.xmin1, Problem.xmax1, Problem.xmin2, Problem.xmax2, Problem.T, Problem.ε, Problem.m0, Problem.uT, Problem.V, Problem.F1, Problem.F2, Problem.update_Q
    begin    
        hs1 = (xmax1-xmin1)/node1
        hs2 = (xmax2-xmin2)/node2
        ht = T/N
        hist_q = Float64[]
        hist_m = Float64[]
        hist_u = Float64[]
        resFP = 0.0
        resHJB = 0.0
        converge = false
    end

    # function L_r_norm(u,r::Int=2;hs1=hs1,hs2=hs2,ht=ht)
    #     norm = sum(abs.(u).^r) * (hs1 * hs2 * ht)
    #     norm = norm^(1/r)
    #     return norm
    # end

    function L_r_norm(u)
        norm = maximum(abs.(u))
        return norm
    end

    # initial 
    begin
        tgrid = Vector(0:ht:T)
        sgrid1 = Vector(xmin1:hs1:xmax1-hs1)
        sgrid2 = Vector(xmin2:hs2:xmax2-hs2)
        M = ones(node1*node2,N+1)
        U = zeros(node1*node2,N+1)
        M0 = m0.(sgrid1,sgrid2')
        C = hs1 * hs2 * sum(M0)
        M0 = M0 ./C
        M[:,1] = reshape(M0, (node1*node2))
        U[:,end] = reshape(uT.(sgrid1,sgrid2'), (node1*node2))
        V = cal_V.(sgrid1,sgrid2')
        V = reshape(V, (node1*node2))

        QL1 = zeros(node1*node2,N+1)
        QR1 = zeros(node1*node2,N+1)
        QL2 = zeros(node1*node2,N+1)
        QR2 = zeros(node1*node2,N+1)
        QL1_new = copy(QL1)
        QR1_new = copy(QR1)
        QL2_new = copy(QL2)
        QR2_new = copy(QR2)
        M_old = copy(M)
        U_old = copy(U)
        lhs = sparse(0.0I,node1*node2,node1*node2)
        rhs = zeros(node1*node2)
    end

    begin
        A,DR1,DL1,DR2,DL2 = build_Linear_operator_TwoDim(node1,node2,hs1,hs2)
    end

    # println("start solve with ε=$(ε)")
    @inbounds for iter in 1:maxit
        # Solve FP with Q
        for ti in 2:N+1
            lhs = spdiagm(0=>ones(node1*node2)) - ht .* (ε .* A + 
                     (DR1*spdiagm(0=>QL1[:,ti]) + DL1*spdiagm(0=>QR1[:,ti])) +
                     (DR2*spdiagm(0=>QL2[:,ti]) + DL2*spdiagm(0=>QR2[:,ti]))
                            )
            M[:,ti] = lhs \ M[:,ti-1]
        end

        # Solve HJB with Q
        for ti in N:-1:1
            lhs = spdiagm(0=>ones(node1*node2)) -  ht .* (ε * A - 
                     (spdiagm(0=>QL1[:,ti])*DL1 + spdiagm(0=>QR1[:,ti])*DR1) -
                     (spdiagm(0=>QL2[:,ti])*DL2 + spdiagm(0=>QR2[:,ti])*DR2)
                            )
            rhs .= U[:,ti+1] + ht .*  (0.5 .*  F1.(M[:,ti+1]) .*(QL1[:,ti+1].^2 + QR1[:,ti+1].^2 + QL2[:,ti+1].^2 + QR2[:,ti+1].^2) +
                     V + F2.(M[:,ti+1]))
            U[:,ti] = lhs \ rhs
        end

        # update Q
        for ti in 1:N+1
            U_t = U[:,ti]
            QL1_new[:,ti] = update_Q.(max.(DL1*U_t,0) , M[:,ti])
            QR1_new[:,ti] = update_Q.(min.(DR1*U_t,0) , M[:,ti])
            QL2_new[:,ti] = update_Q.(max.(DL2*U_t,0) , M[:,ti])
            QR2_new[:,ti] = update_Q.(min.(DR2*U_t,0) , M[:,ti])
        end

        QL1_new, QL1 = QL1, QL1_new
        QR1_new, QR1 = QR1, QR1_new
        QL2_new, QL2 = QL2, QL2_new
        QR2_new, QR2 = QR2, QR2_new
        
        # record history
        L_dist_M = L_r_norm(M-M_old)
        L_dist_U = L_r_norm(U-U_old)
        L_dist_Q = L_r_norm([QL1-QL1_new QR1-QR1_new QL2-QL2_new QR2-QR2_new])
        append!(hist_m, L_dist_M)
        append!(hist_u, L_dist_U)
        append!(hist_q, L_dist_Q)

        verbose && println("iteraton $(iter), ||Q_{k+1} - Q_{k}|| = $(L_dist_Q)")

        M_old = copy(M)
        U_old = copy(U)

        if L_dist_Q < 1e-8
            converge = true
            verbose && println("converge!Iteration $iter")
            for ti in 2:N+1
                lhs =  sparse(I,node1*node2,node1*node2) - ht .* (ε .* A + 
                            (DR1*spdiagm(0=>QL1[:,ti]) + DL1*spdiagm(0=>QR1[:,ti])) +
                            (DR2*spdiagm(0=>QL2[:,ti]) + DL2*spdiagm(0=>QR2[:,ti]))
                                )
                rhs = M[:,ti-1]
                resFP += sum(abs2.(lhs*M[:,ti]-rhs))
            end
            resFP = sqrt(hs1*hs2*ht*resFP)
            verbose && println("M L2 residual $resFP")

            for ti in N:-1:1  
                lhs = sparse(I,node1*node2,node1*node2) - ht .* (ε .* A - 
                            (spdiagm(0=>QL1[:,ti])*DL1 + spdiagm(0=>QR1[:,ti])*DR1) -
                            (spdiagm(0=>QL2[:,ti])*DL2 + spdiagm(0=>QR2[:,ti])*DR2)
                                )
                rhs = U[:,ti+1] + ht .*  (0.5 .*  (F1.(M[:,ti+1])) .*(QL1[:,ti+1].^2 + QR1[:,ti+1].^2 + QL2[:,ti+1].^2 + QR2[:,ti+1].^2) + V + F2.(M[:,ti+1]))
            
                resHJB += sum(abs2.(lhs*U[:,ti]-rhs))
            end
            resHJB = sqrt(hs1*hs2*ht*resHJB)
            verbose && println("U L2 residual $resHJB")
            break
        end
        if iter == maxit
            println("error! not converge!")
        end
    end
    history = Solver_history(hist_q,hist_m,hist_u,resFP,resHJB)
    M = reshape(M,node1,node2,N+1)
    U = reshape(U,node1,node2,N+1)
    QL1,QR1,QL2,QR2 = map(Q->reshape(Q,node1,node2,N+1),(QL1,QR1,QL2,QR2))
    result = MFGTwoDim_result(converge,M,U,QL1,QR1,QL2,QR2,sgrid1,sgrid2,tgrid,length(hist_q),history)
    return result
end