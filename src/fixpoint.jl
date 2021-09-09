using LinearAlgebra, SparseArrays

solve_mfg_fixpoint(Problem::MFGTwoDim; method=:PI2, node1=50, node2=50, N=100, maxit=80, verbose=true) = solve_mfg_fixpoint_2d(Problem, Val(method),  node1, node2, N, maxit, verbose)

function solve_mfg_fixpoint_2d(Problem::MFGTwoDim, ::Val{:PI2}, node1::Int64, node2::Int64, N::Int64, maxit::Int64, verbose::Bool)
    xmin1, xmax1, xmin2, xmax2, T, ε, m0, uT, cal_V, F1, F2, update_Q = Problem.xmin1, Problem.xmax1, Problem.xmin2, Problem.xmax2, Problem.T, Problem.ε, Problem.m0, Problem.uT, Problem.V, Problem.F1, Problem.F2, Problem.update_Q
    begin    
        hs1 = (xmax1-xmin1)/node1
        hs2 = (xmax2-xmin2)/node2
        ht = T/N
        hist_q = Float64[]
        hist_m = Float64[]
        hist_u = Float64[]
        residual_FP = Float64[]
        residual_HJB = Float64[]
        resFP = 0.0
        resHJB = 0.0
        converge = false
    end

    # initial 
    begin
        tgrid = Vector(0:ht:T)
        sgrid1 = Vector(xmin1:hs1:xmax1-hs1)
        sgrid2 = Vector(xmin2:hs2:xmax2-hs2)
        M, U, V, M_old, U_old = Initial_2d_state(sgrid1, sgrid2, hs1, hs2, node1, node2, N, m0, uT, cal_V)
        Q = Initial_2d_Q(node1, node2, N)
        Q_new = map(copy, Q)
        # Linear operators with periodic boundary
        A,D = build_Linear_operator_TwoDim(node1,node2,hs1,hs2)
    end

    function solve_FP!(M, Q; N=N, ht=ht, ε=ε, A=A, D=D)
        solve_FP_helper!(M, Q, N, ht, ε, A, D)
    end

    function solve_HJB_fixpoint!(U_new, U_old, M, Q; N=N, ht=ht, ε=ε, V=V, A=A, D=D, F1=F1, F2=F2)
        solve_HJB_fixpoint_helper!(U_new, U_old, M, Q, N, ht, ε, V, A, D, F1, F2)
    end

    function compute_res(U, M, Q; N=N, ht=ht, ε=ε, V=V, A=A, D=D, F1=F1, F2=F2, hs1=hs1, hs2=hs2)
        compute_res_helper(U, M, Q, N, ht, ε, V, A, D, F1, F2, hs1, hs2)
    end

    # println("start Policy Iteration")
    for iter in 1:maxit
        solve_FP!(M, Q)
        update_control!(Q, U_old, M, D, update_Q)
        solve_HJB_fixpoint!(U, U_old, M, Q)
        update_control!(Q_new, U, M, D, update_Q)
    
        resFP, resHJB = compute_res(U, M, Q_new)
        Q, Q_new = Q_new, Q
        
        # record history
        L_dist_M = L_Inf_norm(M-M_old)
        L_dist_U = L_Inf_norm(U-U_old)
        L_dist_Q = map((q,q_new)->L_Inf_norm(q-q_new), Q, Q_new) |> maximum
        append!(hist_m, L_dist_M)
        append!(hist_u, L_dist_U)
        append!(hist_q, L_dist_Q)
        append!(residual_FP, resFP)
        append!(residual_HJB, resHJB)

        verbose && println("iteraton $(iter), ||U_{k+1} - U_{k}|| = $(L_dist_U)")

        M_old = copy(M)
        U_old = copy(U)

        if L_dist_U < 1e-8
            converge = true
            verbose && println("converge!Iteration $iter")

            
            verbose && println("M L2 residual $resFP")
            verbose && println("U L2 residual $resHJB")
            break
        end
        if iter == maxit
            println("error! not converge!")
        end
    end
    history = Solver_history(hist_q,hist_m,hist_u,residual_FP,residual_HJB)
    M = reshape(M,node1,node2,N+1)
    U = reshape(U,node1,node2,N+1)
    result = MFGTwoDim_result(converge,M,U,Q,sgrid1,sgrid2,tgrid,length(hist_q),history)
    return result
end

function solve_HJB_fixpoint_helper!(
    U_new::Matrix{T}, U_old::Matrix{T}, M::Matrix{T}, 
    Q::NamedTuple{<:Any, NTuple{4, Matrix{T}}}, 
    N::Int64, ht::T, ε::T, V::Vector{T}, A::SparseMatrixCSC{T,Int64},
    D::NamedTuple{<:Any, NTuple{4, SparseMatrixCSC{T,Int64}}},
    F1::Function, F2::Function) where {T<:Float64, Dim}
    """

    Q = DU_old / F1(M) 
    jacon * W = -res
    U_new = U_old + W   # loop with t

    """

    # update U with U_old and M
    @inbounds for ti in N:-1:1  
        jacon = 1/ht*I -  ε .* A + 
            (spdiagm(Q.QL1[:,ti])*D.DL1 + spdiagm(Q.QR1[:,ti])*D.DR1 +
            spdiagm(Q.QL2[:,ti])*D.DL2 + spdiagm(Q.QR2[:,ti])*D.DR2)

        res = -(U_old[:,ti+1]-U_old[:,ti]) ./ ht - ε .* A*U_old[:,ti] + 
                0.5 .* F1.(M[:,ti]) .* (Q.QL1[:,ti].^2 + Q.QR1[:,ti].^2 + Q.QL2[:,ti].^2 + Q.QR2[:,ti].^2) -
                V - F2.(M[:,ti+1])
        U_new[:,ti] = jacon \ (-res) + U_old[:,ti]
    end
    return nothing
end