using LinearAlgebra, SparseArrays

solve_mfg_fixpoint(Problem::MFGTwoDim; method=:FixPoint2, node1=50, node2=50, N=100, maxit=80, verbose=true) = solve_mfg_fixpoint_2d(Problem, Val(method),  node1, node2, N, maxit, verbose)

function solve_mfg_fixpoint_2d(Problem::MFGTwoDim, ::Val{:FixPoint2}, node1::Int64, node2::Int64, N::Int64, maxit::Int64, verbose::Bool)
    xmin1, xmax1, xmin2, xmax2, T, ε, m0, uT, cal_V, F1, F2, update_Q = Problem.xmin1, Problem.xmax1, Problem.xmin2, Problem.xmax2, Problem.T, Problem.ε, Problem.m0, Problem.uT, Problem.V, Problem.F1, Problem.F2, Problem.update_Q
    println("start solving with fixPoint iteration")
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
        Q_tilde = map(copy, Q)
        # Linear operators with periodic boundary
        A,D = build_Linear_operator_TwoDim(node1,node2,hs1,hs2)
    end

    function solve_FP!(M, Q; N=N, ht=ht, ε=ε, A=A, D=D)
        solve_FP_helper!(M, Q, N, ht, ε, A, D)
    end

    function solve_HJB_fixpoint!(U_new, U_old, M, Q; N=N, ht=ht, ε=ε, V=V, A=A, D=D, F1=F1, F2=F2, update_Q=update_Q)
        solve_HJB_fixpoint_helper!(U_new, U_old, M, Q, N, ht, ε, V, A, D, F1, F2, update_Q)
    end

    function compute_res(U, M, Q; N=N, ht=ht, ε=ε, V=V, A=A, D=D, F1=F1, F2=F2, hs1=hs1, hs2=hs2)
        compute_res_fixpoint(U, M, Q, N, ht, ε, V, A, D, F1, F2, hs1, hs2)
    end

    # println("start Policy Iteration")
    for iter in 1:maxit
        solve_FP!(M, Q)
        update_control!(Q_tilde, U_old, M, D, update_Q)
        solve_HJB_fixpoint!(U, U_old, M, Q_tilde)
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

        verbose && println("iteraton $(iter), ||M_{k+1} - M_{k}|| = $(L_dist_M)")

        M_old = copy(M)
        U_old = copy(U)

        if L_dist_M < 1e-8
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
    F1::Function, F2::Function, update_Q::Function) where {T<:Float64, Dim}
    """

    Q = DU_old / F1(M) 
    jacon * W = -res
    U_new = U_old + W   # loop with t

    """
    
    # update U with U_old and M
    for ti in N:-1:1
        U_temp = copy(U_old[:,ti])
        
        for inner_it in 1:30
            jacon = 1/ht*I -  ε .* A + 2 .*
                (spdiagm(Q.QL1[:,ti])*D.DL1 + spdiagm(Q.QR1[:,ti])*D.DR1 +
                spdiagm(Q.QL2[:,ti])*D.DL2 + spdiagm(Q.QR2[:,ti])*D.DR2)

            res = -(U_new[:,ti+1]-U_temp) ./ ht - ε .* A*U_temp + 
                    0.5 .* F1.(M[:,ti+1]) .* (Q.QL1[:,ti].^2 + Q.QR1[:,ti].^2 + Q.QL2[:,ti].^2 + Q.QR2[:,ti].^2) -
                    V - F2.(M[:,ti+1])
            
            if norm(res) < 1e-8
                println("norm_HJB_res: $(norm(res))") 
                U_new[:,ti] = U_temp
                # println("solve inner HJB") 
                break
            elseif inner_it==30
                println("inner HJB solver not converge")
            else
                U_temp = jacon \ (-res) + U_temp
                update_control!(Q, U_temp, M, D, update_Q, ti)
            end
        end
    end
    return nothing
end

function compute_res_fixpoint(
    U::Matrix{T}, M::Matrix{T}, 
    Q::NamedTuple{<:Any, NTuple{4, Matrix{T}}},
    N::Int64, ht::T, ε::T, V::Vector{T}, A::SparseMatrixCSC{T,Int64},
    D::NamedTuple{<:Any, NTuple{4, SparseMatrixCSC{T,Int64}}},
    F1::Function, F2::Function, hs1::T, hs2::T) where {T<:Float64}

    resFP, resHJB = 0, 0
    for ti in 2:N+1
        lhs =  I/ht - (ε .* A - sum(map((q,d)->spdiagm(q[:,ti-1])*d, values(Q), values(D))))
        rhs = M[:,ti-1] ./ ht
        resFP += sum(abs2.(lhs' *M[:,ti]-rhs))
    end
    resFP = sqrt(hs1*hs2*ht*resFP)

    for ti in N:-1:1  
        temp = -(U[:,ti+1]-U[:,ti]) ./ ht - 
                ε .* A * U[:,ti] + (0.5 .*  (F1.(M[:,ti+1])) .*sum(map(q->q[:,ti].^2, Q))) - 
                V - F2.(M[:,ti+1])
    
        resHJB += sum(abs2.(temp))
    end
    resHJB = sqrt(hs1*hs2*ht*resHJB)
    return (resFP, resHJB)
end

function solve_mfg_fixpoint_2d(Problem::MFGTwoDim, ::Val{:PI2}, node1::Int64, node2::Int64, N::Int64, maxit::Int64, verbose::Bool)
    xmin1, xmax1, xmin2, xmax2, T, ε, m0, uT, cal_V, F1, F2, update_Q = Problem.xmin1, Problem.xmax1, Problem.xmin2, Problem.xmax2, Problem.T, Problem.ε, Problem.m0, Problem.uT, Problem.V, Problem.F1, Problem.F2, Problem.update_Q
    println("start solving with Policy Iteration")
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
        Q_tilde = map(copy, Q)
        # Linear operators with periodic boundary
        A,D = build_Linear_operator_TwoDim(node1,node2,hs1,hs2)
    end

    function solve_FP!(M, Q; N=N, ht=ht, ε=ε, A=A, D=D)
        solve_FP_helper!(M, Q, N, ht, ε, A, D)
    end

    function solve_HJB_PI!(U_new, M, Q; N=N, ht=ht, ε=ε, V=V, A=A, D=D, F1=F1, F2=F2)
        solve_HJB_PI_helper!(U_new, M, Q, N, ht, ε, V, A, D, F1, F2)
    end

    function compute_res(U, M, Q; N=N, ht=ht, ε=ε, V=V, A=A, D=D, F1=F1, F2=F2, hs1=hs1, hs2=hs2)
        compute_res_fixpoint(U, M, Q, N, ht, ε, V, A, D, F1, F2, hs1, hs2)
    end

    # println("start Policy Iteration")
    for iter in 1:maxit
        solve_FP!(M, Q)
        update_control!(Q_tilde, U_old, M, D, update_Q)
        solve_HJB_PI!(U, M, Q_tilde)
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

        verbose && println("iteraton $(iter), ||M_{k+1} - M_{k}|| = $(L_dist_M)")

        M_old = copy(M)
        U_old = copy(U)

        if L_dist_M < 1e-8
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

function solve_HJB_PI_helper!(
    U::Matrix{T}, M::Matrix{T}, 
    Q::NamedTuple{<:Any, NTuple{Dim, Matrix{T}}}, 
    N::Int64, ht::T, ε::T, V::Vector{T}, A::SparseMatrixCSC{T,Int64},
    D::NamedTuple{<:Any, NTuple{Dim, SparseMatrixCSC{T,Int64}}},
    F1::Function, F2::Function) where {T<:Float64, Dim}
    # solve HJB equation with control and M
    for ti in N:-1:1  
        lhs = I - ht .* (ε .* A - sum(map((q,d)->spdiagm(q[:,ti])*d, values(Q), values(D))))
        rhs = U[:,ti+1] + ht .*  (0.5 .*  F1.(M[:,ti+1]) .*sum(map(q->q[:,ti].^2, Q)) + V + F2.(M[:,ti+1]))
        U[:,ti] = lhs \ rhs
    end
    return nothing
end


function Initial_2d_state(
    sgrid1::Vector{Float64}, sgrid2::Vector{Float64},
    hs1::Float64, hs2::Float64,
    node1::Int64, node2::Int64, N::Int64, 
    m0::Function, uT::Function, cal_V::Function)
    M = ones(node1*node2,N+1)
    U = zeros(node1*node2,N+1)
    M0 = m0.(sgrid1,sgrid2')
    C = hs1 * hs2 * sum(M0)
    M0 = M0 ./C
    M[:,1] = reshape(M0, (node1*node2))
    U[:,end] = reshape(uT.(sgrid1,sgrid2'), (node1*node2))
    V = float(cal_V.(sgrid1,sgrid2'))
    V = reshape(V, (node1*node2))
    M_old = copy(M)
    U_old = copy(U)
    return (M, U, V, M_old, U_old)
end

function Initial_2d_Q(node1::Int64, node2::Int64, N::Int64)
    # initial guess control QL=QR=0
    QL1 = zeros(node1*node2,N)
    QR1 = zeros(node1*node2,N)
    QL2 = zeros(node1*node2,N)
    QR2 = zeros(node1*node2,N)
    return (;QL1, QR1, QL2, QR2)
end