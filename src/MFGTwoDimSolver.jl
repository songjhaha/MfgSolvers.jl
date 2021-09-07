using LinearAlgebra, SparseArrays

solve_mfg(Problem::MFGTwoDim; method=:PI1, node1=50, node2=50, N=100, maxit=80, verbose=true) = solve_mfg_2d(Problem, Val(method),  node1, node2, N, maxit, verbose)

function solve_mfg_2d(Problem::MFGTwoDim, ::Val{:PI1}, node1::Int64, node2::Int64, N::Int64, maxit::Int64, verbose::Bool)
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

    # initial 
    begin
        tgrid = Vector(0:ht:T)
        sgrid1 = Vector(xmin1:hs1:xmax1-hs1)
        sgrid2 = Vector(xmin2:hs2:xmax2-hs2)
        M, U, V, M_old, U_old = Initial_2d_state(sgrid1, sgrid2, hs1, hs2, node1, node2, N, m0, uT, cal_V)
        QL1, QR1, QL2, QR2 = Initial_2d_Q(node1, node2, N)
        QL1_new, QR1_new, QL2_new, QR2_new = map(copy, (QL1, QR1, QL2, QR2))
        # Linear operators with periodic boundary
        A,DR1,DL1,DR2,DL2 = build_Linear_operator_TwoDim(node1,node2,hs1,hs2)
    end

    function solve_FP!(
        M, QL1, QR1, QL2, QR2;
        N=N, ht=ht, ε=ε, A=A,
        DL1=DL1, DR1=DR1,
        DL2=DL2, DR2=DR2)
        solve_FP_2d_helper!(M, QL1, QR1, QL2, QR2, N, ht, ε, A, DL1, DR1, DL2, DR2)
    end

    function solve_HJB!(
        U, M, QL1, QR1, QL2, QR2;
        N=N, ht=ht, ε=ε, V=V, A=A,
        DL1=DL1, DR1=DR1, DL2=DL2, DR2=DR2,
        F1=F1, F2=F2)
        solve_HJB_2d_helper!(U, M, QL1, QR1, QL2, QR2, N, ht, ε, V, A, DL1, DR1, DL2, DR2, F1, F2)
    end

    function compute_res(
        U, M, QL1, QR1, QL2, QR2;
        N=N, ht=ht, ε=ε, V=V, A=A,
        DL1=DL1, DR1=DR1, DL2=DL2, DR2=DR2,
        F1=F1, F2=F2, hs1=hs1, hs2=hs2)
        compute_res_2d_helper(U, M, QL1, QR1, QL2, QR2, N, ht, ε, V, A, DL1, DR1, DL2, DR2, F1, F2, hs1, hs2)
    end

    # println("start solve with ε=$(ε)")
    for iter in 1:maxit
        solve_FP!(M, QL1, QR1, QL2, QR2)
        solve_HJB!(U, M, QL1, QR1, QL2, QR2)
        update_control!(QL1_new, QR1_new, QL2_new, QR2_new,
                     U, M, DL1, DR1, DL2, DR2, update_Q)
        


        QL1_new, QL1 = QL1, QL1_new
        QR1_new, QR1 = QR1, QR1_new
        QL2_new, QL2 = QL2, QL2_new
        QR2_new, QR2 = QR2, QR2_new
        
        # record history
        L_dist_M = L_Inf_norm(M-M_old)
        L_dist_U = L_Inf_norm(U-U_old)
        L_dist_Q = L_Inf_norm([QL1-QL1_new QR1-QR1_new QL2-QL2_new QR2-QR2_new])
        append!(hist_m, L_dist_M)
        append!(hist_u, L_dist_U)
        append!(hist_q, L_dist_Q)

        verbose && println("iteraton $(iter), ||Q_{k+1} - Q_{k}|| = $(L_dist_Q)")

        M_old = copy(M)
        U_old = copy(U)

        if L_dist_Q < 1e-8
            converge = true
            verbose && println("converge!Iteration $iter")

            resFP, resHJB = compute_res(U, M, QL1, QR1, QL2, QR2)
            verbose && println("M L2 residual $resFP")
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

function solve_mfg_2d(Problem::MFGTwoDim, ::Val{:PI2}, node1::Int64, node2::Int64, N::Int64, maxit::Int64, verbose::Bool)
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

    # initial 
    begin
        tgrid = Vector(0:ht:T)
        sgrid1 = Vector(xmin1:hs1:xmax1-hs1)
        sgrid2 = Vector(xmin2:hs2:xmax2-hs2)
        M, U, V, M_old, U_old = Initial_2d_state(sgrid1, sgrid2, hs1, hs2, node1, node2, N, m0, uT, cal_V)
        QL1, QR1, QL2, QR2 = Initial_2d_Q(node1, node2, N)
        QL1_new, QR1_new, QL2_new, QR2_new = map(copy, (QL1, QR1, QL2, QR2))
        QL1_tilde, QR1_tilde, QL2_tilde, QR2_tilde = map(copy, (QL1, QR1, QL2, QR2))
        # Linear operators with periodic boundary
        A,DR1,DL1,DR2,DL2 = build_Linear_operator_TwoDim(node1,node2,hs1,hs2)
    end


    function solve_FP!(
        M, QL1, QR1, QL2, QR2;
        N=N, ht=ht, ε=ε, A=A,
        DL1=DL1, DR1=DR1,
        DL2=DL2, DR2=DR2)
        solve_FP_2d_helper!(M, QL1, QR1, QL2, QR2, N, ht, ε, A, DL1, DR1, DL2, DR2)
    end

    function solve_HJB!(
        U, M, QL1, QR1, QL2, QR2;
        N=N, ht=ht, ε=ε, V=V, A=A,
        DL1=DL1, DR1=DR1, DL2=DL2, DR2=DR2,
        F1=F1, F2=F2)
        solve_HJB_2d_helper!(U, M, QL1, QR1, QL2, QR2, N, ht, ε, V, A, DL1, DR1, DL2, DR2, F1, F2)
    end

    function compute_res(
        U, M, QL1, QR1, QL2, QR2;
        N=N, ht=ht, ε=ε, V=V, A=A,
        DL1=DL1, DR1=DR1, DL2=DL2, DR2=DR2,
        F1=F1, F2=F2, hs1=hs1, hs2=hs2)
        compute_res_2d_helper(U, M, QL1, QR1, QL2, QR2, N, ht, ε, V, A, DL1, DR1, DL2, DR2, F1, F2, hs1, hs2)
    end

    # println("start solve with ε=$(ε)")
    for iter in 1:maxit
        solve_FP!(M, QL1, QR1, QL2, QR2)
        update_control!(QL1_tilde, QR1_tilde, QL2_tilde, QR2_tilde,
                     U, M, DL1, DR1, DL2, DR2, update_Q)
        solve_HJB!(U, M, QL1_tilde, QR1_tilde, QL2_tilde, QR2_tilde)
        update_control!(QL1_new, QR1_new, QL2_new, QR2_new,
                     U, M, DL1, DR1, DL2, DR2, update_Q)

        QL1_new, QL1 = QL1, QL1_new
        QR1_new, QR1 = QR1, QR1_new
        QL2_new, QL2 = QL2, QL2_new
        QR2_new, QR2 = QR2, QR2_new
        
        # record history
        L_dist_M = L_Inf_norm(M-M_old)
        L_dist_U = L_Inf_norm(U-U_old)
        L_dist_Q = L_Inf_norm([QL1-QL1_new QR1-QR1_new QL2-QL2_new QR2-QR2_new])
        append!(hist_m, L_dist_M)
        append!(hist_u, L_dist_U)
        append!(hist_q, L_dist_Q)

        verbose && println("iteraton $(iter), ||Q_{k+1} - Q_{k}|| = $(L_dist_Q)")

        M_old = copy(M)
        U_old = copy(U)

        if L_dist_Q < 1e-8
            converge = true
            verbose && println("converge!Iteration $iter")

            resFP, resHJB = compute_res(U, M, QL1, QR1, QL2, QR2)
            verbose && println("M L2 residual $resFP")
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
    QL1 = zeros(node1*node2,N+1)
    QR1 = zeros(node1*node2,N+1)
    QL2 = zeros(node1*node2,N+1)
    QR2 = zeros(node1*node2,N+1)
    return (QL1, QR1, QL2, QR2)
end

function solve_FP_2d_helper!(
    M::Matrix{Float64}, 
    QL1::Matrix{Float64}, QR1::Matrix{Float64}, 
    QL2::Matrix{Float64}, QR2::Matrix{Float64},
    N::Int64, ht::Float64, ε::Float64, A::SparseMatrixCSC{Float64,Int64},
    DL1::SparseMatrixCSC{Float64,Int64}, DR1::SparseMatrixCSC{Float64,Int64},
    DL2::SparseMatrixCSC{Float64,Int64}, DR2::SparseMatrixCSC{Float64,Int64})
    # solve FP equation with control
    @inbounds for ti in 2:N+1
        lhs = I - ht .* (ε .* A + 
                     (DR1*spdiagm(0=>QL1[:,ti]) + DL1*spdiagm(0=>QR1[:,ti])) +
                     (DR2*spdiagm(0=>QL2[:,ti]) + DL2*spdiagm(0=>QR2[:,ti]))
                            )
        M[:,ti] = lhs \ M[:,ti-1]
    end
    return nothing
end

function solve_HJB_2d_helper!(
    U::Matrix{Float64}, M::Matrix{Float64}, 
    QL1::Matrix{Float64}, QR1::Matrix{Float64}, 
    QL2::Matrix{Float64}, QR2::Matrix{Float64}, 
    N::Int64, ht::Float64, ε::Float64, V::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64},
    DL1::SparseMatrixCSC{Float64,Int64}, DR1::SparseMatrixCSC{Float64,Int64},
    DL2::SparseMatrixCSC{Float64,Int64}, DR2::SparseMatrixCSC{Float64,Int64},
    F1::Function, F2::Function)
    # solve HJB equation with control and M
    @inbounds for ti in N:-1:1  
        lhs = I -  ht .* (ε * A - 
                     (spdiagm(0=>QL1[:,ti])*DL1 + spdiagm(0=>QR1[:,ti])*DR1) -
                     (spdiagm(0=>QL2[:,ti])*DL2 + spdiagm(0=>QR2[:,ti])*DR2))
        rhs = U[:,ti+1] + ht .*  (0.5 .*  F1.(M[:,ti+1]) .*(QL1[:,ti+1].^2 + QR1[:,ti+1].^2 + QL2[:,ti+1].^2 + QR2[:,ti+1].^2) + V + F2.(M[:,ti+1]))
        U[:,ti] = lhs \ rhs
    end
    return nothing
end

function update_control!(
    QL1_new::Matrix{Float64}, QR1_new::Matrix{Float64},
    QL2_new::Matrix{Float64}, QR2_new::Matrix{Float64},
    U::Matrix{Float64}, M::Matrix{Float64},
    DL1::SparseMatrixCSC{Float64,Int64}, 
    DR1::SparseMatrixCSC{Float64,Int64},
    DL2::SparseMatrixCSC{Float64,Int64}, 
    DR2::SparseMatrixCSC{Float64,Int64},
    update_Q::Function)

    # update control Q from U and M
    QL1_new .= update_Q.(max.(DL1*U,0) , M)
    QR1_new .= update_Q.(min.(DR1*U,0) , M)
    QL2_new .= update_Q.(max.(DL2*U,0) , M)
    QR2_new .= update_Q.(min.(DR2*U,0) , M)
    return nothing
end

function compute_res_2d_helper(
    U::Matrix{Float64}, M::Matrix{Float64}, 
    QL1::Matrix{Float64}, QR1::Matrix{Float64}, 
    QL2::Matrix{Float64}, QR2::Matrix{Float64}, 
    N::Int64, ht::Float64, ε::Float64, V::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64},
    DL1::SparseMatrixCSC{Float64,Int64}, DR1::SparseMatrixCSC{Float64,Int64},
    DL2::SparseMatrixCSC{Float64,Int64}, DR2::SparseMatrixCSC{Float64,Int64},
    F1::Function, F2::Function, hs1::Float64, hs2::Float64)

    resFP, resHJB = 0, 0
    for ti in 2:N+1
        lhs =  I - ht .* (ε .* A + 
                    (DR1*spdiagm(0=>QL1[:,ti]) + DL1*spdiagm(0=>QR1[:,ti])) +
                    (DR2*spdiagm(0=>QL2[:,ti]) + DL2*spdiagm(0=>QR2[:,ti]))
                        )
        rhs = M[:,ti-1]
        resFP += sum(abs2.(lhs*M[:,ti]-rhs))
    end
    resFP = sqrt(hs1*hs2*ht*resFP)

    for ti in N:-1:1  
        lhs = I - ht .* (ε .* A - 
                    (spdiagm(0=>QL1[:,ti])*DL1 + spdiagm(0=>QR1[:,ti])*DR1) -
                    (spdiagm(0=>QL2[:,ti])*DL2 + spdiagm(0=>QR2[:,ti])*DR2)
                        )
        rhs = U[:,ti+1] + ht .*  (0.5 .*  (F1.(M[:,ti+1])) .*(QL1[:,ti+1].^2 + QR1[:,ti+1].^2 + QL2[:,ti+1].^2 + QR2[:,ti+1].^2) + V + F2.(M[:,ti+1]))
    
        resHJB += sum(abs2.(lhs*U[:,ti]-rhs))
    end
    resHJB = sqrt(hs1*hs2*ht*resHJB)
    return (resFP, resHJB)
end