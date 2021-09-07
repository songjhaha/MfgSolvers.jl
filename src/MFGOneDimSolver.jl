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

    # Initial
    begin
        sgrid = Vector(xmin:hs:xmax-hs)
        tgrid = Vector(0:ht:T)
        M, U, V, M_old, U_old = Initial_1d_state(sgrid, hs, node, N, m0, uT, cal_V)
        QL, QR, QL_new, QR_new = Initial_1d_Q(node, N)
        # Linear operators with periodic boundary
        A, DR, DL = build_Linear_operator(node,hs)
    end

    function solve_FP!(M, QL, QR; N=N, ht=ht, ε=ε, A=A, DL=DL, DR=DR)
        solve_FP_1d_helper!(M, QL, QR, N, ht, ε, A, DL, DR)
    end

    function solve_HJB!(U, M, QL, QR; N=N, ht=ht, ε=ε, V=V, A=A, DL=DL, DR=DR, F1=F1, F2=F2)
        solve_HJB_1d_helper!(U, M, QL, QR, N, ht, ε, V, A, DL, DR, F1, F2)
    end

    function compute_res(U, M, QL, QR; N=N, ht=ht, ε=ε, V=V, A=A, DL=DL, DR=DR, F1=F1, F2=F2, hs=hs)
        compute_res_1d_helper(U, M, QL, QR, N, ht, ε, V, A, DL, DR, F1, F2, hs)
    end

    # Start Policy Iteration
    for iter in 1:maxit
        solve_FP!(M, QL, QR)

        solve_HJB!(U, M, QL, QR)     

        update_control!(QL_new, QR_new, U, M, DL, DR, update_Q)
          
        QL_new, QL = QL, QL_new
        QR_new, QR = QR, QR_new

        # record history
        L_dist_M = L_Inf_norm(M-M_old)
        L_dist_U = L_Inf_norm(U-U_old)
        L_dist_Q = L_Inf_norm([QL-QL_new; QR-QR_new])
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

            resFP, resHJB = compute_res(U, M, QL, QR)
            verbose && println("M L2 residual $resFP")
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

    # Initial
    begin
        sgrid = Vector(xmin:hs:xmax-hs)
        tgrid = Vector(0:ht:T)
        M, U, V, M_old, U_old = Initial_1d_state(sgrid, hs, node, N, m0, uT, cal_V)
        QL, QR, QL_new, QR_new = Initial_1d_Q(node, N)
        QL_tilde, QR_tilde = map(copy, (QL, QR))
        # Linear operators with periodic boundary
        A, DR, DL = build_Linear_operator(node,hs)
    end


    function solve_FP!(M, QL, QR; N=N, ht=ht, ε=ε, A=A, DL=DL, DR=DR)
        solve_FP_1d_helper!(M, QL, QR, N, ht, ε, A, DL, DR)
    end

    function solve_HJB!(U, M, QL, QR; N=N, ht=ht, ε=ε, V=V, A=A, DL=DL, DR=DR, F1=F1, F2=F2)
        solve_HJB_1d_helper!(U, M, QL, QR, N, ht, ε, V, A, DL, DR, F1, F2)
    end

    function compute_res(U, M, QL, QR; N=N, ht=ht, ε=ε, V=V, A=A, DL=DL, DR=DR, F1=F1, F2=F2, hs=hs)
        compute_res_1d_helper(U, M, QL, QR, N, ht, ε, V, A, DL, DR, F1, F2, hs)
    end

    # Start Policy Iteration
    for iter in 1:maxit
        solve_FP!(M, QL, QR)

        update_control!(QL_tilde, QR_tilde, U, M, DL, DR, update_Q)

        solve_HJB!(U, M, QL_tilde, QR_tilde)     

        update_control!(QL_new, QR_new, U, M, DL, DR, update_Q)
          
        QL_new, QL = QL, QL_new
        QR_new, QR = QR, QR_new

        # record history
        L_dist_M = L_Inf_norm(M-M_old)
        L_dist_U = L_Inf_norm(U-U_old)
        L_dist_Q = L_Inf_norm([QL-QL_new; QR-QR_new])
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

            resFP, resHJB = compute_res(U, M, QL, QR)
            verbose && println("M L2 residual $resFP")
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


function Initial_1d_state(
    sgrid::Vector{Float64}, hs::Float64, node::Int64, N::Int64, 
    m0::Function, uT::Function, cal_V::Function)
    M0 = m0.(sgrid)
    C = hs * sum(M0)
    M0 = M0 ./ C
    M = ones(node,N+1)
    M[:,1] = M0  # initial distribution
    U = zeros(node,N+1)
    U[:,end] = uT.(sgrid) # final cost
    V = float(cal_V.(sgrid)) # potential
    M_old = copy(M)
    U_old = copy(U)  
    return (M, U, V, M_old, U_old)
end

function Initial_1d_Q(node::Int64, N::Int64)
    # initial guess control QL=QR=0
    QL = zeros(node,N+1)  
    QR = zeros(node,N+1)
    QL_new = copy(QL)
    QR_new = copy(QR)
    return (QL, QR, QL_new, QR_new)
end

function solve_FP_1d_helper!(
    M::Matrix{Float64}, QL::Matrix{Float64}, QR::Matrix{Float64}, 
    N::Int64, ht::Float64, ε::Float64, A::SparseMatrixCSC{Float64,Int64},
    DL::SparseMatrixCSC{Float64,Int64}, DR::SparseMatrixCSC{Float64,Int64})
    # solve FP equation with control
    @inbounds for ti in 2:N+1
        lhs =  I - ht .* (ε .* A + (DR*spdiagm(0=>QL[:,ti]) + DL*spdiagm(0=>QR[:,ti])))
        M[:,ti] = lhs \ M[:,ti-1]
    end
    return nothing
end

function solve_HJB_1d_helper!(
    U::Matrix{Float64}, M::Matrix{Float64}, QL::Matrix{Float64}, QR::Matrix{Float64}, 
    N::Int64, ht::Float64, ε::Float64, V::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64},
    DL::SparseMatrixCSC{Float64,Int64}, DR::SparseMatrixCSC{Float64,Int64},
    F1::Function, F2::Function)
    # solve HJB equation with control and M
    @inbounds for ti in N:-1:1  
        lhs = I - ht .* (ε .* A - (spdiagm(0=>QL[:,ti])*DL + spdiagm(0=>QR[:,ti])*DR))
        rhs = U[:,ti+1] + ht .* (0.5 .* F1.(M[:,ti+1]) .*(QL[:,ti+1].^2 + QR[:,ti+1].^2) + V +  F2.(M[:,ti+1]))
        U[:,ti] = lhs \ rhs
    end
    return nothing
end

function update_control!(
    QL_new::Matrix{Float64}, QR_new::Matrix{Float64},
    U::Matrix{Float64}, M::Matrix{Float64},
    DL::SparseMatrixCSC{Float64,Int64}, 
    DR::SparseMatrixCSC{Float64,Int64},
    update_Q::Function)

    # update control Q from U and M
    QL_new .= update_Q.(max.(DL*U,0), M)
    QR_new .= update_Q.(min.(DR*U,0), M)
    return nothing
end

function compute_res_1d_helper(
    U::Matrix{Float64}, M::Matrix{Float64}, QL::Matrix{Float64}, QR::Matrix{Float64}, 
    N::Int64, ht::Float64, ε::Float64, V::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64},
    DL::SparseMatrixCSC{Float64,Int64}, DR::SparseMatrixCSC{Float64,Int64},
    F1::Function, F2::Function, hs::Float64)
    resFP, resHJB = 0, 0
    @inbounds for ti in 2:N+1
        lhs =  I - ht .* (ε .* A + 
                (DR*spdiagm(0=>QL[:,ti])+DL*spdiagm(0=>QR[:,ti]))
                )
        rhs = M[:,ti-1]
        resFP += sum(abs2.(lhs*M[:,ti]-rhs))
    end
    resFP = sqrt(hs*ht*resFP)

    @inbounds for ti in N:-1:1  
        lhs = I - ht .* (ε .* A -
                (spdiagm(0=>QL[:,ti])*DL+spdiagm(0=>QR[:,ti])*DR)
                )
        rhs = U[:,ti+1] + ht .* (0.5 .* F1.(M[:,ti+1]) .*(QL[:,ti+1].^2 + QR[:,ti+1].^2) + V +  F2.(M[:,ti+1]))
        resHJB += sum(abs2.(lhs*U[:,ti]-rhs))
    end
    resHJB = sqrt(hs*ht*resHJB)
    return (resFP, resHJB)
end