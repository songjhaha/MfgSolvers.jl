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
        Q = Initial_1d_Q(node, N)
        Q_new = map(copy, Q)
        # Linear operators with periodic boundary
        A, D = build_Linear_operator(node,hs)
    end

    function solve_FP!(M, Q; N=N, ht=ht, ε=ε, A=A, D=D)
        solve_FP_helper!(M, Q, N, ht, ε, A, D)
    end

    function solve_HJB!(U, M, Q; N=N, ht=ht, ε=ε, V=V, A=A, D=D, F1=F1, F2=F2)
        solve_HJB_helper!(U, M, Q, N, ht, ε, V, A, D, F1, F2)
    end

    function compute_res(U, M, Q; N=N, ht=ht, ε=ε, V=V, A=A, D=D, F1=F1, F2=F2, hs=hs)
        compute_res_helper(U, M, Q, N, ht, ε, V, A, D, F1, F2, hs)
    end

    # Start Policy Iteration
    for iter in 1:maxit
        solve_FP!(M, Q)

        solve_HJB!(U, M, Q)     

        update_control!(Q_new, U, M, D, update_Q)
          
        Q, Q_new = Q_new, Q

        # record history
        L_dist_M = L_Inf_norm(M-M_old)
        L_dist_U = L_Inf_norm(U-U_old)
        L_dist_Q = map((q,q_new)->L_Inf_norm(q-q_new), Q, Q_new) |> maximum
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

            resFP, resHJB = compute_res(U, M, Q)
            verbose && println("M L2 residual $resFP")
            verbose && println("U L2 residual $resHJB")
            break            
        end
        if iter == maxit
            println("error! not converge!")
        end
    end
    history = Solver_history(hist_q,hist_m,hist_u,resFP,resHJB)
    result = MFGOneDim_result(converge,M,U,Q,sgrid,tgrid,length(hist_q),history)

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
        Q = Initial_1d_Q(node, N)
        Q_new = map(copy, Q)
        Q_tilde = map(copy, Q)
        # Linear operators with periodic boundary
        A, D = build_Linear_operator(node,hs)
    end


    function solve_FP!(M, Q; N=N, ht=ht, ε=ε, A=A, D=D)
        solve_FP_helper!(M, Q, N, ht, ε, A, D)
    end

    function solve_HJB!(U, M, Q; N=N, ht=ht, ε=ε, V=V, A=A, D=D, F1=F1, F2=F2)
        solve_HJB_helper!(U, M, Q, N, ht, ε, V, A, D, F1, F2)
    end

    function compute_res(U, M, Q; N=N, ht=ht, ε=ε, V=V, A=A, D=D, F1=F1, F2=F2, hs=hs)
        compute_res_helper(U, M, Q, N, ht, ε, V, A, D, F1, F2, hs)
    end

    # Start Policy Iteration
    for iter in 1:maxit
        solve_FP!(M, Q)

        update_control!(Q_tilde, U, M, D, update_Q)

        solve_HJB!(U, M, Q_tilde)     

        update_control!(Q_new, U, M, D, update_Q)
          
        Q, Q_new = Q_new, Q

        # record history
        L_dist_M = L_Inf_norm(M-M_old)
        L_dist_U = L_Inf_norm(U-U_old)
        L_dist_Q = map((q,q_new)->L_Inf_norm(q-q_new), Q, Q_new) |> maximum
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

            resFP, resHJB = compute_res(U, M, Q)
            verbose && println("M L2 residual $resFP")
            verbose && println("U L2 residual $resHJB")
            break            
        end
        if iter == maxit
            println("error! not converge!")
        end
    end
    history = Solver_history(hist_q,hist_m,hist_u,resFP,resHJB)
    result = MFGOneDim_result(converge,M,U,Q,sgrid,tgrid,length(hist_q),history)

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
    return (;QL, QR)
end
