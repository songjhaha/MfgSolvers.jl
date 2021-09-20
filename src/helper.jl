using LinearAlgebra, SparseArrays

"""
when solve FP use the tranpose of lhs in HJB.
"""
function solve_FP_helper!(
    M::Matrix{Float64}, 
    Q::NamedTuple{<:Any, NTuple{Dim, Matrix{T}}},
    N::Int64, ht::Float64, ε::Float64, A::SparseMatrixCSC{Float64,Int64},
    D::NamedTuple{<:Any, NTuple{Dim, SparseMatrixCSC{T,Int64}}}) where {T<:Float64, Dim}
    # solve FP equation with control
    for ti in 2:N+1
        lhs = I - ht .* (ε .* A - sum(map((q,d)->spdiagm(q[:,ti-1])*d, values(Q), values(D))))
        M[:,ti] = lhs' \ M[:,ti-1]
    end
    return nothing
end

"""
solve HJB with Q and M
"""
function solve_HJB_helper!(
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

"""
update Q_n = Du_n/F1(M_{n+1})
so the Hamiltonian always use M_{n+1} term.
"""
# Dimension 1
function update_control!(
    Q_new::NamedTuple{<:Any, NTuple{2, Matrix{T}}},
    U::Matrix{T}, M::Matrix{T},
    D::NamedTuple{<:Any, NTuple{2, SparseMatrixCSC{T,Int64}}},
    update_Q::Function) where {T<:Float64}

    # update control Q from U and M
    Q_new.QL .= update_Q.(max.(D.DL*U[:,1:end-1],0) , M[:,2:end])
    Q_new.QR .= update_Q.(min.(D.DR*U[:,1:end-1],0) , M[:,2:end])
    return nothing
end

# Dimension 2
function update_control!(
    Q_new::NamedTuple{<:Any, NTuple{4, Matrix{T}}},
    U::Matrix{T}, M::Matrix{T},
    D::NamedTuple{<:Any, NTuple{4, SparseMatrixCSC{T,Int64}}},
    update_Q::Function) where {T<:Float64}

    # update control Q from U and M
    Q_new.QL1 .= update_Q.(max.(D.DL1*U[:,1:end-1],0) , M[:,2:end])
    Q_new.QR1 .= update_Q.(min.(D.DR1*U[:,1:end-1],0) , M[:,2:end])
    Q_new.QL2 .= update_Q.(max.(D.DL2*U[:,1:end-1],0) , M[:,2:end])
    Q_new.QR2 .= update_Q.(min.(D.DR2*U[:,1:end-1],0) , M[:,2:end])
    return nothing
end


######################################
###### helper of fixed point iteration
######################################

# Dimension 1
function solve_HJB_fixpoint_helper!(
    U_new::Matrix{T}, U_old::Matrix{T}, M::Matrix{T}, 
    Q::NamedTuple{<:Any, NTuple{2, Matrix{T}}}, 
    N::Int64, ht::T, ε::T, V::Vector{T}, A::SparseMatrixCSC{T,Int64},
    D::NamedTuple{<:Any, NTuple{2, SparseMatrixCSC{T,Int64}}},
    F1::Function, F2::Function, update_Q::Function,hs::T) where {T<:Float64}
    """
    Q = DU_old / F1(M) 
    jacon * W = -res
    U_new = U_old + W   # loop with t
    """
    
    # update U with U_old and M
    for ti in N:-1:1
        U_temp = copy(U_old[:,ti])
        # U_temp = copy(U_new[:,ti+1])
        update_control!(Q, U_temp, M, D, update_Q, ti)

        for inner_it in 1:300
            jacon = 1/ht*I -  ε .* A + 1 .*
                (spdiagm(Q.QL[:,ti])*D.DL + spdiagm(Q.QR[:,ti])*D.DR)

            res = -(U_new[:,ti+1]-U_temp) ./ ht - ε .* A*U_temp + 
                    0.5 .* F1.(M[:,ti+1]) .* (Q.QL[:,ti].^2 + Q.QR[:,ti].^2) -
                    V - F2.(M[:,ti+1])
            
            # if residual is small enough, set U_new[:,ti], then go to solve U at ti-1
            if sqrt(hs)*norm(res) < 1e-12
                # println("norm_HJB_res: $(norm(res)), ti: $ti, inner_it: $inner_it") 
                U_new[:,ti] = U_temp
                break

            # if inner loop reach the max iteration, give some warning
            elseif inner_it==300
                println("inner HJB solver not converge")

            # if residual is not small enough, update U_temp
            else
                U_temp = jacon \ (-res) + U_temp
                update_control!(Q, U_temp, M, D, update_Q, ti)
            end
        end
    end
    return nothing
end

# Dimension 1 update Q in time ti 
function update_control!(
    Q_new::NamedTuple{<:Any, NTuple{2, Matrix{T}}},
    Uti::Vector{T}, M::Matrix{T},
    D::NamedTuple{<:Any, NTuple{2, SparseMatrixCSC{T,Int64}}},
    update_Q::Function, ti::Int64) where {T<:Float64}

    # update control Q from U and M
    Q_new.QL[:,ti] = update_Q.(max.(D.DL*Uti,0) , M[:,ti+1])
    Q_new.QR[:,ti] = update_Q.(min.(D.DR*Uti,0) , M[:,ti+1])
    return nothing
end

# Dimension 2
function solve_HJB_fixpoint_helper!(
    U_new::Matrix{T}, U_old::Matrix{T}, M::Matrix{T}, 
    Q::NamedTuple{<:Any, NTuple{4, Matrix{T}}}, 
    N::Int64, ht::T, ε::T, V::Vector{T}, A::SparseMatrixCSC{T,Int64},
    D::NamedTuple{<:Any, NTuple{4, SparseMatrixCSC{T,Int64}}},
    F1::Function, F2::Function, update_Q::Function,hs1::T,hs2::T) where {T<:Float64, Dim}
    """
    Q = DU_old / F1(M) 
    jacon * W = -res
    U_new = U_old + W   # loop with t
    """
    
    # update U with U_old and M
    for ti in N:-1:1
        U_temp = copy(U_old[:,ti])
        # U_temp = copy(U_new[:,ti+1])
        update_control!(Q, U_temp, M, D, update_Q, ti)

        for inner_it in 1:300
            jacon = 1/ht*I -  ε .* A + 1 .*
                (spdiagm(Q.QL1[:,ti])*D.DL1 + spdiagm(Q.QR1[:,ti])*D.DR1 +
                spdiagm(Q.QL2[:,ti])*D.DL2 + spdiagm(Q.QR2[:,ti])*D.DR2)

            res = -(U_new[:,ti+1]-U_temp) ./ ht - ε .* A*U_temp + 
                    0.5 .* F1.(M[:,ti+1]) .* (Q.QL1[:,ti].^2 + Q.QR1[:,ti].^2 + Q.QL2[:,ti].^2 + Q.QR2[:,ti].^2) -
                    V - F2.(M[:,ti+1])
            
            # if residual is small enough, set U_new[:,ti], then go to solve U at ti-1
            if sqrt(hs1*hs2)*norm(res) < 1e-12
                # println("norm_HJB_res: $(norm(res)), ti: $ti, inner_it: $inner_it") 
                U_new[:,ti] = U_temp
                break

            # if inner loop reach the max iteration, give some warning
            elseif inner_it==300
                println("inner HJB solver not converge")

            # if residual is not small enough, update U_temp
            else
                U_temp = jacon \ (-res) + U_temp
                update_control!(Q, U_temp, M, D, update_Q, ti)
            end
        end
    end
    return nothing
end

# Dimension 2 update Q in time ti 
function update_control!(
    Q_new::NamedTuple{<:Any, NTuple{4, Matrix{T}}},
    Uti::Vector{T}, M::Matrix{T},
    D::NamedTuple{<:Any, NTuple{4, SparseMatrixCSC{T,Int64}}},
    update_Q::Function, ti::Int64) where {T<:Float64}
    # update control Q from U and M
    Q_new.QL1[:,ti] = update_Q.(max.(D.DL1*Uti,0) , M[:,ti+1])
    Q_new.QR1[:,ti] = update_Q.(min.(D.DR1*Uti,0) , M[:,ti+1])
    Q_new.QL2[:,ti] = update_Q.(max.(D.DL2*Uti,0) , M[:,ti+1])
    Q_new.QR2[:,ti] = update_Q.(min.(D.DR2*Uti,0) , M[:,ti+1])
    return nothing
end


################################################
############# non quad case
################################################

function update_control_non_quad!(
    Q_new::NamedTuple{<:Any, NTuple{4, Matrix{T}}},
    U::Matrix{T}, M::Matrix{T},
    D::NamedTuple{<:Any, NTuple{4, SparseMatrixCSC{T,Int64}}},
    update_Q::Function) where {T<:Float64}
    
    size_Q_s, size_Q_t = size(Q_new.QL1)
    # update control Q from U and M
    for ti in 1:size_Q_t
        DLu1 = max.(D.DL1*U[:,ti],0)
        DRu1 = min.(D.DR1*U[:,ti],0)
        DLu2 = max.(D.DL2*U[:,ti],0)
        DRu2 = min.(D.DR2*U[:,ti],0)
        Du_norm = sqrt.(abs2.(DLu1)+abs2.(DRu1)+abs2.(DLu2)+abs2.(DRu2))
            
        Q_new.QL1[:,ti] = update_Q.(DLu1, Du_norm, M[:,ti+1]) 
        Q_new.QR1[:,ti] = update_Q.(DRu1, Du_norm, M[:,ti+1]) 
        Q_new.QL2[:,ti] = update_Q.(DLu2, Du_norm, M[:,ti+1]) 
        Q_new.QR2[:,ti] = update_Q.(DRu2, Du_norm, M[:,ti+1]) 
    end
    return nothing
end

function solve_HJB_helper_non_quad!(
    U::Matrix{T}, M::Matrix{T}, 
    Q::NamedTuple{<:Any, NTuple{Dim, Matrix{T}}}, 
    N::Int64, ht::T, ε::T, V::Vector{T}, A::SparseMatrixCSC{T,Int64},
    D::NamedTuple{<:Any, NTuple{Dim, SparseMatrixCSC{T,Int64}}},
    F1::Function, F2::Function) where {T<:Float64, Dim}
    # solve HJB equation with control and M
    
    for ti in N:-1:1  
        lhs = I - ht .* (ε .* A - sum(map((q,d)->spdiagm(q[:,ti])*d, values(Q), values(D))))
        rhs = U[:,ti+1] + ht .*  ((2/3) .*F1.(M[:,ti+1]) .*sum(map(q->q[:,ti].^2, Q)).^0.75 + V + F2.(M[:,ti+1]))
        U[:,ti] = lhs \ rhs
    end
    return nothing
end