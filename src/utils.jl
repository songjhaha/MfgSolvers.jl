using LinearAlgebra, SparseArrays

function L_Inf_norm(u::Array{Float64, 2})
    norm = maximum(abs.(u))
    return norm
end


function build_Linear_operator(node::Int,hs::Float64)
    temp1 = 1/hs^2
    temp2 = 1/hs
    Δ = spdiagm(0=>fill(-2temp1,node), 
                    1=>fill(temp1,node-1), 
                    -1=>fill(temp1,node-1),
                    node-1=>fill(temp1,1),
                    -(node-1)=>fill(temp1,1))
    DL = spdiagm(0=>fill(temp2,node), 
                -1=>fill(-temp2,node-1), 
                node-1=>fill(-temp2,1))
    DR = spdiagm(0=>fill(-temp2,node), 
                1=>fill(temp2,node-1), 
                -(node-1)=>fill(temp2,1))
    return (Δ,(;DL,DR))
end

function build_Linear_operator_TwoDim(node1::Int,node2::Int, hs1::Float64,hs2::Float64)
    Δ1,D1 = build_Linear_operator(node1,hs1)
    Δ2,D2 = build_Linear_operator(node2,hs2)
    eye1 = sparse(I,node1,node1)
    eye2 = sparse(I,node2,node2)
    Δ = kron(Δ2,eye1) + kron(eye2,Δ1)
    DL1 = kron(eye2,D1.DL)
    DR1 = kron(eye2,D1.DR)
    DL2 = kron(D2.DL,eye1)
    DR2 = kron(D2.DR,eye1)
    return (Δ,(;DL1,DR1,DL2,DR2))
end

abstract type MFGProblem end
abstract type MFGResult end

struct Solver_history
    hist_q::Vector{Float64}
    hist_m::Vector{Float64}
    hist_u::Vector{Float64}
    residual_FP::Vector{Float64}
    residual_HJB::Vector{Float64}
end

struct MFGOneDim<:MFGProblem
    xmin::Float64
    xmax::Float64
    T::Float64
    ε::Float64
    m0::Function
    uT::Function
    V::Function
    F1::Function
    F2::Function
    update_Q::Function
end

struct MFGTwoDim<:MFGProblem
    xmin1::Float64
    xmax1::Float64
    xmin2::Float64
    xmax2::Float64
    T::Float64
    ε::Float64
    m0::Function
    uT::Function
    V::Function
    F1::Function
    F2::Function
    update_Q::Function
end

struct MFGOneDim_result<:MFGResult
    converge::Bool
    M::Array{Float64, 2}
    U::Array{Float64, 2}
    Q::NamedTuple{(:QL,:QR), NTuple{2, Matrix{Float64}}}
    sgrid::Vector{Float64}
    tgrid::Vector{Float64}
    iter::Int
    history::Solver_history
end

struct MFGTwoDim_result<:MFGResult
    converge::Bool
    M::Array{Float64, 3}
    U::Array{Float64, 3}
    Q::NamedTuple{(:QL1,:QR1, :QL2, :QR2), NTuple{4, Matrix{Float64}}}
    sgrid1::Vector{Float64}
    sgrid2::Vector{Float64}
    tgrid::Vector{Float64}
    iter::Int
    history::Solver_history
end

function Base.show(io::IO, x::MFGResult)
    println(io, "Converge: ",  x.converge)
    println(io, "iterations: ", x.iter)
    println(io, "FP_Residual: ",  x.history.residual_FP[end])
    println(io, "HJB_Residual: ",  x.history.residual_HJB[end])
end

Base.show(io::IO, m::MIME"text/plain", x::MFGResult) = show(io, x)


function solve_FP_helper!(
    M::Matrix{Float64}, 
    Q::NamedTuple{<:Any, NTuple{Dim, Matrix{T}}},
    N::Int64, ht::Float64, ε::Float64, A::SparseMatrixCSC{Float64,Int64},
    D::NamedTuple{<:Any, NTuple{Dim, SparseMatrixCSC{T,Int64}}}) where {T<:Float64, Dim}
    # solve FP equation with control
    @inbounds for ti in 2:N+1
        lhs = I - ht .* (ε .* A - sum(map((q,d)->spdiagm(q[:,ti])*d, values(Q), values(D))))
        M[:,ti] = lhs' \ M[:,ti-1]
    end
    return nothing
end

function solve_HJB_helper!(
    U::Matrix{T}, M::Matrix{T}, 
    Q::NamedTuple{<:Any, NTuple{Dim, Matrix{T}}}, 
    N::Int64, ht::T, ε::T, V::Vector{T}, A::SparseMatrixCSC{T,Int64},
    D::NamedTuple{<:Any, NTuple{Dim, SparseMatrixCSC{T,Int64}}},
    F1::Function, F2::Function) where {T<:Float64, Dim}
    # solve HJB equation with control and M
    @inbounds for ti in N:-1:1  
        lhs = I - ht .* (ε .* A - sum(map((q,d)->spdiagm(q[:,ti])*d, values(Q), values(D))))
        rhs = U[:,ti+1] + ht .*  (0.5 .*  F1.(M[:,ti+1]) .*sum(map(q->q[:,ti+1].^2, Q)) + V + F2.(M[:,ti+1]))
        U[:,ti] = lhs \ rhs
    end
    return nothing
end

# Dimension 1
function update_control!(
    Q_new::NamedTuple{<:Any, NTuple{4, Matrix{T}}},
    U::Matrix{T}, M::Matrix{T},
    D::NamedTuple{<:Any, NTuple{4, SparseMatrixCSC{T,Int64}}},
    update_Q::Function) where {T<:Float64}

    # update control Q from U and M
    Q_new.QL1 .= update_Q.(max.(D.DL1*U,0) , M)
    Q_new.QR1 .= update_Q.(min.(D.DR1*U,0) , M)
    Q_new.QL2 .= update_Q.(max.(D.DL2*U,0) , M)
    Q_new.QR2 .= update_Q.(min.(D.DR2*U,0) , M)
    return nothing
end

# Dimension 2
function update_control!(
    Q_new::NamedTuple{<:Any, NTuple{2, Matrix{T}}},
    U::Matrix{T}, M::Matrix{T},
    D::NamedTuple{<:Any, NTuple{2, SparseMatrixCSC{T,Int64}}},
    update_Q::Function) where {T<:Float64}

    # update control Q from U and M
    Q_new.QL .= update_Q.(max.(D.DL*U,0) , M)
    Q_new.QR .= update_Q.(min.(D.DR*U,0) , M)
    return nothing
end

# Dimension 1
function compute_res_helper(
    U::Matrix{T}, M::Matrix{T}, 
    Q::NamedTuple{<:Any, NTuple{2, Matrix{T}}},
    N::Int64, ht::T, ε::T, V::Vector{T}, A::SparseMatrixCSC{T,Int64},
    D::NamedTuple{<:Any, NTuple{2, SparseMatrixCSC{T,Int64}}},
    F1::Function, F2::Function, hs::T) where {T<:Float64}

    resFP, resHJB = 0, 0
    for ti in 2:N+1
        lhs =  I/ht - (ε .* A - sum(map((q,d)->spdiagm(q[:,ti])*d, values(Q), values(D))))
        rhs = M[:,ti-1] ./ ht
        resFP += sum(abs2.(lhs' *M[:,ti]-rhs))
    end
    resFP = sqrt(hs*ht*resFP)

    for ti in N:-1:1  
        lhs = I/ht - (ε .* A - sum(map((q,d)->spdiagm(q[:,ti])*d, values(Q), values(D))))
        rhs = U[:,ti+1] ./ ht + (0.5 .*  (F1.(M[:,ti+1])) .*sum(map(q->q[:,ti+1].^2, Q)) + V + F2.(M[:,ti+1]))
    
        resHJB += sum(abs2.(lhs*U[:,ti]-rhs))
    end
    resHJB = sqrt(hs*ht*resHJB)
    return (resFP, resHJB)
end

# Dimension 2
function compute_res_helper(
    U::Matrix{T}, M::Matrix{T}, 
    Q::NamedTuple{<:Any, NTuple{4, Matrix{T}}},
    N::Int64, ht::T, ε::T, V::Vector{T}, A::SparseMatrixCSC{T,Int64},
    D::NamedTuple{<:Any, NTuple{4, SparseMatrixCSC{T,Int64}}},
    F1::Function, F2::Function, hs1::T, hs2::T) where {T<:Float64}

    resFP, resHJB = 0, 0
    for ti in 2:N+1
        lhs =  I/ht -  (ε .* A - sum(map((q,d)->spdiagm(q[:,ti])*d, values(Q), values(D))))
        rhs = M[:,ti-1] ./ ht
        resFP += sum(abs2.(lhs' *M[:,ti]-rhs))
    end
    resFP = sqrt(hs1*hs2*ht*resFP)

    for ti in N:-1:1  
        lhs = I/ht -  (ε .* A - sum(map((q,d)->spdiagm(q[:,ti])*d, values(Q), values(D))))
        rhs = U[:,ti+1] ./ht + (0.5 .*  (F1.(M[:,ti+1])) .*sum(map(q->q[:,ti+1].^2, Q)) + V + F2.(M[:,ti+1]))
    
        resHJB += sum(abs2.(lhs*U[:,ti]-rhs))
    end
    resHJB = sqrt(hs1*hs2*ht*resHJB)
    return (resFP, resHJB)
end

