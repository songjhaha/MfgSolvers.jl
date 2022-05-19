using LinearAlgebra, SparseArrays

# Inf norm 
function L_Inf_norm(u::Array{Float64, 2})
    inf_norm = maximum(abs.(u))
    return inf_norm
end

# Dimension 1 linear operators
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

# Dimension 2 linear operators, build by kron prod
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
    M_List::Vector{Array{Float64, 2}}
    U_List::Vector{Array{Float64, 2}}
    Q_List::Vector{NamedTuple{(:QL,:QR), NTuple{2, Matrix{Float64}}}}
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
    M_List::Vector{Array{Float64, 2}}
    U_List::Vector{Array{Float64, 2}}
    Q_List::Vector{NamedTuple{(:QL1,:QR1, :QL2, :QR2), NTuple{4, Matrix{Float64}}}}
end

function Base.show(io::IO, x::MFGResult)
    # println(io, "Converge: ",  x.converge)
    # println(io, "iterations: ", x.iter)
    println(io, "FP_Residual: ",  x.history.residual_FP[end])
    println(io, "HJB_Residual: ",  x.history.residual_HJB[end])
end

Base.show(io::IO, m::MIME"text/plain", x::MFGResult) = show(io, x)
