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
    return (Δ,DL,DR)
end

function build_Linear_operator_TwoDim(node1::Int,node2::Int, hs1::Float64,hs2::Float64)
    Δ1,DL1,DR1 = build_Linear_operator(node1,hs1)
    Δ2,DL2,DR2 = build_Linear_operator(node2,hs2)
    eye1 = sparse(I,node1,node1)
    eye2 = sparse(I,node2,node2)
    Δ = kron(Δ2,eye1) + kron(eye2,Δ1)
    DL1 = kron(eye2,DL1)
    DR1 = kron(eye2,DR1)
    DL2 = kron(DL2,eye1)
    DR2 = kron(DR2,eye1)
    return (Δ,DL1,DR1,DL2,DR2)
end

abstract type MFGProblem end
abstract type MFGResult end

struct Solver_history
    hist_q::Vector{Float64}
    hist_m::Vector{Float64}
    hist_u::Vector{Float64}
    residual_norm_FP::Float64
    residual_norm_HJB::Float64
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
    QL::Array{Float64, 2}
    QR::Array{Float64, 2}
    sgrid::Vector{Float64}
    tgrid::Vector{Float64}
    iter::Int
    history::Solver_history
end

struct MFGTwoDim_result<:MFGResult
    converge::Bool
    M::Array{Float64, 3}
    U::Array{Float64, 3}
    QL1::Array{Float64, 3}
    QR1::Array{Float64, 3}
    QL2::Array{Float64, 3}
    QR2::Array{Float64, 3}
    sgrid1::Vector{Float64}
    sgrid2::Vector{Float64}
    tgrid::Vector{Float64}
    iter::Int
    history::Solver_history
end

function Base.show(io::IO, x::MFGResult)
    println(io, "Converge: ",  x.converge)
    println(io, "iterations: ", x.iter)
    println(io, "FP_Residual: ",  x.history.residual_norm_FP)
    println(io, "HJB_Residual: ",  x.history.residual_norm_HJB)
end

Base.show(io::IO, m::MIME"text/plain", x::MFGResult) = show(io, x)