using LinearAlgebra, SparseArrays

"""
when compute residual,still use the formula in policy iteration,
because after update Q, the plug Q_new in mfg system, get same result
as nonlinear one.
"""
# Dimension 1
function compute_res_helper(
    U::Matrix{T}, M::Matrix{T}, 
    Q::NamedTuple{<:Any, NTuple{2, Matrix{T}}},
    N::Int64, ht::T, ε::T, V::Vector{T}, A::SparseMatrixCSC{T,Int64},
    D::NamedTuple{<:Any, NTuple{2, SparseMatrixCSC{T,Int64}}},
    F1::Function, F2::Function, hs::T) where {T<:Float64}

    resFP, resHJB = 0, 0
    for ti in 2:N+1
        lhs =  I/ht - (ε .* A - sum(map((q,d)->spdiagm(q[:,ti-1])*d, values(Q), values(D))))
        rhs = M[:,ti-1] ./ ht
        resFP += sum(abs2.(lhs' *M[:,ti]-rhs))
    end
    resFP = sqrt(hs*ht*resFP)

    for ti in N:-1:1  
        temp = -(U[:,ti+1]-U[:,ti]) ./ ht - 
                ε .* A * U[:,ti] + (0.5 .*  (F1.(M[:,ti+1])) .*sum(map(q->q[:,ti].^2, Q))) - 
                V - F2.(M[:,ti+1])
    
        resHJB += sum(abs2.(temp))
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
        lhs =  I/ht -  (ε .* A - sum(map((q,d)->spdiagm(q[:,ti-1])*d, values(Q), values(D))))
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

function compute_res_helper_non_quad(
    U::Matrix{T}, M::Matrix{T}, 
    Q::NamedTuple{<:Any, NTuple{4, Matrix{T}}},
    N::Int64, ht::T, ε::T, V::Vector{T}, A::SparseMatrixCSC{T,Int64},
    D::NamedTuple{<:Any, NTuple{4, SparseMatrixCSC{T,Int64}}},
    F1::Function, F2::Function, hs1::T, hs2::T) where {T<:Float64}

    resFP, resHJB = 0, 0
    for ti in 2:N+1
        lhs =  I/ht -  (ε .* A - sum(map((q,d)->spdiagm(q[:,ti-1])*d, values(Q), values(D))))
        rhs = M[:,ti-1] ./ ht
        resFP += sum(abs2.(lhs' *M[:,ti]-rhs))
    end
    resFP = sqrt(hs1*hs2*ht*resFP)

    for ti in N:-1:1  
        temp = -(U[:,ti+1]-U[:,ti]) ./ ht - 
                ε .* A * U[:,ti] + (1/3) .*(F1.(M[:,ti+1])) .*sum(map(q->q[:,ti].^2, Q)).^0.75 - 
                V - F2.(M[:,ti+1])
    
        resHJB += sum(abs2.(temp))
    end
    resHJB = sqrt(hs1*hs2*ht*resHJB)
    return (resFP, resHJB)
end