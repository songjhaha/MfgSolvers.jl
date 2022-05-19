using MfgSolvers
using LaTeXStrings
using Plots

L_inf_dist(a,b)=maximum(abs.(a-b)) # helper function for compute L inf distance

"""
################# case 1 #####################
"""
function OneDimTest1()    
    xmin = 0.0    
    xmax = 1.0    
    T = 1    
    ε = 0.05    
    m0(x) = ((x>=0.375) && (x<=0.625)) ? 4 : 0
    uT(x) = 10min( (x-0.3)^2, (x-0.7)^2 )
    V(x) = 0
    F1(m) = 1
    F2(m) = m
    update_Q(Du,m) = Du / F1(m)
    problem = MFGOneDim(xmin,xmax,T,ε,m0,uT,V,F1,F2,update_Q)
    re = solve_mfg(problem;method=:PI_FP,node=200,N=200,maxit=40,tol=1e-16,verbose=true)
    return re
end

re_OneDim = OneDimTest1()

begin
    m_pic = plot(re_OneDim.sgrid, re_OneDim.M[:,1], lw=2, label="t=0")
    for ti in [41,81,121,161,201]
        plot!(m_pic, re_OneDim.sgrid, re_OneDim.M[:,ti], lw=2, label="t=$((ti-1)/200)")
    end
    xlabel!("x")
    ylabel!("M")
end

function OneDimTest2()    
    xmin = 0.0    
    xmax = 1.0    
    T = 1    
    ε = 0.05    
    m0(x) = ((x>=0.375) && (x<=0.625)) ? 4 : 0
    uT(x) = 10min( (x-0.3)^2, (x-0.7)^2 )
    V(x) = 0
    F1(m) = 1
    F2(m) = m
    update_Q(Du,m) = Du / F1(m)
    problem = MFGOneDim(xmin,xmax,T,ε,m0,uT,V,F1,F2,update_Q)
    re = solve_mfg(problem;method=:PI1,node=200,N=200,maxit=40,tol=1e-16,verbose=true)
    return re
end

re_OneDim2 = OneDimTest2()

begin
    m_pic2 = plot(re_OneDim2.sgrid, re_OneDim2.M[:,1], lw=2, label="t=0")
    for ti in [41,81,121,161,201]
        plot!(m_pic2, re_OneDim2.sgrid, re_OneDim2.M[:,ti], lw=2, label="t=$((ti-1)/200)")
    end
    xlabel!("x")
    ylabel!("M")
end