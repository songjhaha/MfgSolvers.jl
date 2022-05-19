using MfgSolvers
using LaTeXStrings
using Plots

L_inf_dist(a,b)=maximum(abs.(a-b))

function OneDimTestLongTime(α::Float64,zeta::Float64, T::Real, method=:PI1, ε = 0.05, init_Q = nothing)    
    xmin = 0.0    
    xmax = 1.0    
    T = T    
    ε = ε    
    m0(x) = ((x>=0.375) && (x<=0.625)) ? 4 : 0
    uT(x) = 10min( (x-0.3)^2, (x-0.7)^2 )
    V(x) = 0
    F1(m) = (1+4m)^α
    F2(m) = zeta*m
    update_Q(Du,m) = Du / F1(m)
    problem = MFGOneDim(xmin,xmax,T,ε,m0,uT,V,F1,F2,update_Q)
    re = solve_mfg(problem,init_Q;method=method,node=200,N=floor(Int,200*T),maxit=60,tol=1e-16,verbose=true)
    return re
end

re1 = OneDimTestLongTime(1.5, 0.8, 5.0, :PI1, 0.05) 
# modify α and ζ




















# case1
begin 
    # plot(re_OneDim.history.hist_q, yaxis=:log, label=L"\Vert q^{(k+1)}-q^{(k)} \Vert")
    # plot!(re_OneDim.history.hist_m, yaxis=:log, label=L"\Vert m^{(k+1)}-m^{(k)} \Vert")
    # plot!(re_OneDim.history.hist_u, yaxis=:log, label=L"\Vert u^{(k+1)}-u^{(k)} \Vert")
    # savefig("converge_case1.pdf")
    begin
        m_pic = plot(re_OneDim.sgrid, re_OneDim.M[:,1], lw=2, label="t=0")
        for ti in [201,401, 601,801,1001]
            plot!(m_pic, re_OneDim.sgrid, re_OneDim.M[:,ti], lw=2, label="t=$((ti-1)/1000)")
        end
        xlabel!("x")
        ylabel!("M")
        # savefig("figures/M_case1.pdf")
    end

    begin
        q_pic = plot(re_OneDim.sgrid, (re_OneDim.Q.QL+re_OneDim.Q.QR)[:,1], lw=2, label="t=0",legend=:bottom)
        for ti in [200,400,600,800,1000]
            plot!(q_pic,re_OneDim.sgrid, (re_OneDim.Q.QL+re_OneDim.Q.QR)[:,ti], lw=2, label="t=$(ti/1000)",legend=:bottom)
        end
        xlabel!("x")
        ylabel!("Q")
        # savefig("figures/Q_case1.pdf")
    end
    plot(re_OneDim.history.residual_HJB[1:40], lw=2, yaxis=:log, label="HJB")
    plot!(re_OneDim.history.residual_FP[1:40], lw=2, yaxis=:log, label="FP")
    xlabel!("Iteration")
    savefig("figures/case1_residual.pdf")
end


using MfgSolvers
using LaTeXStrings
using Plots

function OneDimTestLongTime(method=:PI1, ε = 0.3, init_Q = nothing)    
    xmin = 0.0    
    xmax = 1.0    
    T = 0.2    
    ε = ε    
    # m0(x) = ((x>=0.375) && (x<=0.625)) ? 4 : 0
    m0(x) = exp(-200*(x-0.25)^2)
    # uT(x) = 5min( (x-0.3)^2, (x-0.7)^2 )
    uT(x) = -exp(-200*(x-0.75)^2)

    V(x) = 0
    F1(m) = 1
    F2(m) = m
    update_Q(Du,m) = Du / F1(m)
    problem = MFGOneDim(xmin,xmax,T,ε,m0,uT,V,F1,F2,update_Q)
    re = solve_mfg(problem,init_Q;method=method,node=100,N=20,maxit=100,tol=1e-16,verbose=true)
    return re
end

re1 = OneDimTestLongTime(:PI1, 0.3) # converge
sum(re1.M[:,end])
plot(re1.sgrid,re1.M_List[end][:,1],label="t=0.00")
plot!(re1.sgrid,re1.M_List[end][:,2],label="t=0.01")
plot!(re1.sgrid,re1.M_List[end][:,7],label="t=0.06")
plot!(re1.sgrid,re1.M_List[end][:,12],label="t=0.11")
plot!(re1.sgrid,re1.M_List[end][:,17],label="t=0.16")
plot!(re1.sgrid,re1.M_List[end][:,21],label="t=0.20")