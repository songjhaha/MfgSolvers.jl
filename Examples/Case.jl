using MfgSolvers
using LaTeXStrings
using Plots

function OneDimTest1(method::Symbol)    
    xmin = 0.0    
    xmax = 1.0    
    T = 1    
    ε = 0.05    
    m0(x) = ((x>=0.375) && (x<=0.625)) ? 4 : 0
    uT(x) = 10min( (x-0.3)^2, (x-0.7)^2 )
    V(x) = 0.1
    F1(m) = (1+4m)^1.5
    F2(m) = m
    update_Q(Du,m) = Du / F1(m)
    problem = MFGOneDim(xmin,xmax,T,ε,m0,uT,V,F1,F2,update_Q)
    re = solve_mfg(problem;method=method,node=200,N=200,verbose=true) # iter 10, Q diff 7e-9
    return re
end

function TwoDimTest4_PI1(Nh::Int64)
    xmin1, xmax1, xmin2, xmax2 = 0, 1, 0, 1
    T = 0.5
    ε = 0.3
    m0(x1,x2) = exp(-10((x1-0.25)^2+(x2-0.25)^2))
    uT(x1,x2) = 1.2*cospi(2*x1) + cospi(2*x2)
    V(x1,x2) = 0.1
    F1(m) = m^0.5
    F2(m) = 0
    update_Q(Du,m) = Du / F1(m)
    problem = MFGTwoDim(xmin1,xmax1,xmin2,xmax2,T,ε,m0,uT,V,F1,F2,update_Q)
    re = solve_mfg(problem;node1=Nh,node2=Nh,N=50) 
    return re
end

function TwoDimTest4_PI2(Nh::Int64)
    xmin1, xmax1, xmin2, xmax2 = 0, 1, 0, 1
    T = 0.5
    ε = 0.3
    m0(x1,x2) = exp(-10((x1-0.25)^2+(x2-0.25)^2))
    uT(x1,x2) = 1.2*cospi(2*x1) + cospi(2*x2)
    V(x1,x2) = 0.1
    F1(m) = m^0.5
    F2(m) = 0
    update_Q(Du,m) = Du / F1(m)
    # Nh=100
    problem = MFGTwoDim(xmin1,xmax1,xmin2,xmax2,T,ε,m0,uT,V,F1,F2,update_Q) 
    re_algo2 = solve_mfg(problem;method=:PI2,node1=Nh,node2=Nh,N=50,verbose=true)
    return re_algo2
end

function TwoDimTest4_fixpoint(Nh::Int64)
    xmin1, xmax1, xmin2, xmax2 = 0, 1, 0, 1
    T = 0.5
    ε = 0.3
    m0(x1,x2) = exp(-10((x1-0.25)^2+(x2-0.25)^2))
    uT(x1,x2) = 1.2*cospi(2*x1) + cospi(2*x2)
    V(x1,x2) = 0.1
    F1(m) = m^0.5
    F2(m) = 0
    update_Q(Du,m) = Du / F1(m)
    # Nh = 100
    problem = MFGTwoDim(xmin1,xmax1,xmin2,xmax2,T,ε,m0,uT,V,F1,F2,update_Q) 
    re_algo2 = solve_mfg_fixpoint(problem;method=:FixPoint2,node1=Nh,node2=Nh,N=50,maxit=100,verbose=true)
    return re_algo2
end

re_OneDim = OneDimTest1(:PI1)
# case1
begin
    
    plot(re_OneDim.history.hist_q, yaxis=:log, label=L"\Vert q^{(k+1)}-q^{(k)} \Vert")
    plot!(re_OneDim.history.hist_m, yaxis=:log, label=L"\Vert m^{(k+1)}-m^{(k)} \Vert")
    plot!(re_OneDim.history.hist_u, yaxis=:log, label=L"\Vert u^{(k+1)}-u^{(k)} \Vert")
    savefig("converge_case1.pdf")

    begin
        m_pic = plot(re_OneDim.sgrid, re_OneDim.M[:,1], label="t=0")
        for ti in [41,81,121,161,201]
            plot!(m_pic, re_OneDim.sgrid, re_OneDim.M[:,ti], label="t=$((ti-1)/200)")
        end
        savefig("M_case1.pdf")
    end

    begin
        q_pic = plot(re_OneDim.sgrid, (re_OneDim.Q.QL+re_OneDim.Q.QR)[:,1], label="t=0",legend=:bottomright)
        for ti in [40,80,120,160,200]
            plot!(q_pic,re_OneDim.sgrid, (re_OneDim.Q.QL+re_OneDim.Q.QR)[:,ti], label="t=$(ti/200)",legend=:bottomright)
        end
        savefig("Q_case1.pdf")
    end
    plot(re_OneDim.history.residual_HJB, yaxis=:log, label="HJB")
    plot!(re_OneDim.history.residual_FP, yaxis=:log, label="FP")
    savefig("case1_residual.pdf")

    re_OneDim_PI2 = OneDimTest1(:PI2)
    begin
        m_pic = plot(re_OneDim_PI2.sgrid, re_OneDim_PI2.M[:,1], label="t=0")
        for ti in [41,81,121,161,201]
            plot!(m_pic, re_OneDim_PI2.sgrid, re_OneDim_PI2.M[:,ti], label="t=$((ti-1)/200)")
        end
        savefig("M_case1_PI2.pdf")
    end

    begin
        q_pic = plot(re_OneDim_PI2.sgrid, (re_OneDim_PI2.Q.QL+re_OneDim_PI2.Q.QR)[:,1], label="t=0",legend=:bottomright)
        for ti in [40,80,120,160,200]
            plot!(q_pic,re_OneDim_PI2.sgrid, (re_OneDim_PI2.Q.QL+re_OneDim_PI2.Q.QR)[:,ti], label="t=$(ti/200)",legend=:bottomright)
        end
        savefig("Q_case1_PI2.pdf")
    end
end

L_inf_dist(a,b)=maximum(abs.(a-b))
hist = Float64[]
for Q in re_OneDim.Q_List
    dist = maximum(map(L_inf_dist, Q,re_OneDim.Q_List[end]))
    append!(hist,dist)
end
plot(hist[1:end-1],yaxis=:log, label=L"\Vert q^{(k)}-q^{*} \Vert")

hist = Float64[]
for M in re_OneDim.M_List
    dist = maximum(abs.(M-re_OneDim.M_List[end]))
    append!(hist,dist)
end
plot!(hist[1:end-1],yaxis=:log, label=L"\Vert m^{(k)}-m^{*} \Vert")

hist = Float64[]
for U in re_OneDim.U_List
    dist = maximum(abs.(U-re_OneDim.U_List[end]))
    append!(hist,dist)
end
plot!(hist[1:end-1],yaxis=:log, label=L"\Vert u^{(k)}-u^{*} \Vert")
savefig("total_converge_case1.pdf")

re_PI1 = TwoDimTest4_PI1(50)
re_PI2_Nh50 = TwoDimTest4_PI2(50)
re_fixpoint2_Nh50 = TwoDimTest4_fixpoint(50)

@time for i in 1:5 # 148.66s
    TwoDimTest4_PI2(50)
end

@time for i in 1:5 # 281.4s
    TwoDimTest4_fixpoint(50)
end

@time for i in 1:5
    TwoDimTest4_PI2(100)
end

@time for i in 1:5
    TwoDimTest4_fixpoint(100)
end

@time for i in 1:5
    TwoDimTest4_PI2(200)
end

@time for i in 1:5
    TwoDimTest4_fixpoint(200)
end

# Mass policy iteration
contour(re_PI2_Nh50.sgrid2,re_PI2_Nh50.sgrid1,re_PI2_Nh50.M[:,:,1],c=:rainbow,title="M,t=0",size=(450,400))
savefig("M_case2_t0.pdf")
contour(re_PI2_Nh50.sgrid2,re_PI2_Nh50.sgrid1,re_PI2_Nh50.M[:,:,17],c=:rainbow,title="M,t=0.16",size=(450,400))
savefig("M_case2_t016.pdf")
contour(re_PI2_Nh50.sgrid2,re_PI2_Nh50.sgrid1,re_PI2_Nh50.M[:,:,34],c=:rainbow,title="M,t=0.33",size=(450,400))
savefig("M_case2_t033.pdf")
contour(re_PI2_Nh50.sgrid2,re_PI2_Nh50.sgrid1,re_PI2_Nh50.M[:,:,end],levels=0:0.5:4.5,c=:rainbow,title="M,t=0.5",size=(450,400))
savefig("M_case2_t05.pdf")

# Mass fixpoint
contour(re_fixpoint2_Nh50.sgrid2,re_fixpoint2_Nh50.sgrid1,re_fixpoint2_Nh50.M[:,:,1],c=:rainbow,title="M,t=0",size=(450,400))
savefig("M_case2_fixpoint_t0.pdf")
contour(re_fixpoint2_Nh50.sgrid2,re_fixpoint2_Nh50.sgrid1,re_fixpoint2_Nh50.M[:,:,17],c=:rainbow,title="M,t=0.16",size=(450,400))
savefig("M_case2_fixpoint_t016.pdf")
contour(re_fixpoint2_Nh50.sgrid2,re_fixpoint2_Nh50.sgrid1,re_fixpoint2_Nh50.M[:,:,34],c=:rainbow,title="M,t=0.33",size=(450,400))
savefig("M_case2_fixpoint_t033.pdf")
contour(re_fixpoint2_Nh50.sgrid2,re_fixpoint2_Nh50.sgrid1,re_fixpoint2_Nh50.M[:,:,end],levels=0:0.5:4.5,c=:rainbow,title="M,t=0.5",size=(450,400))
savefig("M_case2_fixpoint_t05.pdf")

# residual fixPoint & PI2
plot(re_fixpoint2_Nh50.history.residual_HJB, yaxis=:log, label="Fixed Point Iteration")
plot!(re_PI2_Nh50.history.residual_HJB, yaxis=:log, label="Policy Iteration")
savefig("residual_HJB_Nh50.pdf")

plot(re_fixpoint2_Nh50.history.residual_FP, yaxis=:log, label="Fixed Point Iteration")
plot!(re_PI2_Nh50.history.residual_FP, yaxis=:log, label="Policy Iteration")
savefig("residual_FP_Nh50.pdf")


# PI1 & PI2
plot(re_PI1.history.residual_HJB, yaxis=:log, label="PI1-HJB")
plot!(re_PI2_Nh50.history.residual_HJB, yaxis=:log, label="PI2-HJB")
# savefig("residual_HJB_PI1_2.pdf")

plot!(re_PI1.history.residual_FP, yaxis=:log, label="PI1-FP")
plot!(re_PI2_Nh50.history.residual_FP, yaxis=:log, label="PI2-FP")
savefig("residual_PI1_2.pdf")

# converge case2 PI1&PI2
plot(re_PI1.history.hist_q, yaxis=:log, label="PI1")
plot!(re_PI2_Nh50.history.hist_q, yaxis=:log, label="PI2")
savefig("converge_case2_q.pdf")
plot(re_PI1.history.hist_m, yaxis=:log, label="PI1")
plot!(re_PI2_Nh50.history.hist_m, yaxis=:log, label="PI2")
savefig("converge_case2_m.pdf")
plot(re_PI1.history.hist_u, yaxis=:log, label="PI1")
plot!(re_PI2_Nh50.history.hist_u, yaxis=:log, label="PI2")
savefig("converge_case2_u.pdf")



hist = Float64[]
for Q in re_PI1.Q_List
    dist = maximum(map(L_inf_dist, Q,re_PI1.Q_List[end]))
    append!(hist,dist)
end
plot(hist[1:end-1],yaxis=:log, label="PI1")

hist = Float64[]
for Q in re_PI2_Nh50.Q_List
    dist = maximum(map(L_inf_dist, Q,re_PI2_Nh50.Q_List[end]))
    append!(hist,dist)
end
plot!(hist[1:end-1],yaxis=:log, label="PI2")
savefig("total_converge_case2_q.pdf")

hist = Float64[]
for M in re_PI1.M_List
    dist = maximum(abs.(M-re_PI1.M_List[end]))
    append!(hist,dist)
end
plot(hist[1:end-1],yaxis=:log, label="PI1")

hist = Float64[]
for M in re_PI2_Nh50.M_List
    dist = maximum(abs.(M-re_PI2_Nh50.M_List[end]))
    append!(hist,dist)
end
plot!(hist[1:end-1],yaxis=:log, label="PI2")
savefig("total_converge_case2_m.pdf")

hist = Float64[]
for U in re_PI1.U_List
    dist = maximum(abs.(U-re_PI1.U_List[end]))
    append!(hist,dist)
end
plot(hist[1:end-1],yaxis=:log, label="PI1")

hist = Float64[]
for U in re_PI2_Nh50.U_List
    dist = maximum(abs.(U-re_PI2_Nh50.U_List[end]))
    append!(hist,dist)
end
plot!(hist[1:end-1],yaxis=:log, label="PI2")
savefig("total_converge_case2_u.pdf")



function non_quad()
    xmin1, xmax1, xmin2, xmax2 = 0, 1, 0, 1
    T = 0.5
    ε = 0.3
    m0(x1,x2) = exp(-10((x1-0.25)^2+(x2-0.25)^2))
    uT(x1,x2) = 1.2*cospi(2*x1) + cospi(2*x2)
    V(x1,x2) = 0.1
    F1(m) = 1
    F2(m) = 0
    update_Q(DU,m) = 0
    Nh=50
    problem = MFGTwoDim(xmin1,xmax1,xmin2,xmax2,T,ε,m0,uT,V,F1,F2,update_Q)
    re = solve_mfg_non_quad(problem;method=:PI1, node1=Nh,node2=Nh,N=50) 
    return re
end

re = non_quad()


# contour(re.sgrid2,re.sgrid1,re.M[:,:,51],c=:rainbow,title="M,t=0.5",size=(450,400))

# Mass policy iteration
contour(re.sgrid2,re.sgrid1,re.M[:,:,1],c=:rainbow,title="M,t=0",size=(450,400))
# savefig("M_case2_t0.pdf")
contour(re.sgrid2,re.sgrid1,re.M[:,:,17],c=:rainbow,title="M,t=0.16",size=(450,400))
# savefig("M_case2_t016.pdf")
contour(re.sgrid2,re.sgrid1,re.M[:,:,34],c=:rainbow,title="M,t=0.33",size=(450,400))
# savefig("M_case2_t033.pdf")
contour(re.sgrid2,re.sgrid1,re.M[:,:,end],c=:rainbow,title="M,t=0.5",size=(450,400))
# savefig("M_case2_t05.pdf")

plot(re.history.hist_q, yaxis=:log, label="q")
plot!(re.history.hist_m, yaxis=:log, label="m")
plot!(re.history.hist_u, yaxis=:log, label="u")
plot(re.history.residual_HJB, yaxis=:log, label="HJB")
plot!(re.history.residual_FP, yaxis=:log, label="FP")



