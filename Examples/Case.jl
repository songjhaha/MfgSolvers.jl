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
    V(x) = 0.1
    F1(m) = (1+4m)^1.5
    F2(m) = m
    update_Q(Du,m) = Du / F1(m)
    problem = MFGOneDim(xmin,xmax,T,ε,m0,uT,V,F1,F2,update_Q)
    re = solve_mfg(problem;method=:PI1,node=200,N=200,tol=1e-16,verbose=true)
    return re
end

function OneDimTest1_fixpoint()    
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
    re = solve_mfg_fixpoint(problem;method=:FixPoint2,node=200,N=200,tol=1e-16,verbose=true)
    return re
end

re_OneDim = OneDimTest1()

# q* get with 80 iterations
re_OneDim_fixpoint = OneDimTest1_fixpoint()

# case1
begin 
    # plot(re_OneDim.history.hist_q, yaxis=:log, label=L"\Vert q^{(k+1)}-q^{(k)} \Vert")
    # plot!(re_OneDim.history.hist_m, yaxis=:log, label=L"\Vert m^{(k+1)}-m^{(k)} \Vert")
    # plot!(re_OneDim.history.hist_u, yaxis=:log, label=L"\Vert u^{(k+1)}-u^{(k)} \Vert")
    # savefig("converge_case1.pdf")
    begin
        m_pic = plot(re_OneDim.sgrid, re_OneDim.M[:,1], lw=2, label="t=0")
        for ti in [41,81,121,161,201]
            plot!(m_pic, re_OneDim.sgrid, re_OneDim.M[:,ti], lw=2, label="t=$((ti-1)/200)")
        end
        xlabel!("x")
        ylabel!("M")
        savefig("figures/M_case1.pdf")
    end

    begin
        q_pic = plot(re_OneDim.sgrid, (re_OneDim.Q.QL+re_OneDim.Q.QR)[:,1], lw=2, label="t=0",legend=:bottom)
        for ti in [40,80,120,160,200]
            plot!(q_pic,re_OneDim.sgrid, (re_OneDim.Q.QL+re_OneDim.Q.QR)[:,ti], lw=2, label="t=$(ti/200)",legend=:bottom)
        end
        xlabel!("x")
        ylabel!("Q")
        savefig("figures/Q_case1.pdf")
    end
    plot(re_OneDim.history.residual_HJB[1:40], lw=2, yaxis=:log, label="HJB")
    plot!(re_OneDim.history.residual_FP[1:40], lw=2, yaxis=:log, label="FP")
    xlabel!("Iteration")
    savefig("figures/case1_residual.pdf")
end


hist = Float64[]
for Q in re_OneDim.Q_List
    dist = maximum(map(L_inf_dist, Q, re_OneDim_fixpoint.Q))
    append!(hist,dist)
end
plot(hist[1:40],yaxis=:log, lw=2, label=L"\Vert q^{(k)}-q^{*} \Vert")

hist = Float64[]
for M in re_OneDim.M_List
    dist = maximum(abs.(M-re_OneDim_fixpoint.M))
    append!(hist,dist)
end
plot!(hist[1:40],yaxis=:log, lw=2, label=L"\Vert m^{(k)}-m^{*} \Vert")

hist = Float64[]
for U in re_OneDim.U_List
    dist = maximum(abs.(U-re_OneDim_fixpoint.U))
    append!(hist,dist)
end
plot!(hist[1:40],yaxis=:log, lw=2, label=L"\Vert u^{(k)}-u^{*} \Vert")
xlabel!("Iteration")
savefig("figures/total_converge_case1.pdf")


"""
################## case2 ###########################
"""

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
    re = solve_mfg(problem;method=:PI1,node1=Nh,node2=Nh,N=50,tol=1e-16) 
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

    problem = MFGTwoDim(xmin1,xmax1,xmin2,xmax2,T,ε,m0,uT,V,F1,F2,update_Q) 
    re_algo2 = solve_mfg(problem;method=:PI2,node1=Nh,node2=Nh,N=50,verbose=true,tol=1e-16)
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

    problem = MFGTwoDim(xmin1,xmax1,xmin2,xmax2,T,ε,m0,uT,V,F1,F2,update_Q) 
    re_algo2 = solve_mfg_fixpoint(problem;method=:FixPoint2,node1=Nh,node2=Nh,N=50,maxit=80,verbose=true,tol=1e-16)
    return re_algo2
end
re_PI1 = TwoDimTest4_PI1(50)
re_PI2_Nh50 = TwoDimTest4_PI2(50)
re_fixpoint2_Nh50 = TwoDimTest4_fixpoint(50) 
# q* get with 80 iteration 

# Mass policy iteration 2
contour(re_PI2_Nh50.sgrid2,re_PI2_Nh50.sgrid1,re_PI2_Nh50.M[:,:,1],fill=true,c=:rainbow,size=(580,520))
xlabel!("x2")
ylabel!("x1")
savefig("figures/M_case2_t0.pdf")

contour(re_PI2_Nh50.sgrid2,re_PI2_Nh50.sgrid1,re_PI2_Nh50.M[:,:,17],fill=true,c=:rainbow,size=(580,520))
xlabel!("x2")
ylabel!("x1")
savefig("figures/M_case2_t016.pdf")

contour(re_PI2_Nh50.sgrid2,re_PI2_Nh50.sgrid1,re_PI2_Nh50.M[:,:,34],fill=true,c=:rainbow,size=(580,520))
xlabel!("x2")
ylabel!("x1")
savefig("figures/M_case2_t033.pdf")

contour(re_PI2_Nh50.sgrid2,re_PI2_Nh50.sgrid1,re_PI2_Nh50.M[:,:,end],fill=true,c=:rainbow,size=(580,520))
xlabel!("x2")
ylabel!("x1")
savefig("figures/M_case2_t05.pdf")

# Mass policy iteration 2 heatmap
heatmap(re_PI2_Nh50.sgrid2,re_PI2_Nh50.sgrid1,re_PI2_Nh50.M[:,:,1],c=:rainbow,size=(580,520))
xlabel!("x2")
ylabel!("x1")
savefig("figures/M_case2_t0_heatmap.pdf")

heatmap(re_PI2_Nh50.sgrid2,re_PI2_Nh50.sgrid1,re_PI2_Nh50.M[:,:,17],c=:rainbow,size=(580,520))
xlabel!("x2")
ylabel!("x1")
savefig("figures/M_case2_t016_heatmap.pdf")

heatmap(re_PI2_Nh50.sgrid2,re_PI2_Nh50.sgrid1,re_PI2_Nh50.M[:,:,34],c=:rainbow,size=(580,520))
xlabel!("x2")
ylabel!("x1")
savefig("figures/M_case2_t033_heatmap.pdf")

heatmap(re_PI2_Nh50.sgrid2,re_PI2_Nh50.sgrid1,re_PI2_Nh50.M[:,:,end],c=:rainbow,size=(580,520))
xlabel!("x2")
ylabel!("x1")
savefig("figures/M_case2_t05_heatmap.pdf")


# residual fixPoint & PI2
plot(re_fixpoint2_Nh50.history.residual_HJB, lw=2, yaxis=:log, label="Fixed Point Iteration")
plot!(re_PI2_Nh50.history.residual_HJB, lw=2, yaxis=:log, label="Policy Iteration")
xlabel!("Iteration")
ylabel!("Residual of HJB")
savefig("figures/residual_HJB_Nh50.pdf")

plot(re_fixpoint2_Nh50.history.residual_FP, lw=2, yaxis=:log, label="Fixed Point Iteration")
plot!(re_PI2_Nh50.history.residual_FP, lw=2, yaxis=:log, label="Policy Iteration")
xlabel!("Iteration")
ylabel!("Residual of FP")
savefig("figures/residual_FP_Nh50.pdf")


# PI1 & PI2 residual
plot(re_PI1.history.residual_HJB, lw=2, yaxis=:log, label="PI1-HJB")
plot!(re_PI2_Nh50.history.residual_HJB, lw=2, linestyles=:dash, yaxis=:log, label="PI2-HJB")
# savefig("residual_HJB_PI1_2.pdf")
plot!(re_PI1.history.residual_FP, lw=2, yaxis=:log, label="PI1-FP")
plot!(re_PI2_Nh50.history.residual_FP, lw=2, linestyles=:dash, yaxis=:log, label="PI2-FP")
xlabel!("Iteration")
ylabel!("Residual")
savefig("figures/residual_PI1_2.pdf")

begin
# converge case2 PI1&PI2 |q^{k+1}-q^{k}|
    # plot(re_PI1.history.hist_q, yaxis=:log, label="PI1")
    # plot!(re_PI2_Nh50.history.hist_q, yaxis=:log, label="PI2")
    # savefig("converge_case2_q.pdf")
    # plot(re_PI1.history.hist_m, yaxis=:log, label="PI1")
    # plot!(re_PI2_Nh50.history.hist_m, yaxis=:log, label="PI2")
    # savefig("converge_case2_m.pdf")
    # plot(re_PI1.history.hist_u, yaxis=:log, label="PI1")
    # plot!(re_PI2_Nh50.history.hist_u, yaxis=:log, label="PI2")
    # savefig("converge_case2_u.pdf")
end

# |q^{k}-q^*|
hist = Float64[]
for Q in re_PI1.Q_List
    dist = maximum(map(L_inf_dist, Q, re_fixpoint2_Nh50.Q))
    append!(hist,dist)
end
plot(hist[1:end],yaxis=:log, lw=2, label="PI1")

hist = Float64[]
for Q in re_PI2_Nh50.Q_List
    dist = maximum(map(L_inf_dist, Q, re_fixpoint2_Nh50.Q))
    append!(hist,dist)
end
plot!(hist[1:end],yaxis=:log, lw=2, label="PI2")
xlabel!("Iteration")
savefig("figures/total_converge_case2_q.pdf")

hist = Float64[]
for M in re_PI1.M_List
    dist = maximum(abs.(reshape(M,50,50,51)-re_fixpoint2_Nh50.M))
    append!(hist,dist)
end
plot(hist[1:end],yaxis=:log, lw=2, label="PI1")

hist = Float64[]
for M in re_PI2_Nh50.M_List
    dist = maximum(abs.(reshape(M,50,50,51)-re_fixpoint2_Nh50.M))
    append!(hist,dist)
end
plot!(hist[1:end],yaxis=:log, lw=2, label="PI2")
xlabel!("Iteration")
savefig("figures/total_converge_case2_m.pdf")

hist = Float64[]
for U in re_PI1.U_List
    dist = maximum(abs.(reshape(U,50,50,51)-re_fixpoint2_Nh50.U))
    append!(hist,dist)
end
plot(hist[1:end],yaxis=:log, lw=2, label="PI1")

hist = Float64[]
for U in re_PI2_Nh50.U_List
    dist = maximum(abs.(reshape(U,50,50,51)-re_fixpoint2_Nh50.U))
    append!(hist,dist)
end
plot!(hist[1:end],yaxis=:log, lw=2, label="PI2")
xlabel!("Iteration")
savefig("figures/total_converge_case2_u.pdf")


"""
################ case3 ######################
"""
function non_quad()
    xmin1, xmax1, xmin2, xmax2 = 0, 1, 0, 1
    T = 0.5
    ε = 0.3
    m0(x1,x2) = exp(-10((x1-0.25)^2+(x2-0.25)^2))
    uT(x1,x2) = 1.2*cospi(2*x1) + cospi(2*x2)
    V(x1,x2) = 0.1
    F1(m) = m^0.25
    F2(m) = 0
    update_Q(DU, Du_norm, m) = DU*Du_norm / (m^0.5)
    Nh=50
    problem = MFGTwoDim(xmin1,xmax1,xmin2,xmax2,T,ε,m0,uT,V,F1,F2,update_Q)
    re = solve_mfg_non_quad(problem;method=:PI1,tol=1e-16, node1=Nh,node2=Nh,N=50) 
    return re
end

re_non_quad = non_quad()

# Mass policy iteration contour
contour(re_non_quad.sgrid2,re_non_quad.sgrid1,re_non_quad.M[:,:,1],fill=true,c=:rainbow,size=(580,520))
xlabel!("x2")
ylabel!("x1")
savefig("figures/M_case3_t0.pdf")

contour(re_non_quad.sgrid2,re_non_quad.sgrid1,re_non_quad.M[:,:,17],fill=true,c=:rainbow,size=(580,520))
xlabel!("x2")
ylabel!("x1")
savefig("figures/M_case3_t016.pdf")

contour(re_non_quad.sgrid2,re_non_quad.sgrid1,re_non_quad.M[:,:,34],fill=true,c=:rainbow,size=(580,520))
xlabel!("x2")
ylabel!("x1")
savefig("figures/M_case3_t033.pdf")

contour(re_non_quad.sgrid2,re_non_quad.sgrid1,re_non_quad.M[:,:,end],fill=true,c=:rainbow,size=(580,520))
xlabel!("x2")
ylabel!("x1")
savefig("figures/M_case3_t05.pdf")


# Mass policy iteration heatmap
heatmap(re_non_quad.sgrid2,re_non_quad.sgrid1,re_non_quad.M[:,:,1],c=:rainbow,size=(580,520))
xlabel!("x2")
ylabel!("x1")
savefig("figures/M_case3_t0_heatmap.pdf")

heatmap(re_non_quad.sgrid2,re_non_quad.sgrid1,re_non_quad.M[:,:,17],c=:rainbow,size=(580,520))
xlabel!("x2")
ylabel!("x1")
savefig("figures/M_case3_t016_heatmap.pdf")

heatmap(re_non_quad.sgrid2,re_non_quad.sgrid1,re_non_quad.M[:,:,34],c=:rainbow,size=(580,520))
xlabel!("x2")
ylabel!("x1")
savefig("figures/M_case3_t033_heatmap.pdf")

heatmap(re_non_quad.sgrid2,re_non_quad.sgrid1,re_non_quad.M[:,:,end],c=:rainbow,size=(580,520))
xlabel!("x2")
ylabel!("x1")
savefig("figures/M_case3_t05_heatmap.pdf")


plot(re_non_quad.history.hist_q, yaxis=:log, label="q")
plot!(re_non_quad.history.hist_m, yaxis=:log, label="m")
plot!(re_non_quad.history.hist_u, yaxis=:log, label="u")
plot(re_non_quad.history.residual_HJB, yaxis=:log, label="HJB")
plot!(re_non_quad.history.residual_FP, yaxis=:log, label="FP")

# example of GR's contour
# x1 = 0:0.1:1
# x2 = 1:0.05:2
# V(x1,x2) = 2x1+x2
# v = V.(x1,x2')
# contour(x2,x1,v)
# the Horizontal axis is x2 and vertical axis is x1

"""
########## compute cost ##############
"""

# stop when reach  maxit
function TwoDimTest4_PI_cost(maxit::Int64)
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
    re_algo2 = solve_mfg(problem;method=:PI2,node1=50,node2=50,N=50,maxit=maxit,verbose=false)
    return re_algo2
end

# stop when reach maxit 
function TwoDimTest4_fixpoint_cost(maxit::Int64)
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
    re_algo2 = solve_mfg_fixpoint(problem;method=:FixPoint2,node1=50,node2=50,N=50,maxit=maxit,verbose=false)
    return re_algo2
end

# warm up
@time TwoDimTest4_PI_cost(10)
@time TwoDimTest4_fixpoint_cost(10)

@time for i in 1:5
    TwoDimTest4_PI_cost(29)
end 
@time for i in 1:5
    TwoDimTest4_fixpoint_cost(27)
end

@time for i in 1:5
    TwoDimTest4_PI_cost(25)
end
@time for i in 1:5
    TwoDimTest4_fixpoint_cost(25)
end 

@time for i in 1:5
    TwoDimTest4_PI_cost(80)
end
@time for i in 1:5
    TwoDimTest4_fixpoint_cost(80)
end 

