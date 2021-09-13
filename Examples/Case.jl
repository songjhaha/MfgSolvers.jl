using MfgSolvers
using LaTeXStrings
using Plots

function OneDimTest1()    
    xmin = 0.0    
    xmax = 1.0    
    T = 1    
    ε = 0.05    
    m0(x) = ((x>=0.375) && (x<=0.625)) ? 4 : 0
    uT(x) = 100min( (x-0.3)^2, (x-0.7)^2 )
    V(x) = 0.1
    F1(m) = 20(1+4m)^1.5
    F2(m) = m
    update_Q(Du,m) = Du / F1(m)
    problem = MFGOneDim(xmin,xmax,T,ε,m0,uT,V,F1,F2,update_Q)
    re = solve_mfg(problem;node=200,N=200,verbose=true) # iter 10, Q diff 7e-9
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
    re = solve_mfg(problem;node1=Nh,node2=Nh,N=50) # 50s # iter 41
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
    re_algo2 = solve_mfg_fixpoint(problem;method=:PI2,node1=Nh,node2=Nh,N=50,verbose=true)
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

# case1
begin
    re_OneDim = OneDimTest1()
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
end


re_PI1 = TwoDimTest4_PI1(50)
re_PI2_Nh50 = TwoDimTest4_PI2(50)

contour(re_PI2_Nh50.sgrid2,re_PI2_Nh50.sgrid1,re_PI2_Nh50.M[:,:,1],c=:rainbow,title="M,t=0",size=(450,400))
savefig("M_case2_t0.pdf")
contour(re_PI2_Nh50.sgrid2,re_PI2_Nh50.sgrid1,re_PI2_Nh50.M[:,:,17],c=:rainbow,title="M,t=0.16",size=(450,400))
savefig("M_case2_t016.pdf")
contour(re_PI2_Nh50.sgrid2,re_PI2_Nh50.sgrid1,re_PI2_Nh50.M[:,:,34],c=:rainbow,title="M,t=0.33",size=(450,400))
savefig("M_case2_t033.pdf")
contour(re_PI2_Nh50.sgrid2,re_PI2_Nh50.sgrid1,re_PI2_Nh50.M[:,:,end],levels=0:0.5:4.5,c=:rainbow,title="M,t=0.5",size=(450,400))
savefig("M_case2_t05.pdf")

re_fixpoint2_Nh50 = TwoDimTest4_fixpoint(50)

contour(re_fixpoint2_Nh50.sgrid2,re_fixpoint2_Nh50.sgrid1,re_fixpoint2_Nh50.M[:,:,1],c=:rainbow,title="M,t=0",size=(450,400))
# savefig("M_case2_t0.pdf")
contour(re_fixpoint2_Nh50.sgrid2,re_fixpoint2_Nh50.sgrid1,re_fixpoint2_Nh50.M[:,:,17],c=:rainbow,title="M,t=0.16",size=(450,400))
# savefig("M_case2_t016.pdf")
contour(re_fixpoint2_Nh50.sgrid2,re_fixpoint2_Nh50.sgrid1,re_fixpoint2_Nh50.M[:,:,34],c=:rainbow,title="M,t=0.33",size=(450,400))
# savefig("M_case2_t033.pdf")
contour(re_fixpoint2_Nh50.sgrid2,re_fixpoint2_Nh50.sgrid1,re_fixpoint2_Nh50.M[:,:,end],levels=0:0.5:4.5,c=:rainbow,title="M,t=0.5",size=(450,400))
# savefig("M_case2_t05.pdf")


