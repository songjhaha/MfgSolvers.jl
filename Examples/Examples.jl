using MfgSolvers

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

function OneDimTest2()
    xmin = 0.0
    xmax = 1.0
    T = 1
    ε = 0.01
    m0(x) = ((x>=0.375) && (x<=0.625)) ? 4 : 0
    uT(x) = 150min( (x-0.3)^2, (x-0.7)^2 )
    V(x) = 0.1
    F1(m) = 20(1+4m)^0.75
    F2(m) = m
    function update_Q(Du, m)
        out = Du / F1(m)
        if out > 0.1
            out = 0.1
        elseif out < -0.1
            out = -0.1
        end
        return out
    end
    problem = MFGOneDim(xmin,xmax,T,ε,m0,uT,V,F1,F2,update_Q)
    re = solve_mfg(problem;node=200,N=200,verbose=true) #iter 15, Q diff 7e-9
    return re
end

function TwoDimTest3()
    xmin1, xmax1, xmin2, xmax2 = 0, 1, 0, 1
    T = 1
    ε = 0.8
    m0(x1,x2) = exp(-40((x1-0.5)^2+(x2-0.5)^2))
    uT(x1,x2) = 5*(cospi(2*x1) + cospi(2*x2))
    V(x1,x2) = -(sinpi(2x1)+sinpi(2x2)+cospi(4x1))
    F1(m) = (1+m)^0.75
    F2(m) = m
    update_Q(Du,m) = Du / F1(m)
    problem = MFGTwoDim(xmin1,xmax1,xmin2,xmax2,T,ε,m0,uT,V,F1,F2,update_Q)
    re = solve_mfg(problem;node1=50,node2=50,N=100,verbose=true) # iter 58
    return re 
end

function TwoDimTest4()
    xmin1, xmax1, xmin2, xmax2 = 0, 1, 0, 1
    T = 0.5
    ε = 0.7
    m0(x1,x2) = exp(-10((x1-0.25)^2+(x2-0.25)^2))
    uT(x1,x2) = cospi(2*x1) + cospi(2*x2)
    V(x1,x2) = 0.1
    F1(m) = m^0.5
    F2(m) = 0
    update_Q(Du,m) = Du / F1(m)
    problem = MFGTwoDim(xmin1,xmax1,xmin2,xmax2,T,ε,m0,uT,V,F1,F2,update_Q)
    re = solve_mfg(problem;N=50) # 50s # iter 41
    return re
end

function TwoDimTest4_PI2()
    xmin1, xmax1, xmin2, xmax2 = 0, 1, 0, 1
    T = 0.5
    ε = 0.7
    m0(x1,x2) = exp(-10((x1-0.25)^2+(x2-0.25)^2))
    uT(x1,x2) = cospi(2*x1) + cospi(2*x2)
    V(x1,x2) = 0.1
    F1(m) = m^0.5
    F2(m) = 0
    update_Q(Du,m) = Du / F1(m)
    problem = MFGTwoDim(xmin1,xmax1,xmin2,xmax2,T,ε,m0,uT,V,F1,F2,update_Q) 
    re_algo2 = solve_mfg(problem;method=:PI2,node1=50,node2=50,N=50,verbose=true) # iter 23
    return re_algo2
end

function TwoDimTest5()
    xmin1, xmax1, xmin2, xmax2 = 0, 1, 0, 1
    T = 0.5
    ε = 0.2
    m0(x1,x2) = exp(-10((x1-0.25)^2+(x2-0.25)^2))
    uT(x1,x2) = cospi(2*x1) + cospi(2*x2)
    V(x1,x2) = 0.1
    F1(m) = (1+m)^0.5
    F2(m) = 0
    update_Q(Du,m) = Du / F1(m)
    mfg_problem = MFGTwoDim(xmin1,xmax1,xmin2,xmax2,T,ε,m0,uT,V,F1,F2,update_Q)
    re_mfg = solve_mfg(mfg_problem;node1=50,node2=50,N=50,verbose=true)
    Phi(m) = (1+m)^0.5 + 0.5*m*(1+m)^(-0.5)
    mftc_problem = MFGTwoDim(xmin1,xmax1,xmin2,xmax2,T,ε,m0,uT,V,Phi,F2,update_Q)
    re_mftc = solve_mfg(mftc_problem;node1=50,node2=50,N=50,verbose=true)
    return re_mfg, re_mftc 
end

function TwoDimTestAnother()
    xmin1, xmax1, xmin2, xmax2 = 0, 1, 0, 1
    T = 1
    ε = 0.3
    m0(x1,x2) = exp(-40((x1-0.5)^2+(x2-0.5)^2))
    uT(x1,x2) = -exp(-40((x1-0.5)^2+(x2-0.5)^2))
    V(x1,x2) = -abs(sinpi(2x1)*sinpi(2x2))
    F1(m) = 1
    F2(m) = m^2
    update_Q(Du,m) = Du
    problem = MFGTwoDim(xmin1,xmax1,xmin2,xmax2,T,ε,m0,uT,V,F1,F2,update_Q)
    re = solve_mfg(problem;node1=50,node2=50,N=100,verbose=true) # 120s, iter 65
    return re 
end

function TwoDimTest4_fixpoint()
    xmin1, xmax1, xmin2, xmax2 = 0, 1, 0, 1
    T = 0.5
    ε = 0.7
    m0(x1,x2) = exp(-10((x1-0.25)^2+(x2-0.25)^2))
    uT(x1,x2) = cospi(2*x1) + cospi(2*x2)
    V(x1,x2) = 0.1
    F1(m) = m^0.5
    F2(m) = 0
    update_Q(Du,m) = Du / F1(m)
    problem = MFGTwoDim(xmin1,xmax1,xmin2,xmax2,T,ε,m0,uT,V,F1,F2,update_Q) 
    re_algo2 = solve_mfg_fixpoint(problem;method=:FixPoint2,node1=50,node2=50,N=50,maxit=100,verbose=true) # iter 23
    return re_algo2
end

re = TwoDimTest4_fixpoint()

using Plots
contour(re.sgrid2,re.sgrid1,re.M[:,:,1],c=:rainbow,title="M,t=0",size=(450,400))
savefig("M_2d_case4_0.pdf")
contour(re.sgrid2,re.sgrid1,re.M[:,:,17],c=:rainbow,title="M,t=0.16",size=(450,400))
savefig("M_2d_case4_016.pdf")
contour(re.sgrid2,re.sgrid1,re.M[:,:,34],c=:rainbow,title="M,t=0.33",size=(450,400))
savefig("M_2d_case4_033.pdf")
contour(re.sgrid2,re.sgrid1,re.M[:,:,end],levels=0:0.2:3,c=:rainbow,title="M,t=0.5",size=(450,400))
savefig("M_2d_case4_05.pdf")

function TwoDimTest4_fixpoint2()
    xmin1, xmax1, xmin2, xmax2 = 0, 1, 0, 1
    T = 0.5
    ε = 0.7
    m0(x1,x2) = exp(-10((x1-0.25)^2+(x2-0.25)^2))
    uT(x1,x2) = cospi(2*x1) + cospi(2*x2)
    V(x1,x2) = 0.1
    F1(m) = m^0.5
    F2(m) = 0
    update_Q(Du,m) = Du / F1(m)
    problem = MFGTwoDim(xmin1,xmax1,xmin2,xmax2,T,ε,m0,uT,V,F1,F2,update_Q) 
    re_algo2 = solve_mfg_fixpoint(problem;method=:PI2,node1=50,node2=50,N=50,verbose=true) # iter 23
    return re_algo2
end

re2 = TwoDimTest4_fixpoint2()

contour(re2.sgrid2,re2.sgrid1,re2.M[:,:,1],c=:rainbow,title="M,t=0",size=(450,400))
# savefig("M_2d_case4_0.pdf")
contour(re2.sgrid2,re2.sgrid1,re2.M[:,:,17],c=:rainbow,title="M,t=0.16",size=(450,400))
# savefig("M_2d_case4_016.pdf")
contour(re2.sgrid2,re2.sgrid1,re2.M[:,:,34],c=:rainbow,title="M,t=0.33",size=(450,400))
# savefig("M_2d_case4_033.pdf")
contour(re2.sgrid2,re2.sgrid1,re2.M[:,:,end],levels=0:0.2:3,c=:rainbow,title="M,t=0.5",size=(450,400))
# savefig("M_2d_case4_05.pdf")