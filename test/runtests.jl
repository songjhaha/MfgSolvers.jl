using Test
using MfgSolvers

function OneDimTest()    
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
    re = solve_mfg(problem;method=:PI1,node=200,N=200,verbose=true)
    re2 = solve_mfg(problem;method=:PI2,node=200,N=200,verbose=true)
    re_fixp = solve_mfg_fixpoint(problem;method=:FixPoint2,node=200,N=200,verbose=true)
    re_pifp = solve_mfg(problem;method=:PI_FP,node=200,N=200,verbose=true,tol=1e-3)
    return (re.converge,  re2.converge, re_fixp.converge, re_pifp.converge)
end

function TwoDimTest()
    xmin1, xmax1, xmin2, xmax2 = 0, 1, 0, 1
    T = 0.5
    ε = 0.5
    m0(x1,x2) = exp(-10((x1-0.25)^2+(x2-0.25)^2))
    uT(x1,x2) = 1.2*cospi(2*x1) + cospi(2*x2)
    V(x1,x2) = 0.1
    F1(m) = m^0.5
    F2(m) = 0
    update_Q(Du,m) = Du / F1(m)
    problem = MFGTwoDim(xmin1,xmax1,xmin2,xmax2,T,ε,m0,uT,V,F1,F2,update_Q)
    re = solve_mfg(problem;method=:PI1,node1=50,node2=50,N=50) 
    re2 = solve_mfg(problem;method=:PI2,node1=50,node2=50,N=50)
    re_fixp = solve_mfg_fixpoint(problem;method=:FixPoint2,node1=50,node2=50,N=50)  
    return (re.converge,  re2.converge, re_fixp.converge)
end

function TwoDim_non_quad_Test()
    xmin1, xmax1, xmin2, xmax2 = 0, 1, 0, 1
    T = 0.5
    ε = 0.5
    m0(x1,x2) = exp(-10((x1-0.25)^2+(x2-0.25)^2))
    uT(x1,x2) = 1.2*cospi(2*x1) + cospi(2*x2)
    V(x1,x2) = 0.1
    F1(m) = m^0.25
    F2(m) = 0
    update_Q(DU, Du_norm, m) = DU*Du_norm / (m^0.5)
    problem = MFGTwoDim(xmin1,xmax1,xmin2,xmax2,T,ε,m0,uT,V,F1,F2,update_Q)
    re = solve_mfg_non_quad(problem;method=:PI1, node1=50,node2=50,N=50) 
    re2 = solve_mfg_non_quad(problem;method=:PI2, node1=50,node2=50,N=50) 
    return (re.converge,  re2.converge)
end


@testset "OneDim" begin
    converge1, converge2, converge_fixp, converge_pifp = OneDimTest() 
    @test converge1
    @test converge2
    @test converge_fixp
    @test converge_pifp
end

@testset "TwoDim" begin
    converge1, converge2, converge_fixp = TwoDimTest() 
    @test converge1
    @test converge2
    @test converge_fixp
end

@testset "NonQuad" begin
    converge1, converge2 = TwoDim_non_quad_Test() 
    @test converge1
    @test converge2
end

