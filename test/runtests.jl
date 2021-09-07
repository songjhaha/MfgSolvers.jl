using Test
using MfgSolvers

function OneDimTest1()
    xmin = 0.0
    xmax = 1.0
    T = 1
    ε = 0.05
    m0(x) = ((x>=0.375) && (x<=0.625)) ? 1 : 0
    uT(x) = 100min( (x-0.3)^2, (x-0.7)^2 )
    V(x) = 0.1
    F1(m) = 20(1+4m)^1.5
    F2(m) = m
    update_Q(Du,m) = Du / F1(m)
    problem = MFGOneDim(xmin,xmax,T,ε,m0,uT,V,F1,F2,update_Q)
    re = solve_mfg(problem;verbose=false)
    return re.converge
end

function OneDimTest2()
    xmin = 0.0
    xmax = 1.0
    T = 1
    ε = 0.01
    m0(x) = ((x>=0.375) && (x<=0.625)) ? 1 : 0
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
    re = solve_mfg(problem;verbose=false)
    return re.converge
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
    re = solve_mfg(problem;N=100,verbose=false)
    return re.converge
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
    re = solve_mfg(problem;N=50,verbose=false)
    return re.converge
end


function TwoDimTestSimple()
    # simple 2d case without meanfield
    xmin1, xmax1, xmin2, xmax2 = 0, 1, 0, 1
    T = 1
    ε = 1
    m0(x1,x2) = exp(-40((x1-0.5)^2+(x2-0.5)^2))
    uT(x1,x2) = (cospi(2*x1) + cospi(2*x2))
    V(x1,x2) = 0
    F1(m) = 1
    F2(m) = 0
    update_Q(Du,m) = Du / F1(m)
    problem = MFGTwoDim(xmin1,xmax1,xmin2,xmax2,T,ε,m0,uT,V,F1,F2,update_Q)
    re = solve_mfg(problem;N=100,verbose=false)
    return re.converge
end

function OneDimTestPI2()
    xmin = 0.0
    xmax = 1.0
    T = 1
    ε = 0.05
    m0(x) = ((x>=0.375) && (x<=0.625)) ? 1 : 0
    uT(x) = 100min( (x-0.3)^2, (x-0.7)^2 )
    V(x) = 0.1
    F1(m) = 20(1+4m)^1.5
    F2(m) = m
    update_Q(Du,m) = Du / F1(m)
    problem = MFGOneDim(xmin,xmax,T,ε,m0,uT,V,F1,F2,update_Q)
    re = solve_mfg(problem; method=:PI2, verbose=false)
    return re.converge
end

function TwoDimTestPI2()
    # simple 2d case without meanfield
    xmin1, xmax1, xmin2, xmax2 = 0, 1, 0, 1
    T = 1
    ε = 1
    m0(x1,x2) = exp(-40((x1-0.5)^2+(x2-0.5)^2))
    uT(x1,x2) = (cospi(2*x1) + cospi(2*x2))
    V(x1,x2) = 0
    F1(m) = 1
    F2(m) = 0
    update_Q(Du,m) = Du / F1(m)
    problem = MFGTwoDim(xmin1,xmax1,xmin2,xmax2,T,ε,m0,uT,V,F1,F2,update_Q)
    re = solve_mfg(problem; method=:PI2, N=100, verbose=false)
    return re.converge
end


@testset "OneDim" begin
    @test OneDimTest1()
    @test OneDimTest2()
end

@testset "TwoDim" begin
    @test TwoDimTestSimple()
end

@testset "Alogrithm2" begin
    @test OneDimTestPI2()
    @test TwoDimTestPI2()
end
