########################################
## File Name: sac_control_test.jl
## Author: Haruki Nishimura (hnishimura@stanford.edu)
## Date Created: 2020/05/20
## Description: Test code for src/sac_controller.jl
########################################

using LinearAlgebra
using Random

@testset "SAC Control Test" begin

# Localization Task
dtc = 0.01;
dto = 0.2;
dtexec = dto*0.8;
Q = Matrix(1.0I, 2, 2);
Cu = Matrix(1.0I, 2, 2) #+ [0. 0.01; 0.01 0.];
transPos = TransModel_Pos();
ukfModel = UKFPosRange();
simPosUKF = SimulatePosRangeLocalization2D(transPos,ukfModel,dtc,dto,dtexec,Q,Cu);

rng = MersenneTwister(1234)
p = PhysPos(0.,zeros(2));
bVec = VecBelMvNormal([BelMvNormal(0.,rand(rng, 2),10*Matrix(1.0I, 2, 2)) for ii = 1:100]);
s = AugState(p,bVec);

UArray = [PosControl(round(dtc*t,digits=5),zeros(2)) for t = 0:Int64(2.0/dtc)-1];

coeffUKF = getControlCoeffs(simPosUKF,s,UArray,MersenneTwister(1234))

@test length(coeffUKF) == 201
@test isapprox(coeffUKF[1], [2298.44, 2561.91], rtol=1e-5)
@test isapprox(coeffUKF[end-1], [270.112, 308.623], rtol=1e-5)
@test coeffUKF[end] == [0.0, 0.0]

UCMatPosUKF = controlCoeffsExpected(simPosUKF,s,UArray,100,MersenneTwister(1234))

@test size(UCMatPosUKF) == (2, 201)
@test isapprox(UCMatPosUKF[1, 1], 2441.96, rtol=1e-5)
@test isapprox(UCMatPosUKF[2, 1], 2505.37, rtol=1e-5)
@test UCMatPosUKF[1, end] == 0.0
@test UCMatPosUKF[2, end] == 0.0
@test isapprox(UCMatPosUKF[1, end-1], 270.42, rtol=1e-5)
@test isapprox(UCMatPosUKF[2, end-1], 285.107, rtol=1e-5)

UOptArray, CostArray = optControlSchedule(simPosUKF, UArray, UCMatPosUKF, [-1., -1.], [1., 1.]);

@test all([u.vel == [-1., -1.] for u in UOptArray])
@test isapprox(CostArray[1], -4946.33, rtol=1e-5)
@test isapprox(CostArray[end], -554.527, rtol=1e-5)

UOpt,tcalc = determineControlTime(0.1,simPosUKF,UOptArray,CostArray)

@test UOpt.t == 0.27
@test UOpt.vel == [-1., -1.]

UArrayNew, act_time_init, act_time_final, tcalc = sacControlUpdate(simPosUKF,s,UArray,[-1.,-1.],[1.,1.],10,MersenneTwister(1234))

@test act_time_init == 0.06
@test act_time_final == 0.21
@test UArrayNew[7].vel == [-1.0, -1.0]

# Manipulation Task
dtc = 0.02;
dto = 0.2;
dtexec = dto*0.8;
QMP = [Matrix(1.0I, 6, 6) zeros(6,5); zeros(5,6) 0.00*Matrix(1.0I, 5, 5)];
RMP = Matrix(1.0I, 9, 9);
CsMP = 10*QMP;
CuMP = 10*Matrix(1.0I, 3, 3) #+ [0. 0. 0.01; 0. 0. 0.; 0.01 0. 0.];
simMP = SimulateManipulate2D(dtc,dto,dtexec,QMP,RMP,CsMP,CuMP)

xMP = PhysManipulate2D(0.,[0.1,0.2],0.5,[0.3,0.4],0.6,3.,5.,[10.,11.],0.2);
uMP = MControl2D(0.,6.,7.,8.);
bMP = BelMvNormal(xMP.t,vec(xMP),diagm([10.,10.,10.,10.,10.,0.1,1.,1.,10.,10.,0.1]));

UArray = [MControl2D(round(dtc*t,digits=5),0.,0.0,0.0) for t = 0:Int64(2.0/dtc)-1];

cMP = getControlCoeffs(simMP,bMP,UArray,MersenneTwister(1234))

@test length(cMP) == 101
@test isapprox(cMP[1], [1929.07, -483.397, -134.458], rtol=1e-5)
@test isapprox(cMP[end], [-31.1805, 3.37386, -1.69659], rtol=1e-5)

cMPExpected = controlCoeffsExpected(simMP,bMP,UArray,10,MersenneTwister(1))

@test size(cMPExpected) == (3, 101)
@test isapprox(cMPExpected[1, 1], 6532.73, rtol=1e-5)
@test isapprox(cMPExpected[2, 1], -1597.58, rtol=1e-5)
@test isapprox(cMPExpected[3, 1], -453.342, rtol=1e-5)
@test isapprox(cMPExpected[1, end], 31.7051, rtol=1e-5)
@test isapprox(cMPExpected[2, end], 26.0962, rtol=1e-5)
@test isapprox(cMPExpected[3, end], 3.52962, rtol=1e-5)
@test isapprox(cMPExpected[1, end-1], 38.9432, rtol=1e-5)
@test isapprox(cMPExpected[2, end-1], 242.227, rtol=1e-5)
@test isapprox(cMPExpected[3, end-1], -8.07914, rtol=1e-5)

UOptArray, CostArray = optControlSchedule(simMP,UArray,cMPExpected, [-1.,-1.,-1.],[1.,1.,1])

@test length(UOptArray) == 100
@test isapprox(vec(UOptArray[1]), [-1.0, 1.0, 1.0], rtol=1e-5)
@test isapprox(vec(UOptArray[end]), [-1.0, -1.0, 0.807914], rtol=1e-5)

@test isapprox(CostArray[1], -8568.65, rtol=1e-5)
@test isapprox(CostArray[end], -274.434, rtol=1e-5)

UOpt,tcalc = determineControlTime(0.1,simMP,UOptArray,CostArray)

@test UOpt.t == 0.28
@test isapprox(vec(UOpt), [-1.0, -1.0, -0.9884261676986377], rtol=1e-5)

UArrayNew, act_time_init, act_time_final, tcalc = sacControlUpdate(simMP,bMP,UArray,[-1.,-1.,-1.],[1.,1.,1.],1,MersenneTwister(1234))

@test act_time_init == 0.06
@test act_time_final == 0.2

@test isapprox(vec(UArrayNew[4]),  [-0.0158972, 0.544518, 0.0988366], rtol=1e-5)

end
