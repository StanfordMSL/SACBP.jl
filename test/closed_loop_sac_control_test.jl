########################################
## File Name: closed_loop_sac_control_test.jl
## Author: Haruki Nishimura (hnishimura@stanford.edu)
## Date Created: 2020/05/20
## Description: Test code for src/sac_controller.jl with
## closed-loop nominal policies.
########################################

using LinearAlgebra
using Random

@testset "Closed-Loop SAC Control Test" begin

# Localization Task
dtc = 0.01;
dto = 0.2;
dtexec = dto*0.8;
Q = Matrix(1.0I, 2, 2);
Cu = Matrix(1.0I, 2, 2) #+ [0. 0.01; 0.01 0.];
transPos = TransModel_Pos();
ukfModel = UKFPosRange();
simPosUKF = SimulatePosRangeLocalization2D(transPos,ukfModel,dtc,dto,dtexec,Q,Cu);
gradientPolicy = GradientMultiTargetLocalizationPolicy();
u_param_min_2D = [-2.,-2.];
u_param_max_2D = [2.,2.];

rng = MersenneTwister(1234)
p = PhysPos(0.,zeros(2));
bVec = VecBelMvNormal([BelMvNormal(0.,rand(rng, 2),10*Matrix(1.0I, 2, 2)) for ii = 1:100]);
s = AugState(p,bVec);

UArray = [PosControl(round(dtc*t,digits=5),zeros(2)) for t = 0:Int64(2.0/dtc)-1];

CoeffArray,NominalControlCostArray = getControlCoeffs(simPosUKF,gradientPolicy,s,u_param_min_2D,u_param_max_2D,UArray,MersenneTwister(1234))

@test length(CoeffArray) == 201
@test isapprox(CoeffArray[1], [1310.62, 2216.47], rtol=1e-5)
@test CoeffArray[end] == [0.0, 0.0]
@test isapprox(CoeffArray[end-1], [343.952, 574.194], rtol=1e-5)

@test length(NominalControlCostArray) == 200
@test isapprox(NominalControlCostArray[1], -2495.9, rtol=1e-5)
@test isapprox(NominalControlCostArray[end], -668.758, rtol=1e-5)

UCMatPosUKF_1,UCMatPosUKF_2 = controlCoeffsExpected(simPosUKF,gradientPolicy,s,u_param_min_2D,u_param_max_2D,UArray,10,MersenneTwister(1234))

@test size(UCMatPosUKF_1) == (2, 201)
@test isapprox(UCMatPosUKF_1[1, 1], 1460.23, rtol=1e-5)
@test isapprox(UCMatPosUKF_1[2, 1], 2182.91, rtol=1e-5)
@test UCMatPosUKF_1[1, end] == 0.0
@test UCMatPosUKF_1[2, end] == 0.0
@test isapprox(UCMatPosUKF_1[1, end-1], 383.388, rtol=1e-5)
@test isapprox(UCMatPosUKF_1[2, end-1], 580.1, rtol=1e-5)

@test length(UCMatPosUKF_2) == 200
@test isapprox(UCMatPosUKF_2[1], -2577.48, rtol=1e-5)
@test isapprox(UCMatPosUKF_2[end], -695.22, rtol=1e-5)

UOptArray,CostArray = optControlSchedule(simPosUKF,UArray,UCMatPosUKF_1,u_param_min_2D,u_param_max_2D)

@test all([vec(u) == [-2.0, -2.0] for u in UOptArray])

@test length(CostArray) == 200
@test isapprox(CostArray[1], -7282.29, rtol=1e-5)
@test isapprox(CostArray[end], -1922.98, rtol=1e-5)

UOpt,tcalc = determineControlTime(0.1,simPosUKF,UOptArray,CostArray,UCMatPosUKF_2)

@test UOpt.t == 0.27
@test vec(UOpt) == [-2.0, -2.0]

U,time_init,time_final,tcalc = sacControlUpdate(simPosUKF,gradientPolicy,s,UArray,u_param_min_2D,u_param_max_2D,10,MersenneTwister(1234))

@test time_init == 0.06
@test time_final == 0.21
@test length(U) == 200
@test vec(U[7]) == [-2.0, -2.0]

# Mainpulation Task
dtc = 0.02;
dto = 0.2;
dtexec = dto*0.8;
QMP = [Matrix(1.0I, 6, 6) zeros(6,5); zeros(5,6) 0.00*Matrix(1.0I, 5, 5)];
RMP = Matrix(1.0I, 9, 9);
CsMP = 10*QMP;
CuMP = 10*Matrix(1.0I, 3, 3) #+ [0. 0. 0.01; 0. 0. 0.; 0.01 0. 0.];
simMP = SimulateManipulate2D(dtc,dto,dtexec,QMP,RMP,CsMP,CuMP)
pcontrolPolicy = ManipulatePositionControlPolicy();

xMP = PhysManipulate2D(0.,[0.1,0.2],0.5,[0.3,0.4],0.6,3.,5.,[10.,11.],0.2);
uMP = MControl2D(0.,6.,7.,8.);
bMP = BelMvNormal(xMP.t,vec(xMP),diagm([10.,10.,10.,10.,10.,0.1,1.,1.,10.,10.,0.1]));
u_param_min_3D = [-3.,-3.,-3.];
u_param_max_3D = [3.,3.,3.];
posGain = 1.0;
rotGain = 0.2;

UArray = [MControl2D(round(dtc*t,digits=5),0.,0.0,0.0) for t = 0:Int64(2.0/dtc)-1];

cMP_1,cMP_2 = controlCoeffsExpected(simMP,pcontrolPolicy,bMP,u_param_min_3D,u_param_max_3D,posGain,rotGain,UArray,10,MersenneTwister(1))

@test size(cMP_1) == (3, 101)
@test isapprox(cMP_1[1, 1], -9.32791e6, rtol=1e-5)
@test isapprox(cMP_1[2, 1], 2.5791e6, rtol=1e-5)
@test isapprox(cMP_1[3, 1], 6.37581e5, rtol=1e-5)
@test isapprox(cMP_1[1, end], -101.311, rtol=1e-5)
@test isapprox(cMP_1[2, end], -102.395, rtol=1e-5)
@test isapprox(cMP_1[3, end], -16.9331, rtol=1e-5)

@test length(cMP_2) == 100
@test isapprox(cMP_2[1], 7.53819e5, rtol=1e-5)
@test isapprox(cMP_2[end], 5152.82, rtol=1e-5)

UOptArray,CostArray = optControlSchedule(simMP,UArray,cMP_1,u_param_min_3D,u_param_max_3D)

@test length(UOptArray) == 100
@test isapprox(vec(UOptArray[1]), [3.0, -3.0, -3.0])
@test isapprox(vec(UOptArray[end]), [-3.0, 3.0, 3.0])

@test length(CostArray) == 100
@test isapprox(CostArray[1], -3.76336e7, rtol=1e-5)
@test isapprox(CostArray[end], -14172.4, rtol=1e-5)

UOpt,tcalc = determineControlTime(0.1,simMP,UOptArray,CostArray,cMP_2)

@test UOpt.t == 0.28
@test vec(UOpt) == [-3.0, -3.0, 3.0]

UArrayNew, act_time_init, act_time_final, tcalc = sacControlUpdate(simMP,pcontrolPolicy,bMP,UArray,u_param_min_3D,u_param_max_3D,posGain,rotGain,10,MersenneTwister(1234))

@test length(UArrayNew) == 100
@test act_time_init == 0.06
@test act_time_final == 0.2
@test vec(UArrayNew[4]) == [3.0, -3.0, -3.0]

end
