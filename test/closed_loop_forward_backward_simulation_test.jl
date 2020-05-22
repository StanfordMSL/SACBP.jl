########################################
## File Name: forward_backward_simulation_test.jl
## Author: Haruki Nishimura (hnishimura@stanford.edu)
## Date Created: 2020/05/19
## Description: Test code for src/forward_backward_simulation.jl with
## closed-loop nominal policies.
########################################

using LinearAlgebra
using Random

@testset "Closed-Loop Forward-Backward Simulation Test" begin

# Localization Task
dtc = 0.01;
dto = 0.2;
dtexec = dto*0.4;
Q = Matrix(1.0I, 2, 2);
Cu = Matrix(1.0I, 2, 2);
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

Y,S,S_before_update,UArray = simulateForward(simPosUKF,gradientPolicy,s,u_param_min_2D,u_param_max_2D,UArray,MersenneTwister(1234));

@test length(UArray) == 200
@test isapprox(UArray[1].vel, [-0.704459, -0.709745], rtol=1e-5)
@test isapprox(UArray[end].vel, [-0.526402, -0.850236], rtol=1e-5)

@test length(S_before_update) == 10
@test S_before_update[1].t == 0.19
@test isapprox(S_before_update[1].p.pos, [-0.133847, -0.134851], rtol=1e-5)
@test isapprox(S_before_update[1].b[1].params, [0.590845, 0.766797, 10.0, 0.0, 0.0, 10.0], rtol=1e-5)
@test S_before_update[10].t == 1.99
@test isapprox(S_before_update[10].p.pos, [-1.27097, -1.52526], rtol=1e-5)
@test isapprox(S_before_update[10].b[1].params, [0.0249566, 0.0788592, 10.0039, -2.21546, -2.21546, 9.06684], rtol=1e-5)

CoSArray = simulateBackward(simPosUKF, S, S_before_update, UArray, Y)

@test length(CoSArray) == 201
@test CoSArray[1].t == 0.0
@test isapprox(CoSArray[1].p.pos, [1310.62, 2216.47], rtol=1e-5)
@test isapprox(CoSArray[1].b[1].params, [-19.9427, -30.5194, 10.22, -1.94483, -2.56533, 10.45], rtol=1e-5)

# Manipulation Task
dtc = 0.02;
dto = 0.2;
dtexec = dto*0.8;
QMP = [Matrix(1.0I, 6, 6) zeros(6,5); zeros(5,6) 0.00*Matrix(1.0I, 5, 5)];
RMP = Matrix(1.0I, 9, 9);
CsMP = QMP;
CuMP = Matrix(1.0I, 3, 3);
simMP = SimulateManipulate2D(dtc,dto,dtexec,QMP,RMP,CsMP,CuMP)
pcontrolPolicy = ManipulatePositionControlPolicy();

xMP = PhysManipulate2D(0.,[0.1,0.2],0.5,[0.3,0.4],0.6,3.,5.,[10.,11.],0.2);
uMP = MControl2D(0.,6.,7.,8.);
bMP = BelMvNormal(xMP.t,vec(xMP),diagm([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]));
u_param_min_3D = [-3.,-3.,-3.];
u_param_max_3D = [3.,3.,3.];
posGain = 1.0;
rotGain = 0.2;

UArray = [MControl2D(round(dtc*t,digits=5),0.,0.0,0.0) for t = 0:Int64(2.0/dtc)-1];
Y,S,S_before_update,UArray = simulateForward(simMP,pcontrolPolicy,bMP,u_param_min_3D,u_param_max_3D,posGain,rotGain,UArray,MersenneTwister(123));

@test length(UArray) == 100
@test isapprox(vec(UArray[1]), [-0.1, -0.2, 0.528319], rtol=1e-5)
@test isapprox(vec(UArray[end]), [-0.253098, -0.637423, -0.598584], rtol=1e-5)

@test length(S_before_update) == 10
@test S_before_update[1].t == 0.18
@test isapprox(S_before_update[1].params[1:11], [0.153234, 0.27066, 0.611928, 0.290451, 0.383289, 0.651479, 3.0, 5.0, 10.0, 11.0, 0.2], rtol=1e-5)
@test S_before_update[10].t == 1.98
@test isapprox(S_before_update[10].params[1:11], [0.300082, 0.74635, 0.478632, 0.250788, 0.579737, 3.4435, 2.99207, 5.34697, 10.5676, 11.1221, 0.205542], rtol=1e-5)

CoSArray = simulateBackward(simMP,S,S_before_update,UArray,Y)

@test length(CoSArray) == 101
@test CoSArray[1].t == 0.0
@test isapprox(CoSArray[1].params[1:11], [0.884668, -0.817838, -16.6739, 1.02655, 0.0991195, -6.46987, -0.171831, -12.4607, 4.27364, 3.02749, -0.656432], rtol=1e-5)

end
