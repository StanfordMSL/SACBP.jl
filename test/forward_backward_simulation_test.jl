########################################
## File Name: forward_backward_simulation_test.jl
## Author: Haruki Nishimura (hnishimura@stanford.edu)
## Date Created: 2020/05/18
## Description: Test code for src/forward_backward_simulation.jl
########################################

using LinearAlgebra
using Random

@testset "Forward-Backward Simulation Test" begin

# Localization Task
dtc = 0.01;
dto = 0.5;
dtexec = dto*0.8;
Q = Matrix(1.0I, 2, 2);
Cu = Matrix(1.0I, 2, 2);
transPos = TransModel_Pos();
ukfModel = UKFPosRange();
simPosUKF = SimulatePosRangeLocalization2D(transPos,ukfModel,dtc,dto,dtexec,Q,Cu);

rng = MersenneTwister(1234);
p = PhysPos(0.,zeros(2));
bVec = VecBelMvNormal([BelMvNormal(0.,rand(rng, 2),10*Matrix(1.0I, 2, 2)) for ii = 1:100]);
s = AugState(p,bVec);

UArray = [PosControl(round(dtc*t,digits=5),zeros(2)) for t = 0:Int64(2.0/dtc)-1];

Y,S,S_before_update = simulateForward(simPosUKF,s,UArray,MersenneTwister(1234));

@test length(S_before_update) == 4
@test isequal(S_before_update[1].t, 0.49)
@test isequal(S_before_update[1].p.pos, zeros(2))
@test isapprox(S_before_update[1].b[1].params, [0.590845, 0.766797, 10.0, 0.0, 0.0, 10.0], rtol=1e-5)
@test isequal(S_before_update[4].t, 1.99)
@test isequal(S_before_update[4].p.pos, zeros(2))
@test isapprox(S_before_update[4].b[1].params, [0.828148, 1.07501, 11.2299, -0.352906, -0.352906, 11.0388], rtol=1e-5)

CoSArray = simulateBackward(simPosUKF,S,S_before_update,UArray,Y)

@test length(CoSArray) == 201
@test CoSArray[1].t == 0.0
@test isapprox(CoSArray[1].p.pos, [752.687, 833.817], rtol=1e-5)
@test isapprox(CoSArray[1].b[1].params, [-11.2424, -15.1043, 8.97812, -0.435698, -0.782821, 8.91428], rtol=1e-5)

# Manipulation Task
dtc = 0.02;
dto = 0.2;
dtexec = dto*0.8;
QMP = [Matrix(1.0I, 6, 6) zeros(6,5); zeros(5,6) 0.00*Matrix(1.0I, 5, 5)];
RMP = Matrix(1.0I, 9, 9);
CsMP = QMP;
CuMP = Matrix(1.0I, 3, 3);
simMP = SimulateManipulate2D(dtc,dto,dtexec,QMP,RMP,CsMP,CuMP)

xMP = PhysManipulate2D(0.,[0.1,0.2],0.5,[0.3,0.4],0.6,3.,5.,[10.,11.],0.2);
uMP = MControl2D(0.,6.,7.,8.);
bMP = BelMvNormal(xMP.t,vec(xMP),diagm([10.,10.,1.,10.,10.,10.,1.,1.,10.,10.,0.1]));

UArray = [MControl2D(round(dtc*t,digits=5),0.,0.0,0.0) for t = 0:Int64(2.0/dtc)-1];

Y,S,S_before_update = simulateForward(simMP,bMP,UArray,MersenneTwister(13));

@test length(S_before_update) == 10
@test S_before_update[1].t == 0.18
@test isapprox(S_before_update[1].params[1:11], [0.153713, 0.271617, 0.608, 0.296419, 0.395226, 0.6, 3.0, 5.0, 10.0, 11.0, 0.2], rtol=1e-5)
@test S_before_update[10].t == 1.98
@test isapprox(S_before_update[10].params[1:11], [8.91097, -4.67647, 3.30309, 2.53044, -4.53245, 4.0016, 3.21738, 5.0, 10.3436, 14.5485, 0.0583465], rtol=1e-5)

CoSArray = simulateBackward(simMP,S,S_before_update,UArray,Y)

@test length(CoSArray) == 101
@test CoSArray[1].t == 0.0
@test isapprox(CoSArray[1].params[1:11], [-0.606155, 0.0748971, 170.315, 0.574117, -0.324966, -154.575, -0.469705, 0.0, -19.4836, 13.3016, -3.49951], rtol=1e-5)

end
