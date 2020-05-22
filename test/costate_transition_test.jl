########################################
## File Name: costate_transition_test.jl
## Author: Haruki Nishimura (hnishimura@stanford.edu)
## Date Created: 2020/05/14
## Description: Test code for src/costate_transition_models.jl
########################################

using LinearAlgebra

@testset "Costate Transition Test" begin

# Localization Task
ukfModel = UKFPosRange();
transPos = TransModel_Pos();
bPos = BelMvNormal(0.,zeros(2),100*Matrix(1.0I, 2, 2));
bVecPos = [bPos for ii = 1:100];
p_robot = PhysPos(0.,[5.,7.]);
sV = AugState(p_robot,bVecPos);
costModel = CostPosRangeLocalization();
cosP = PhysPos(0.,termCost_grad_p(costModel,sV))
cosBV = [BelMvNormal(0.0,params) for params in termCost_grad_b(costModel,sV)];
coS = AugState(cosP,cosBV);

cosP_new = cotrans(transPos,costModel,coS,sV,PosControl(0.,zeros(2)),0.01)
@test cosP_new.t == -0.01
@test cosP_new.p.pos == zeros(2)
@test all([isapprox(cosP_new.b[ii].params, [0.0, 0.0, 8.53973, 0.0, 0.0, 8.53973], atol=1e-4) for ii = 1:100])

cosP_new_ukf = cotrans(ukfModel,costModel,coS,sV,ones(100),0.2,Matrix(1.0I, 2, 2))
@test cosP_new_ukf.t == 0.0
@test isapprox(cosP_new_ukf.p.pos, [-2002.46, -2992.51], rtol=1e-5)
@test all([isapprox(cosP_new_ukf.b[ii].params, [20.0246, 29.9251, 8.87919, 0.0, -2.44889, 8.39393], rtol=1e-5) for ii = 1:100])

# Manipulation Task
xMP = PhysManipulate2D(0.,[0.1,0.2],0.5,[0.3,0.4],0.6,3.,5.,[10.,11.],0.7);
uMP = MControl2D(0.,6.,7.,8.);
bMP = BelMvNormal(xMP.t,vec(xMP),100*Matrix(1.0I, 11, 11));

QMP = [Matrix(1.0I, 6, 6) zeros(6,5); zeros(5,6) 0.01*Matrix(1.0I, 5, 5)];
RMP = Matrix(1.0I, 9, 9);
CsMP = QMP;
CuMP = Matrix(1.0I, 3, 3);

costModel = CostManipulate2D()
cob = BelMvNormal(bMP.t,termCost_grad(costModel,bMP,CsMP))
transModelC = CPredictManipulate2D();

cob_new = cotrans(transModelC,costModel,cob,bMP,uMP,0.01,QMP,CsMP)
@test cob_new.t == -0.01
@test isapprox(cob_new.params[1:11], [0.101, 0.202, -2.81458, 0.3033, 0.405067, 0.579584, 0.180939, 0.063501, 0.10492, 0.100754, -0.66043], rtol=1e-4)

transModelD = DUpdateManipulate2D();
cob_new_u = cotrans(transModelD,costModel,cob,bMP,uMP,ones(9),RMP)
@test cob_new_u.t == 0.0
@test isapprox(cob_new_u.params[1:11], [0.0226903, -0.00757826, -3.87298, 0.164995, -0.0386729, 0.118948, -0.604347, 0.463712, 0.924217, 0.857975, 0.186299], rtol=1e-5)

end
