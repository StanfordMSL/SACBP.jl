########################################
## File Name: nominal_policies_test.jl
## Author: Haruki Nishimura (hnishimura@stanford.edu)
## Date Created: 2020/05/14
## Description: Test code for src/nominal_policies.jl
########################################

using LinearAlgebra

@testset "Nominal Policies Test" begin

# Localization Task
dtc = 0.01;
dto = 0.5;
dtexec = dto*0.8;
Q = Matrix(1.0I, 2, 2);
Cu = Matrix(1.0I, 2, 2);
transPos = TransModel_Pos();
ukfModel = UKFPosRange();
gradientPolicy = GradientMultiTargetLocalizationPolicy();

bPos = BelMvNormal(0.,zeros(2),100*Matrix(1.0I, 2, 2));
bVecPos = [bPos for ii = 1:100];
p_robot = PhysPos(0.,[5.,7.]);
sV = AugState(p_robot,bVecPos);
u_param_min_2D = [-2.,-2.];
u_param_max_2D = [2.,2.];

uPos = control_nominal(gradientPolicy,ukfModel,sV,u_param_min_2D,u_param_max_2D,dto,Q)
@test uPos.t == 0.0
@test isapprox(uPos.vel, [0.556133, 0.831093], rtol=1e-4)

# Manipulation Task
dtc = 0.02;
dto = 0.2;
dtexec = dto*0.8;
QMP = [Matrix(1.0I, 6, 6) zeros(6,5); zeros(5,6) 0.00*Matrix(1.0I, 5, 5)];
RMP = Matrix(1.0I, 9, 9);
CsMP = QMP;
CuMP = Matrix(1.0I, 3, 3);
pcontrolPolicy = ManipulatePositionControlPolicy();

xMP = PhysManipulate2D(0.,[0.1,0.2],0.5,[0.3,0.4],0.6,3.,5.,[10.,11.],0.2);
uMP = MControl2D(0.,6.,7.,8.);
bMP = BelMvNormal(xMP.t,vec(xMP),diagm([10.,10.,1.,10.,10.,10.,1.,1.,10.,10.,0.1]));
u_param_min_3D = [-3.,-3.,-3.];
u_param_max_3D = [3.,3.,3.];
posGain = 1.0;
rotGain = 0.2;

uMP = control_nominal(pcontrolPolicy,bMP,u_param_min_3D,u_param_max_3D,posGain,rotGain)
@test uMP.t == 0.0
@test uMP.fx == -0.1
@test uMP.fy == -0.2
@test uMP.tr == 0.5283185307179586

end
