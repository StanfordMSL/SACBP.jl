########################################
## File Name: belief_transition_test.jl
## Author: Haruki Nishimura (hnishimura@stanford.edu)
## Date Created: 2020/05/13
## Description: Test code for src/belief_transition_models.jl
########################################

using LinearAlgebra
using Random
using Distributions

@testset "Belief Transition Test" begin

# covObservePos
p = randn(2);
qi = randn(2);
C = SACBP.covObservePos(p, qi)
@test all(isapprox(C, diagm((0.001 + norm(p - qi)*0.01)*ones(2))))

# UKFPosRange with BelMvNormal
ukfModel = UKFPosRange();
bPos = BelMvNormal(0.,zeros(2),100*Matrix(1.0I, 2, 2));
bVecPos = [bPos for ii = 1:100];
p_robot = PhysPos(0.,[5.,7.]);
s = AugState(p_robot,bPos);
sV = AugState(p_robot,bVecPos);

b_new = trans(ukfModel,bPos,p_robot,0.01,2.,Matrix(1.0I, 2, 2))
@test all(isapprox(b_new.params, [3.24374, 4.62673, 94.7848, -7.45299, -7.45299, 89.3794], atol=1e-4))

Jp = trans_jacobi_auto_p(ukfModel,bPos,p_robot,0.01,2.,Matrix(1.0I, 2, 2))
@test all(isapprox(Jp, [0.851283   0.243418
                        0.251906   1.05732
                       -2.23007   -0.0795886
                       -1.63071   -1.25742
                       -1.63071   -1.25742
                       -0.114878  -3.42515  ], atol=1e-4))

Jb = trans_jacobi_auto_b(ukfModel,bPos,p_robot,0.01,2.,Matrix(1.0I, 2, 2))
@test all(isapprox(Jb, [0.148717  -0.243418    0.00303772  0.0   0.0437606  -0.0139536
                       -0.251906  -0.0573231  -0.0206755   0.0   0.027708    0.00430951
                        2.23007    0.0795886   0.975254    0.0  -0.147259    0.0310205
                        1.63071    1.25742     0.00498744  0.5   0.345869    0.00524383
                        1.63071    1.25742     0.00498744  0.5   0.345869    0.00524383
                        0.114878   3.42515     0.0645743   0.0  -0.140094    0.951848  ], atol=1e-4))

# UKFPosRange with VecBelMVNormal
yvec = [2. for ii = 1:100];
b_new_array = trans(ukfModel,bVecPos,p_robot,0.01,yvec,Matrix(1.0I, 2, 2));
@test all([b_new_array[ii].params == b_new.params for ii = 1:100])

Jp_array = trans_jacobi_auto_p(ukfModel,bVecPos,p_robot,0.01,yvec,Matrix(1.0I, 2, 2));
@test all([isapprox(Jp_array[ii], Jp) for ii = 1:100])

Jb_array = trans_jacobi_auto_b(ukfModel,bVecPos,p_robot,0.01,yvec,Matrix(1.0I, 2, 2));
@test all([isapprox(Jb_array[ii], Jb) for ii = 1:100])

# UKFPosRange with AugState and BelMvNormal
s_new = trans(ukfModel,s,0.01,2.,Matrix(1.0I, 2, 2))
@test s_new.p.pos == [5., 7.]
@test s_new.b.params == b_new.params
@test all(isapprox.(trans_jacobi_auto_p(ukfModel,s,0.01,2.,Matrix(1.0I,2,2)), Jp))
@test all(isapprox.(trans_jacobi_auto_b(ukfModel,s,0.01,2.,Matrix(1.0I,2,2)), Jb))

# UKFPosRange with AugState and VecBelMvNormal
sV_new = trans(ukfModel,sV,0.01,yvec,Matrix(1.0I,2,2))
@test sV_new.p.pos == [5., 7.]
@test all([all(isapprox.(b.params, b_new.params)) for b in sV_new.b])
@test all([all(isapprox.(J, Jp)) for J in trans_jacobi_auto_p(ukfModel,sV,0.01,yvec,Matrix(1.0I,2,2))])
@test all([all(isapprox.(J, Jb)) for J in trans_jacobi_auto_b(ukfModel,sV,0.01,yvec,Matrix(1.0I,2,2))])

trans(ukfModel,bVecPos,p_robot,0.01,Matrix(1.0I,2,2),MersenneTwister(1234))
trans(ukfModel,sV,0.01,Matrix(1.0I,2,2),MersenneTwister(1234))

# CPredictManipulate2D
xMP = PhysManipulate2D(0.,[0.1,0.2],0.5,[0.3,0.4],0.6,3.,5.,[10.,11.],0.7);
uMP = MControl2D(0.,6.,7.,8.);

predictEKFMP = CPredictManipulate2D();
bMP = BelMvNormal(xMP.t,vec(xMP),10*Matrix(1.0I, 11, 11));

bMP_new = trans(predictEKFMP,bMP,uMP,0.01,Matrix(1.0I, 11, 11))
@test bMP_new.t == 0.01
d = Distributions.MvNormal(bMP_new)
@test all(isapprox.(d.μ, [0.103, 0.204, 0.506, 0.3193, 0.4224, 0.491658, 3.0, 5.0, 10.0, 11.0, 0.7], atol=1e-4))

J1 = trans_jacobi_auto(predictEKFMP,bMP,uMP,Matrix(1.0I,11,11))
J2 = trans_jacobi(predictEKFMP,bMP,uMP,Matrix(1.0I,11,11))
@test all(isapprox.(J1, J2))

Hu = trans_u_coeff(predictEKFMP,bMP,Matrix(1.0I,11,11))

# DUpdateManipulate2D
updateEKFMP = DUpdateManipulate2D();
bMP_new2 = trans(updateEKFMP, bMP, uMP, ones(9), Matrix(1.0I, 9, 9))
@test bMP_new2.t == 0.0
d2 = Distributions.MvNormal(bMP_new2)
@test all(isapprox.(d2.μ, [-2.60194, -11.0021, 0.478504, 0.552211, 0.657014, 0.0276191, 2.31804, 8.64641, 9.45285, 9.87449, 0.595059], atol=1e-4))

trans_jacobi_auto(updateEKFMP,bMP,uMP,ones(9),Matrix(1.0I, 9, 9))

# CDEKFManipulate2D
cdEKFMP = CDEKFManipulate2D();
b_new_2 = trans(cdEKFMP,bMP,uMP,0.01,0.2,ones(9),Matrix(1.0I, 11, 11),Matrix(1.0I, 9, 9))
@test b_new_2.t == 0.2
d3 = Distributions.MvNormal(b_new_2);
@test all(isapprox.(d3.μ, [-4.91794, -9.74958, 0.293202, -1.55987, 4.92263, -0.437129, 3.12938, 6.42346, 9.27099, 9.46519, 0.814831], atol=1e-4))

end
