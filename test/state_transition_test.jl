########################################
## File Name: state_transition_test.jl
## Author: Haruki Nishimura (hnishimura@stanford.edu)
## Date Created: 2020/05/12
## Description: Test code for transition functions in
## src/state_transition_observation_models.jl
########################################

using LinearAlgebra
using Random

@testset "State Transition Test" begin

# TransModel_Pos
transPos = TransModel_Pos();
xPos = PhysPos(0.,zeros(2));
uPos = PosControl(0.,[2.,3.2]);
p = trans(transPos,xPos,uPos,0.01);
@test p.t == 0.01
@test p.pos == [0.02, 0.032]
@test p.dim == 2

p2 = trans(transPos, xPos, uPos, 0.01, Matrix(1.0I, 2, 2), MersenneTwister(1234));

J = trans_jacobi(transPos,xPos,uPos);
@test J == zeros(2, 2);

J2 = trans_jacobi_euler(transPos,xPos,uPos,0.01)
@test J2 == Matrix(1.0I, 2, 2)

C = trans_u_coeff(transPos, xPos)
@test C == Matrix(1.0I, 2, 2)

# TransModel_Manipulate2D
transMP = TransModel_Manipulate2D();
xMP = PhysManipulate2D(0.,[0.1,0.2],0.5,[0.3,0.4],0.6,3.,5.,[10.,11.],0.7);
uMP = MControl2D(0.,6.,7.,8.);
m = trans(transMP,xMP,uMP,0.01)
@test m.t == 0.01
@test all(isapprox.(m.pos, [0.103, 0.204]))
@test isapprox(m.θ, 0.506)
@test all(isapprox.(m.vel, [0.3193, 0.4224]))
@test isapprox(m.ω, 0.4916580629)
@test m.m == 3.0
@test m.J == 5.0
@test m.r == [10., 11.]
@test m.μ == 0.7
@test m.dim == 2

m2 = trans(transMP, xMP, uMP, 0.01, Matrix(1.0I, 11, 11), MersenneTwister(1234))

J3 = trans_jacobi(transMP, xMP, uMP)
@test all(isapprox.(trans_jacobi_auto(transMP, xMP, uMP), J3))

J4 = trans_jacobi_euler(transMP, xMP, uMP, 0.01)
@test all(isapprox.(J4, J3.*0.01 + Matrix(1.0I, 11, 11)))

C2 = trans_u_coeff(transMP, xMP)
@test all(isapprox.(C2, [0.0       0.0       0.0
                         0.0       0.0       0.0
                         0.0       0.0       0.0
                         0.333333  0.0       0.0
                         0.0       0.333333  0.0
                        -2.88953   0.700429  0.2
                         0.0       0.0       0.0
                         0.0       0.0       0.0
                         0.0       0.0       0.0
                         0.0       0.0       0.0
                         0.0       0.0       0.0], atol=1e-5))

end
