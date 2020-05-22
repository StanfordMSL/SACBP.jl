########################################
## File Name: observation_test.jl
## Author: Haruki Nishimura (hnishimura@stanford.edu)
## Date Created: 2020/05/12
## Description: Test code for observation functions in
## src/state_transition_observation_models.jl
########################################

using LinearAlgebra
using Random

@testset "Observation Test" begin

# ObserveModel_Range
observeRange = ObserveModel_Range();
xPos = PhysPos(0.,zeros(2));
q_pos = [0.4,10.0];

o = observe(observeRange,xPos,q_pos)
@test isapprox(o, 10.0079968)

o2 = observe(observeRange,xPos,q_pos,Matrix(1.0I, 2, 2),MersenneTwister(123))

J = observe_jacobi_auto(observeRange,xPos,q_pos)
J2 = observe_jacobi(observeRange,xPos,q_pos)
@test all(isapprox.(J, J2))

# ObserveModel_Manipulate2D
observeMP = ObserveModel_Manipulate2D();
xMP = PhysManipulate2D(0.,[0.1,0.2],0.5,[0.3,0.4],0.6,3.,5.,[10.,11.],0.7);
uMP = MControl2D(0.,6.,7.,8.);

o3 = observe(observeMP,xMP,uMP)
@test all(isapprox.(o3, [3.60214
                        14.64766
                         0.5
                        -8.36860
                         2.50129
                         0.6
                       157.19801
                       -40.90407
                       -10.83419], atol=1e-5))

o4 = observe(observeMP, xMP, uMP, Matrix(1.0I, 9, 9), MersenneTwister(123))

J3 = observe_jacobi_auto(observeMP,xMP,uMP)
J4 = observe_jacobi(observeMP,xMP,uMP)
@test all(isapprox.(J3, J4))

end
