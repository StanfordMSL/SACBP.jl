########################################
## File Name: type_test.jl
## Author: Haruki Nishimura (hnishimura@stanford.edu)
## Date Created: 2020/05/12
## Description: Test code for src/state_types.jl, src/belief_types.jl,
## src/control_types.jl
########################################

using LinearAlgebra
using Distributions

@testset "Type Definition Test" begin

# Belief Types
b1 = BelMvNormal(1., zeros(2), Matrix(1.0I, 2, 2));
@test b1.t == 1.0
@test b1.params == [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
@test b1.dim == 2

d1 = MvNormal(b1)
@test d1.μ == zeros(2)
@test Matrix(d1.Σ) == Matrix(1.0I, 2, 2)

b2 = VecBelMvNormal([b1 for ii = 1:10])
@test length(b2) == 10

d2 = MvNormal.(b2);
@test length(d2) == 10

# State Types
p1 = PhysPos(1., zeros(2), 2)
p2 = PhysPos(1., zeros(2))
@test p2.t == p1.t
@test p2.pos == p1.pos
@test p2.dim == p1.dim
@test vec(p2) == zeros(2)

p3 = PhysManipulate2D(1., zeros(2), pi/2, zeros(2), 0., 1., 1., ones(2), 0.3)
@test vec(p3) == [0.0, 0.0, pi/2, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.3]

p4 = PhysManipulate2D(1., vec(p3))
@test p4.t == 1.0
@test p4.pos == [0.0, 0.0]
@test p4.θ == pi/2
@test p4.vel == [0.0, 0.0]
@test p4.ω == 0.
@test p4.m == 1.
@test p4.J == 1.
@test p4.r == [1.0, 1.0]
@test p4.μ == 0.3
@test p4.dim == 2

s1 = AugState(p1, b2)
@test s1.t == p1.t
@test s1.p == p1
@test s1.b == b2

# Control Types
c1 = PosControl(0., zeros(2))
@test c1.t == 0.
@test c1.vel == [0.0, 0.0]
@test c1.dim == 2

c2 = PosControl(0., 0.)
@test c2.t == 0.
@test c2.vel == [0.0]
@test c2.dim == 1

c3 = MControl2D(0., 1., 2., 3.)
@test c3 == MControl2D(0., [1., 2., 3.])
@test vec(c3) == [1., 2., 3.]

end
