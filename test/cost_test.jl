########################################
## File Name: cost_test.jl
## Author: Haruki Nishimura (hnishimura@stanford.edu)
## Date Created: 2020/05/14
## Description: Test code for src/cost_models.jl
########################################

using LinearAlgebra
using Distributions
using ForwardDiff

@testset "Cost Test" begin

# CostPosRangeLocalization
cost_prl = CostPosRangeLocalization();
p_robot = PhysPos(0.,[5.,7.]);
bPos = BelMvNormal(0.,zeros(2),100*Matrix(1.0I, 2, 2));
bVecPos = [bPos for ii = 1:100];
sV = AugState(p_robot,bVecPos);
uPos = PosControl(0.,[2.,3.2]);
Cu = Matrix(1.0I, 2, 2);

ic = instCost(cost_prl, sV, uPos, Cu)
@test isapprox(ic, 0.5*(2.0^2 + 3.2^2))

ic_grad_p = instCost_grad_p(cost_prl, sV)
@test ic_grad_p == zeros(2)

ic_grad_b = instCost_grad_b(cost_prl, sV)
@test all([g == zeros(6) for g in ic_grad_b])

tc = termCost(cost_prl, sV)
@test isapprox(tc, 100*exp(entropy(MvNormal(bPos))))

tc_grad_p = termCost_grad_p(cost_prl, sV)
@test tc_grad_p == zeros(2)

tc_grad_b = termCost_grad_b(cost_prl, sV)
function test_termCost(b::Vector{<:Real})
    μ,Σ = b[1:2], reshape(b[3:6], 2, 2)
    d = MvNormal(μ, Σ)
    return exp(entropy(d))
end
test_termCost_grad_b(b::Vector{Float64}) = ForwardDiff.gradient(test_termCost, b)
@test all([all(isapprox.(b, test_termCost_grad_b([0., 0., 100., 0., 0., 100.]))) for b in tc_grad_b])


# CostManipulate2D
cost_m = CostManipulate2D()
xMP = PhysManipulate2D(0.,[0.1,0.2],0.5,[0.3,0.4],0.6,3.,5.,[10.,11.],0.7);
uMP = MControl2D(0.,6.,7.,8.);
bMP = BelMvNormal(xMP.t,vec(xMP),10*Matrix(1.0I, 11, 11));
CsMP = Matrix(1.0I, 11, 11);
CuMP = Matrix(1.0I, 3, 3);

icMP = instCost(cost_m, bMP, uMP, CsMP, CuMP)
μ_target = [0.,0.,pi,0.,0.,0.,0.,0.,0.,0.,0.];
@test isapprox(icMP, 0.5*tr(CsMP*10*Matrix(1.0I, 11, 11)) + 0.5*(vec(xMP) - μ_target)'*CsMP*(vec(xMP) - μ_target) + 0.5*vec(uMP)'*CuMP*vec(uMP))

ic_grad_mp = instCost_grad(cost_m, bMP, CsMP)
function test_instCostMP(b::Vector{<:Real})
    bel = BelMvNormal(0.0, b)
    return instCost(cost_m, bel, uMP, CsMP, CuMP)
end
test_instCostMP_grad(b::Vector{Float64}) = ForwardDiff.gradient(test_instCostMP, b)
@test all(isapprox.(ic_grad_mp, test_instCostMP_grad(bMP.params)))

tcMP = termCost(cost_m, bMP, CsMP)
@test isapprox(tcMP, icMP - 0.5*vec(uMP)'*CuMP*vec(uMP))

tc_grad_mp = termCost_grad(cost_m, bMP, CsMP)
@test all(isapprox.(tc_grad_mp, ic_grad_mp))

end
