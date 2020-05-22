########################################
## File Name: runtests.jl
## Author: Haruki Nishimura (hnishimura@stanford.edu)
## Date Created: 2020/05/12
## Description: Test script for SACBP
########################################

using Test
using SACBP

@testset "SACBP Unit Tests" begin
@info "Executing Type Definition Test"
include("type_test.jl");
@info "Executing State Transition Test"
include("state_transition_test.jl");
@info "Executing Observation Test"
include("observation_test.jl")
@info "Executing Belief Transition Test"
include("belief_transition_test.jl")
@info "Executing Cost Test"
include("cost_test.jl")
@info "Executing Costate Transition Test"
include("costate_transition_test.jl")
@info "Executing Nominal Policies Test"
include("nominal_policies_test.jl")
@info "Executing Forward-Backward Simulation Test"
include("forward_backward_simulation_test.jl")
@info "Executing Closed-Loop Forward-Backward Simulation Test"
include("closed_loop_forward_backward_simulation_test.jl")
@info "Executing SAC Control Test"
include("sac_control_test.jl")
@info "Executing Closed-Loop SAC Control Test"
include("closed_loop_sac_control_test.jl")
end
