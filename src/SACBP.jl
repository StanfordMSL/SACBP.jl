#///////////////////////////////////////
#// File Name: SACBP.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2020/05/12
#// Description: Julia package for SACBP
#///////////////////////////////////////

module SACBP

using AutoHashEquals
using Distributed
using Distributions
using ForwardDiff
using Plots
using Convex
# using SCS
using ECOS
import Future
using StatsFuns
using LinearAlgebra
using Random

export
    Belief,
    BelMvNormal,
    VecBelMvNormal,
    plot_e_ellipse!
include("belief_types.jl")

export
    PhysState,
    PhysPos,
    PhysManipulate2D,
    AugState
include("state_types.jl")

export
    Control,
    PosControl,
    MControl2D
include("control_types.jl")

export
    TransModel,
    StateTransModel,
    TransModel_Pos,
    TransModel_Manipulate2D,
    trans,
    trans_jacobi,
    trans_jacobi_auto,
    trans_jacobi_euler,
    trans_u_coeff,
    ObserveModel,
    ObserveModel_Range,
    ObserveModel_Manipulate2D,
    observe,
    observe_jacobi,
    observe_jacobi_auto
include("state_transition_observation_models.jl")

export
    BeliefTransModel,
    GaussianFilter,
    UKFPosRange,
    CPredictManipulate2D,
    DUpdateManipulate2D,
    CDEKFManipulate2D,
    covObservePos,
    trans_jacobi_auto_p,
    trans_jacobi_auto_b
include("belief_transition_models.jl")

export
    CostModel,
    CostPosRangeLocalization,
    CostManipulate2D,
    instCost,
    instCost_grad,
    instCost_grad_p,
    instCost_grad_b,
    termCost,
    termCost_grad,
    termCost_grad_p,
    termCost_grad_b
include("cost_models.jl")

export
    cotrans
include("costate_transition_models.jl")

export
    NominalPolicy,
    GradientMultiTargetLocalizationPolicy,
    ManipulatePositionControlPolicy,
    control_nominal
include("nominal_policies.jl")

export
    SACSimulationModel,
    SimulatePosRangeLocalization2D,
    SimulateManipulate2D,
    simulateForward,
    simulateBackward,
    evaluateCost
include("forward_backward_simulation.jl")

export
    getControlCoeffs,
    controlCoeffsExpected,
    optControlSchedule,
    determineControlTime,
    sacControlUpdate
include("sac_controller.jl")

end # module
