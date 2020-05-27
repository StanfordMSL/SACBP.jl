using SACBP
using POMDPs
using MCTS
import Base.isequal

# include("gradient_greedy.jl")

struct AugMDP <: POMDPs.MDP{AugState{<:PhysPos,<:VecBelMvNormal{T} where T<:Real},Vector{Float64}}
    dtc::Float64
    dto::Float64
    u_param_min::Vector{Float64}
    u_param_max::Vector{Float64}
    numActions::Int64
    Cu::Matrix{Float64}
    Q::Matrix{Float64}
end
function AugMDP(simPosUKF::SimulatePosRangeLocalization2D,
                u_param_min::Vector{Float64}, u_param_max::Vector{Float64},
                numActions)
    return AugMDP(simPosUKF.dtc, simPosUKF.dto,
                  u_param_min, u_param_max, numActions,
                  simPosUKF.Cu, simPosUKF.Q)
end

function POMDPs.actions(mdp::AugMDP)
    u_x = range(mdp.u_param_min[1],stop=mdp.u_param_max[1],length=mdp.numActions);
    u_y = range(mdp.u_param_min[2],stop=mdp.u_param_max[2],length=mdp.numActions);
    return [[u_x[ii],u_y[jj]] for ii = 1:length(u_x) for jj = 1:length(u_y)];
end
#=
struct Greedy <: POMDPs.Policy
    u_param_min::Vector{Float64}
    u_param_max::Vector{Float64}
    Q::Matrix{Float64}
    dto::Float64
end

function POMDPs.action(p::Greedy,s::AugState{PhysPos{Float64},VecBelMvNormal{Float64}})
    return gradientControlVal(UKFPosRange(),s,p.u_param_min,p.u_param_max,p.Q,p.dto)
end
=#

function POMDPs.gen(::DDNOut{(:sp, :r)},
                    mdp::AugMDP,
                    s::AugState{<:PhysPos,<:VecBelMvNormal{T} where T<:Real},
                    u::Vector{Float64},
                    rng::AbstractRNG)
    pathCost = 0.;
    SArray = Vector{typeof(s)}(undef,Int64(mdp.dto/mdp.dtc)+1);
    SArray[1] = s;
    for ii = 1:length(SArray)-1
        if SArray[ii].t < round(s.t + mdp.dto,digits=5)
            u_current = PosControl(round(s.t + mdp.dtc*(ii-1),digits=5),u);
            p = trans(TransModel_Pos(),SArray[ii].p,u_current,mdp.dtc);
            bVec = similar(SArray[ii].b);
            for jj = 1:length(bVec)
                bVec[jj] = BelMvNormal(p.t,SArray[ii].b[jj].params);
            end
            SArray[ii+1] = AugState(p,bVec);
            pathCost += instCost(CostPosRangeLocalization(),SArray[ii+1],u_current,mdp.Cu)*mdp.dtc;
        else
            u_current = PosControl(round(s.t + mdp.dtc*(ii-1),5),zeros(2));
            p = trans(TransModel_Pos(),SArray[ii].p,u_current,mdp.dtc);
            bVec = similar(SArray[ii].b);
            for jj = 1:length(bVec)
                bVec[jj] = BelMvNormal(p.t,SArray[ii].b[jj].params);
            end
            SArray[ii+1] = AugState(p,bVec);
            pathCost += instCost(CostPosRangeLocalization(),SArray[ii+1],u_current,mdp.Cu)*mdp.dtc;
        end
    end
    yVec,SArray[end] = trans(UKFPosRange(),SArray[end],mdp.dto,mdp.Q,rng)
    termCostDifference = termCost(CostPosRangeLocalization(),SArray[end]) - termCost(CostPosRangeLocalization(),s); # This is equivalent to having the terminal cost.
    reward = -(pathCost + termCostDifference);
    return SArray[end],reward
end;

function isequal(s1::AugState{SACBP.PhysPos{Float64},Array{SACBP.BelMvNormal{Float64},1}},
                 s2::AugState{SACBP.PhysPos{Float64},Array{SACBP.BelMvNormal{Float64},1}})
    return s1.t == s2.t && s1.p.pos == s2.p.pos && all([s1.b[ii].params == s2.b[ii].params for ii = 1:length(s1.b)])
end

POMDPs.discount(::AugMDP) = 1.0;

function mctsControlUpdate(policy::MCTS.DPWPlanner,
                           s::AugState{PhysPos{Float64},VecBelMvNormal{Float64}},
                           UArray::Vector{<:PosControl}, dto::Float64)
    u_val = action(policy,s)
    UArray_new = copy(UArray)
    for ii = 1:length(UArray_new)
        if UArray_new[ii].t < round(s.t + dto,digits=5)
            UArray_new[ii] = PosControl(UArray_new[ii].t,u_val);
        end
    end
    return UArray_new
end

struct MCTSParams
    numActions::Int64
    # treeDepth::Int64
    numSamples::Int64
    expConst::Float64
    kAct::Float64
    αAct::Float64
    kState::Float64
    αState::Float64
end
