using SACBP
using POMDPs
using MCTS
import Base.isequal

# Localiztion Task

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
                numActions::Int64)
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


# Manipulation Task

include("p_control.jl")

struct BelMDP <: POMDPs.MDP{BelMvNormal{T} where T <: Real, Vector{Float64}}
    dtc::Float64
    dto::Float64
    u_param_min::Vector{Float64}
    u_param_max::Vector{Float64}
    numActions::Int64
    Q::Matrix{Float64}
    R::Matrix{Float64}
    Cs::Matrix{Float64}
    Cu::Matrix{Float64}
end
function BelMDP(simModel::SimulateManipulate2D,
                u_param_min::Vector{Float64}, u_param_max::Vector{Float64},
                numActions::Int64)
    return BelMDP(simModel.dtc, simModel.dto,
                  u_param_min, u_param_max, numActions,
                  simModel.Q, simModel.R, simModel.Cs, simModel.Cu)
end


function POMDPs.actions(mdp::BelMDP)
    u_fx = range(mdp.u_param_min[1],stop=mdp.u_param_max[1],length=mdp.numActions);
    u_fy = range(mdp.u_param_min[2],stop=mdp.u_param_max[2],length=mdp.numActions);
    u_tr = range(mdp.u_param_min[3],stop=mdp.u_param_max[3],length=mdp.numActions);
    return [[u_fx[ii],u_fy[jj],u_tr[kk]] for ii = 1:length(u_fx) for jj = 1:length(u_fy) for kk = 1:length(u_tr)];
end

struct DEKFManipulate2D <: GaussianFilter
# Discrete EKF model for belief transition in planar manipulation problem.
end

function trans_mcts(model::DEKFManipulate2D, b::BelMvNormal{T}, u::MControl2D{<:Real},
                    dt::Real, Q::Matrix{Float64}, R::Matrix{Float64}, rng::AbstractRNG) where T <: Real
    # EKF Predict and then Update for MCTS Transition
    if b.t != u.t
        error(ArgumentError("BelState and Control have inconsistent time parameters."));
    else
        transModel = TransModel_Manipulate2D();
        μ,Σ = b.params[1:b.dim],reshape(b.params[b.dim+1:end],b.dim,b.dim);

        # Predict.
        μMP = PhysManipulate2D(b.t,μ);
        A = trans_jacobi(transModel,μMP,u)*dt + Matrix(1.0I, b.dim, b.dim); # euler approximation
        Γ = A*Σ*A' + Q;
        Γ = Symmetric(Γ)
        try
            W = cholesky(Γ).L;
        catch
            println(μ)
            println(u)
            println(Σ)
        end
        W = cholesky(Γ).L;
        μ = vec(trans(transModel,PhysManipulate2D(b.t,μ),u,dt));

        # Update. (square-root form)
        μMP = PhysManipulate2D(b.t,μ);
        observeModel = ObserveModel_Manipulate2D();
        C = observe_jacobi(observeModel,μMP,u);
        Z = W'*C';
        H = Z'*Z + R
        H = Symmetric(H);
        try
            U = cholesky(H).L;
        catch
            println(μ)
            println(u)
            println(W)
        end
        U = cholesky(H).L;
        V = cholesky(R).L;
        K = Γ*C'/H;
        y = observe(observeModel,μMP,u,Matrix(H),rng);
        y_pred_mean = observe(observeModel,μMP,u);
        if y[3] - y_pred_mean[3] > pi   # Correct angular difference.
            y[3] -= 2*pi;
        elseif y[3] - y_pred_mean[3] < -pi
            y[3] += 2*pi;
        end
        μ += K*(y - y_pred_mean);
        μ[3] = mod2pi(μ[3]);
        Ω = W*(Matrix(1.0I, b.dim, b.dim) - Z/(U')/(U+V)*Z'); # See Andrews 1968
        g = [μ;vec(Symmetric(Ω*Ω'))];
        return BelMvNormal(round(b.t+dt,digits=5),g)
    end
end

function POMDPs.gen(::DDNOut{(:sp, :r)},
                    mdp::BelMDP,
                    b::BelMvNormal{<:Real},
                    u::Vector{Float64},
                    rng::AbstractRNG)
    b_new = trans_mcts(DEKFManipulate2D(),b,MControl2D(b.t,u),mdp.dto,mdp.Q,mdp.R,rng)
    reward = -instCost(CostManipulate2D(),b,MControl2D(b.t,u),mdp.Cs,mdp.Cu)*mdp.dto;
    return b_new,reward
end;

function isequal(b1::BelMvNormal{Float64},
                 b2::BelMvNormal{Float64})
    return b1.t == b2.t && b1.params == b2.params
end;

POMDPs.discount(::BelMDP) = 1.0;

function mctsControlUpdate(policy::MCTS.DPWPlanner,
                           b::BelMvNormal{Float64},
                           UArray::Vector{<:MControl2D},
                           dt::Float64)
    u_val = action(policy,b)
    UArray_new = copy(UArray)
    for ii = 1:length(UArray_new)
        if UArray_new[ii].t < round(b.t + dt,digits=5)
            UArray_new[ii] = MControl2D(UArray_new[ii].t,u_val);
        end
    end
    return UArray_new
end

struct PControl <: POMDPs.Policy
    u_param_min::Vector{Float64}
    u_param_max::Vector{Float64}
    posGain::Real
    rotGain::Real
end;

function POMDPs.action(p::PControl,b::BelMvNormal{Float64})
    return pControlVal(b,p.u_param_min,p.u_param_max,p.posGain,p.rotGain)
end
