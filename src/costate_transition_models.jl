########################################
## File Name: costate_transition_models.jl
## Author: Haruki Nishimura (hnishimura@stanford.edu)
## Date Created: 2020/05/14
## Description: Costate Transition Models for SACBP
########################################

function cotrans(transModel::TransModel_Pos,costModel::CostPosRangeLocalization,
                 cos::AugState{PhysPos{T},VecBelMvNormal{T}},
                 s::AugState{PhysPos{T},VecBelMvNormal{T}},u::PosControl,
                 dtc::Real) where T <: Real
    # Continuous Time Transition. (Backward) Euler Approximation.
    if cos.t != u.t || s.t != u.t
        error(ArgumentError("AugState and Control have incosistent time parameters."))
    else
        t = round(s.t - dtc,digits=5);
        pVec = vec(cos.p) + instCost_grad_p(costModel,s)*dtc + trans_jacobi(transModel,s.p,u)'*vec(cos.p)*dtc;
        p = PhysPos(t,pVec);
        b = similar(s.b);
        instCostGradB = instCost_grad_b(costModel,s);
        for ii = 1:length(s.b)
            params =  cos.b[ii].params + instCostGradB[ii]*dtc;
            b[ii] = BelMvNormal(t,params);
        end
        return AugState(t,p,b);
    end
end

function cotrans(transModel::UKFPosRange,costModel::CostPosRangeLocalization,
                 cos::AugState{PhysPos{T},VecBelMvNormal{T}},
                 s::AugState{PhysPos{T},VecBelMvNormal{T}},
                 yvec::Vector{<:Real},dto::Real,Q::Matrix{Float64}) where T <: Real

    pVec = vec(cos.p);
    b = similar(cos.b);
    jpVec = trans_jacobi_auto_p(transModel,s,dto,yvec,Q);
    jbVec = trans_jacobi_auto_b(transModel,s,dto,yvec,Q);
    for ii = 1:length(cos.b)
        pVec += jpVec[ii]'*cos.b[ii].params;
        params = jbVec[ii]'*cos.b[ii].params
        b[ii] = BelMvNormal(s.t,params);
    end
    p = PhysPos(s.t,pVec);
    return AugState(s.t,p,b);
end


function cotrans(transModel::CPredictManipulate2D,costModel::CostManipulate2D,
                 cob::BelMvNormal{T},b::BelMvNormal{T},u::MControl2D{<:Real},dt::Real,
                 Q::Matrix{Float64},Cs::Matrix{Float64}) where T <: Real
    # Continuous Time Transition. (Backward) Euler Approximation.
    if cob.t != u.t || b.t != u.t
        error(ArgumentError("BelState and Control have inconsistent time parameters."))
    else
        t = round(b.t - dt,digits=5);
        params = cob.params + instCost_grad(costModel,b,Cs)*dt + trans_jacobi(transModel,b,u,Q)'*cob.params*dt;
        return BelMvNormal(t,params)
    end
end

function cotrans(transModel::DUpdateManipulate2D,costModel::CostManipulate2D,
                 cob::BelMvNormal{T},b::BelMvNormal{T},u::MControl2D{<:Real},y::Vector{<:Real},
                 R::Matrix{Float64}) where T <: Real
    # Discrete Time Transition. (Backward) Euler Approximation.
    if b.t != u.t
        error(ArgumentError("BelState and Control have inconsistent time parameters."))
    else
        params =  trans_jacobi_auto(transModel,b,u,y,R)'*cob.params;
        return BelMvNormal(b.t,params)
    end
end
