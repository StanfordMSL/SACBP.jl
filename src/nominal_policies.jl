########################################
## File Name: nominal_policies.jl
## Author: Haruki Nishimura (hnishimura@stanford.edu)
## Date Created: 2020/05/14
## Description: Nominal policies for SACBP
########################################

abstract type NominalPolicy end

struct GradientMultiTargetLocalizationPolicy <: NominalPolicy end

function control_nominal(policy::GradientMultiTargetLocalizationPolicy,
                         model::UKFPosRange,
                         s::AugState{PhysPos{T},VecBelMvNormal{T}},
                         u_param_min::Vector{<:Real},
                         u_param_max::Vector{<:Real},
                         dt::Real,
                         Q::Matrix{T}) where T <: Real
    costModel = CostPosRangeLocalization();
    grad = sum(transpose.(trans_jacobi_auto_p(model,s,dt,zeros(length(s.b)),Q)).*termCost_grad_b(costModel,trans(model,s,dt,zeros(length(s.b)),Q)));
    u_val = -1.0*(grad/(norm(grad)+1e-10));
    for ii = 1:length(u_val)
        if u_val[ii] < u_param_min[ii]
            u_val[ii] = u_param_min[ii];
        elseif u_val[ii] > u_param_max[ii]
            u_val[ii] = u_param_max[ii];
        end
    end
    return PosControl(s.t,u_val)
end


struct ManipulatePositionControlPolicy <: NominalPolicy end

function control_nominal(policy::ManipulatePositionControlPolicy,
                         b::BelMvNormal{<:Real},
                         u_param_min::Vector{<:Real},
                         u_param_max::Vector{<:Real},
                         posGain::Real,
                         rotGain::Real)
    μ,Σ = b.params[1:b.dim],reshape(b.params[b.dim+1:end],b.dim,b.dim);
    fVec = -posGain.*μ[1:2];
    torque = -rotGain*(μ[3] - pi); # Target θ is pi.
    u_val = [fVec;torque];
    for ii = 1:length(u_val)
        if u_val[ii] < u_param_min[ii]
            u_val[ii] = u_param_min[ii];
        elseif u_val[ii] > u_param_max[ii]
            u_val[ii] = u_param_max[ii];
        end
    end
    return MControl2D(b.t,u_val)
end
