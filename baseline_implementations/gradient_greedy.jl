using SACBP
using LinearAlgebra

function gradientControlUpdate(model::UKFPosRange,
                               s::AugState{PhysPos{T},VecBelMvNormal{T}},
                               UArray::Vector{<:PosControl},
                               u_param_min::Vector{<:Real},
                               u_param_max::Vector{<:Real},
                               Q::Matrix{T},
                               dto::Float64) where T <: Real
    gradientPolicy = GradientMultiTargetLocalizationPolicy();
    u_val = vec(control_nominal(gradientPolicy,model,s,u_param_min,u_param_max,dto,Q));
    UArray_new = copy(UArray)
    for ii = 1:length(UArray_new)
        if UArray_new[ii].t < round(s.t + dto,digits=5)
            UArray_new[ii] = PosControl(UArray_new[ii].t,u_val);
        end
    end
    return UArray_new
end;
#=
function gradientControlVal(model::UKFPosRange,
                            s::AugState{PhysPos{T},VecBelMvNormal{T}},
                            u_param_min::Vector{<:Real},
                            u_param_max::Vector{<:Real},
                            Q::Matrix{T},
                            dto::Float64) where T <: Real
    gradientPolicy = GradientMultiTargetLocalizationPolicy();
    u_val = vec(control_nominal(gradientPolicy,model,s,u_param_min,u_param_max,dto,Q));
    return u_val
end;
=#
