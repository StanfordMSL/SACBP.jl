using SACBP

function pControlUpdate(model::SimulateManipulate2D,
                        b::BelMvNormal{T},
                        UArray::Vector{<:MControl2D},
                        u_param_min::Vector{<:Real},
                        u_param_max::Vector{<:Real},
                        posGain::Real,
                        rotGain::Real) where T <: Real
    nominalPolicy = ManipulatePositionControlPolicy();
    u_val = vec(control_nominal(nominalPolicy,b,u_param_min,u_param_max,posGain,rotGain))
    UArray_new = copy(UArray)
    for ii = 1:length(UArray_new)
        if UArray_new[ii].t < round(b.t + model.dto,digits=5)
            UArray_new[ii] = MControl2D(UArray_new[ii].t,u_val);
        end
    end
    return UArray_new
end;

function pControlVal(b::BelMvNormal{T},
                     u_param_min::Vector{<:Real},
                     u_param_max::Vector{<:Real},
                     posGain::Real,
                     rotGain::Real) where T <: Real
    nominalPolicy = ManipulatePositionControlPolicy();
    u_val = vec(control_nominal(nominalPolicy,b,u_param_min,u_param_max,posGain,rotGain))
    return u_val
end;
