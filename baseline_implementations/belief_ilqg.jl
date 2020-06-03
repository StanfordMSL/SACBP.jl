# The offline policy in "ilqg_pcontrol_policy_20sec_2.jld2" was computed
# using the iterative LQG algorithm in belief space presented in
# "Motion Planning under Uncertainty using Iterative Local Optimization in
# Belief Space" by Jur van den Berg, Sachin Patil, and Ron Alterovitz.
# The implementation of the algorithm that derived the policy was written in
# Julia v0.6 and is not currently included in this codebase, although we plan
# to migrate it to Julia v1.3 as well.

using SACBP

function ilqgControlUpdate(ilqgPolicy::Dict{String,Any},b::BelMvNormal{<:Real},UArray::Vector{<:MControl2D},
                           u_param_min::Vector{<:Real},u_param_max::Vector{<:Real},dto::Float64,dtexec::Float64)
    uVec,bVec,lVec,LVec = ilqgPolicy["uVec_new"],ilqgPolicy["bVec_new"],ilqgPolicy["lVec"],ilqgPolicy["LVec"];
    index = Int64(round(b.t/dto,digits=5))+1;
    if (bVec[index].t != b.t)
        error("Time Parameter Inconsistent.")
    end
    u_val = LVec[index]*(b.params - bVec[index].params) + (lVec[index] + vec(uVec[index]));
    for ii = 1:length(u_val)
        if u_val[ii] < u_param_min[ii]
            u_val[ii] = u_param_min[ii];
        elseif u_val[ii] > u_param_max[ii]
            u_val[ii] = u_param_max[ii];
        end
    end
    UArray_new = copy(UArray)
    for ii = 1:length(UArray_new)
        if UArray_new[ii].t < round(b.t + dtexec,digits=5)
            UArray_new[ii] = MControl2D(UArray_new[ii].t,u_val);
        end
    end
    return UArray_new
end
