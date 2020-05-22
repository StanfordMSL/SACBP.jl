########################################
## File Name: cost_models.jl
## Author: Haruki Nishimura (hnishimura@stanford.edu)
## Date Created: 2020/05/13
## Description: Cost Models for SACBP
########################################

using StatsFuns

abstract type CostModel end


struct CostPosRangeLocalization <: CostModel
end
function instCost(model::CostPosRangeLocalization,s::AugState{PhysPos{U},VecBelMvNormal{T}},
                  u::PosControl,Cu::Matrix{Float64}) where T <: Real where U <: Real
    return 1/2*vec(u)'*Cu*vec(u)
end
function instCost_grad_p(model::CostPosRangeLocalization,s::AugState{PhysPos{U},VecBelMvNormal{T}}) where T <: Real where U <: Real
    return zeros(length(vec(s.p)))
end
function instCost_grad_b(model::CostPosRangeLocalization,s::AugState{PhysPos{U},VecBelMvNormal{T}}) where T <: Real where U <: Real
    gVecs = Vector{Vector{T}}(undef, length(s.b));
    for ii = 1:length(s.b)
        gVecs[ii] = zeros(length(s.b[ii].params))
    end
    return gVecs
end
function termCost(model::CostPosRangeLocalization,
                  s::AugState{PhysPos{U},VecBelMvNormal{T}}) where T <: Real where U <: Real
    DArray_end = Distributions.MvNormal.(s.b);
    entropies = Distributions.entropy.(DArray_end);
    weighted = exp.(entropies); # Check coefficient!
    return sum(weighted);
    #return sum(Distributions.entropy.(DArray_end))
end
function termCost_grad_p(model::CostPosRangeLocalization,s::AugState{PhysPos{U},VecBelMvNormal{T}}) where T <: Real where U <: Real
    return zeros(length(vec(s.p)))
end
function termCost_grad_b(model::CostPosRangeLocalization,s::AugState{PhysPos{U},VecBelMvNormal{T}}) where T <: Real where U <: Real
    gVecs = Vector{Vector{T}}(undef, length(s.b));
    entropies = Distributions.entropy.(Distributions.MvNormal.(s.b));
    for ii = 1:length(s.b)
        gVecs[ii] = [zeros(s.b[ii].dim);exp(entropies[ii])*vec(0.5*inv(Matrix(Distributions.MvNormal(s.b[ii]).Σ)))]  # Check coefficient!
        #gVecs[ii] = [zeros(s.b[ii].dim);vec(0.5*inv(full(Distributions.MvNormal(s.b[ii]).Σ)))];
    end
    return gVecs
end


struct CostManipulate2D <: CostModel
end
function instCost(model::CostManipulate2D,b::BelMvNormal,u::MControl2D,Cs::Matrix{Float64},Cu::Matrix{Float64})
    μ,Σ = b.params[1:11],reshape(b.params[12:end],11,11); # \theta is between 0 and 2*pi;
    μ_target = [0.,0.,pi,0.,0.,0.,0.,0.,0.,0.,0.];    # Changing target \theta to pi.
    return 1/2*tr(Cs*Σ) + 1/2*(μ - μ_target)'*Cs*(μ - μ_target) + 1/2*vec(u)'*Cu*vec(u);
end
function instCost_grad(model::CostManipulate2D,b::BelMvNormal,Cs::Matrix{Float64})
    μ = b.params[1:11];
    μ_target = [0.,0.,pi,0.,0.,0.,0.,0.,0.,0.,0.];    # Changing target \theta to pi.
    return [Cs*(μ - μ_target);1/2*vec(Cs)]
end
function termCost(model::CostManipulate2D,b::BelMvNormal,Cs::Matrix{Float64})
    μ,Σ = b.params[1:11],reshape(b.params[12:end],11,11);
    μ_target = [0.,0.,pi,0.,0.,0.,0.,0.,0.,0.,0.];    # Changing target \theta to pi.
    return 1/2*tr(Cs*Σ) + 1/2*(μ - μ_target)'*Cs*(μ - μ_target);
end
termCost_grad(model::CostManipulate2D,b::BelMvNormal,Cs::Matrix{Float64}) = instCost_grad(model,b,Cs);
