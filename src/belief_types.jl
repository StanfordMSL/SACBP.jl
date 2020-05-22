########################################
## File Name: belief_types.jl
## Author: Haruki Nishimura (hnishimura@stanford.edu)
## Date Created: 2020/05/12
## Description: Belief Type Definitions for SACBP
########################################

import Distributions: Normal, MvNormal
import Plots.Plot
using LinearAlgebra

abstract type Belief end
BelState = Union{Belief,Array{<:Belief}};

# MultiVariate Gaussian Belief.
struct BelMvNormal{T<:Real} <:Belief
    t::Float64
    params::Vector{T} #[μ;vcat(Σ...)];
    dim::Int64
    function BelMvNormal{T}(t,params,dim) where {T<:Real}
        if length(params) != dim*(dim+1)
            error(ArgumentError("Invalid parameter vector length."))
#        elseif !isposdef(reshape(params[dim+1:end],dim,dim))
#            error(ArgumentError("Σ must be positive definite."))
        else
            return new(t,params,dim)
        end
    end
end
BelMvNormal(t::Real,params::Vector{T},dim::Int64) where {T<:Real} = BelMvNormal{T}(t,params,dim);
function BelMvNormal(t::Real,params::Vector{T}) where {T<:Real}
    l = length(params)
    try
        dim = Int64((-1+sqrt(1+4l))/2) # Speculate dimension from the parameter vector length.
        return BelMvNormal(t,params,dim)
    catch
        error(ArgumentError("Invalid parameter vector length."))
    end
end
function BelMvNormal(t::Real,μ::Vector{<:Real},Σ::Matrix{<:Real})
    if size(Σ,1) != size(Σ,2) || size(Σ,2) != length(μ)
        error(ArgumentError("Invalid parameter vector length."))
    else
        return BelMvNormal(t,[μ;vcat(Σ...)],length(μ));
    end
end
function Distributions.MvNormal(b::BelMvNormal)
    μ = b.params[1:b.dim];
    Σ = reshape(b.params[b.dim+1:end],b.dim,b.dim);
    return Distributions.MvNormal(μ,Σ)
end
function plot_e_ellipse!(b::BelMvNormal,probability::Float64,plt::Plots.Plot)
    d = Distributions.MvNormal(b);
    μ = d.μ;
    Σ = Matrix(d.Σ);
    ϵ = (1. - probability)/(2*pi*sqrt(det(Σ)));
    theta = range(0.,stop=2.0*pi, length=100);
    radius = sqrt(-2.0*log(2.0*pi) - logdet(Σ) - 2*log(ϵ));
    x = zeros(length(theta));
    y = zeros(length(theta));
    for jj = 1:length(theta)
        pos = sqrt(Σ)*[radius*cos(theta[jj]);radius*sin(theta[jj])] + μ;
        x[jj] = pos[1];
        y[jj] = pos[2];
    end
    plt = plot!(x,y,color=:aquamarine, fill=true, fillalpha=0.3,label="")
end;

VecBelMvNormal{T<:Real} = Vector{BelMvNormal{T}};
function VecBelMvNormal(bvec::VecBelMvNormal)
    if !all([isequal(bvec[1].t,b.t) for b in bvec])
        error(ArgumentError("Beliefs have inconsistent time parameters."))
    else
        return bvec
    end
end
