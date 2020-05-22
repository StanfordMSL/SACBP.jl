########################################
## File Name: state_types.jl
## Author: Haruki Nishimura (hnishimura@stanford.edu)
## Date Created: 2020/05/12
## Description: State Type Definitions for SACBP
########################################

import Base.vec

abstract type PhysState end;

struct PhysPos{T<:Real} <: PhysState
    t::Float64
    pos::Vector{T} # [x1,x2,x3,...]
    dim::Int64
    function PhysPos{T}(t,pos,dim) where {T<:Real}
        if dim != length(pos)
            error(ArgumentError("Invalid position vector length."));
        else
            return new(t,pos,dim);
        end
    end
end
PhysPos(t::Real,pos::Vector{T},dim=length(pos)) where {T<:Real} = PhysPos{T}(t,pos,dim);
PhysPos(t::Real,pos::T,dim=1) where {T<:Real} = PhysPos{T}(t,[pos],dim);
vec(p::PhysPos) = p.pos;


struct PhysManipulate2D{T<:Real} <: PhysState
    t::Float64
    pos::Vector{T} # [x,y]
    θ::T # 0 <= θ < 2pi
    vel::Vector{T} # [vx,vy]
    ω::T
    m::T
    J::T
    r::Vector{T}
    μ::T
    dim::Int64
    function PhysManipulate2D{T}(t,pos,θ,vel,ω,m,J,r,μ,dim) where {T<:Real}
        if dim != 2 || length(pos) != 2 || length(vel) != 2 || length(r) != 2
            error(ArgumentError("Invalid dimension or parameter vector length."));
        else
            return new(t,pos,mod2pi(θ),vel,ω,m,J,r,μ,dim);
        end
    end
end
function PhysManipulate2D(t::Real,pos::Vector{T},θ::T,vel::Vector{T},ω::T,m::T,J::T,r::Vector{T},μ::T,dim::Int64=2) where {T<:Real}
    return PhysManipulate2D{T}(t,pos,θ,vel,ω,m,J,r,μ,dim)
end
function PhysManipulate2D(t::Real,params::Vector{T}) where {T<:Real}
    if length(params) != 11
        error(ArgumentError("Invalid parameter vector length."))
    else
        return PhysManipulate2D(t,params[1:2],params[3],params[4:5],params[6],params[7],params[8],params[9:10],params[11])
    end
end
vec(p::PhysManipulate2D) = vcat(p.pos,p.θ,p.vel,p.ω,p.m,p.J,p.r,p.μ);

struct AugState{P<:PhysState,B<:BelState}
    t::Float64
    p::P
    b::B
    function AugState{P,B}(t,p,b) where {P<:PhysState} where {B<:BelState}
        if isa(b,Belief)
            if t != p.t || t != b.t
                error(ArgumentError("PhysState and BelState have inconsistent time parameters."))
            else
                return new(t,p,b)
            end
        else
            if t != p.t || t != b[1].t
                error(ArgumentError("PhysState and BelState have inconsistent time parameters."))
            else
                return new(t,p,b)
            end
        end
    end
end
AugState(t::Real,p::P,b::B) where {P<:PhysState} where {B<:BelState} = AugState{P,B}(t,p,b);
AugState(p::P,b::B) where {P<:PhysState} where {B<:BelState} = AugState{P,B}(p.t,p,b);
