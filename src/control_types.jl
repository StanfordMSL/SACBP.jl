########################################
## File Name: control_types.jl
## Author: Haruki Nishimura (hnishimura@stanford.edu)
## Date Created: 2020/05/12
## Description: Control Type Definitions for SACBP
########################################

import Base.vec

abstract type Control end

struct PosControl{T<:Real} <: Control
    t::Float64
    vel::Vector{T}
    dim::Int64
    function PosControl{T}(t,vel,dim) where {T<:Real}
        if dim != length(vel)
            error(ArgumentError("Invalid control vector length."));
        else
            return new(t,vel,dim);
        end
    end
end
PosControl(t::Real,vel::Vector{T},dim::Int64=length(vel)) where {T<:Real} = PosControl{T}(t,vel,dim);
PosControl(t::Real,vel::T) where {T<:Real} = PosControl{T}(t,[vel],1);
vec(u::PosControl) = u.vel;

struct PosHeadControl2D{T<:Real} <: Control
    t::Float64
    vel::T
    ω::T
end

struct MControl2D{T<:Real} <: Control
    t::Float64
    fx::T  # Force x
    fy::T  # Force y
    tr::T  # Torque
end
function MControl2D(t::Real,params::Vector{T}) where {T<:Real}
    if length(params) != 3
        error(ArgumentError("Invalid parameter vector length."))
    else
        return MControl2D{T}(t,params[1],params[2],params[3]);
    end
end
vec(u::MControl2D) = [u.fx,u.fy,u.tr];
