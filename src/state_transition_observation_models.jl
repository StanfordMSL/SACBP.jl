########################################
## File Name: state_transition_observation_models.jl
## Author: Haruki Nishimura (hnishimura@stanford.edu)
## Date Created: 2020/05/12
## Description: State Transition and Observation Models for SACBP
########################################

import ForwardDiff: gradient, jacobian, JacobianConfig, Chunk
using LinearAlgebra
using Random

##### State Transition Models ####
abstract type TransModel end
abstract type StateTransModel <: TransModel end

struct TransModel_Pos <: StateTransModel
# Linear state transition model for 2D position.
end
function trans(model::TransModel_Pos,x::PhysPos{T},u::PosControl{<:Real},dt::Real)::PhysPos{T} where T <: Real
# Euler Approximation.
    if x.dim != u.dim
        error(ArgumentError("PhysState and Control have inconsistent dimension values."));
    elseif x.t != u.t
        error(ArgumentError("PhysState and Control have inconsistent time parameters."));
    else
        pos = x.pos + u.vel*dt;
        t = round(x.t + dt, digits=5);
        return PhysPos(t,pos,x.dim)
    end
end
function trans(model::TransModel_Pos,x::PhysPos{T},u::PosControl{<:Real},dt::Real,
               Q::Matrix{Float64},rng::MersenneTwister)::PhysPos{T} where T <: Real
# Euler Approximation with stochastic disturbance.
    if x.dim != u.dim
        error(ArgumentError("PhysState and Control have inconsistent dimension values."));
    elseif x.t != u.t
        error(ArgumentError("PhysState and Control have inconsistent time parameters."));
    else
        pos = x.pos + u.vel*dt + rand(rng,MvNormal(zeros(x.dim),Q*dt));
        t = round(x.t + dt, digits=5);
        return PhysPos(t,pos,x.dim)
    end
end
trans_jacobi(model::TransModel_Pos,x::PhysPos,u::PosControl{<:Real}) = diagm(zeros(x.dim)); # Jacobian of dynamics.
trans_jacobi_euler(model::TransModel_Pos,x::PhysPos,u::PosControl{<:Real},dt::Real) = Matrix(1.0I, x.dim, x.dim) + trans_jacobi(model,x,u)*dt; # Jacobian of euler state transition for discrete-time EKF update..
trans_u_coeff(model::TransModel_Pos,x::PhysPos) = Matrix(1.0I, x.dim, x.dim); # Liner control coefficient for computing optimal control value.


struct TransModel_Manipulate2D <: StateTransModel
# Non-linear state transition model for planar object transport.
end
function trans(model::TransModel_Manipulate2D,x::PhysManipulate2D{T},u::MControl2D{<:Real},dt::Real)::PhysManipulate2D{T} where T <: Real
# Euler Approximation.
    if x.t != u.t
        error(ArgumentError("PhysState and Control have inconsistent time parameters."))
    else
        pos = x.pos + x.vel*dt; # position of the center of mass.
        θ   = x.θ + x.ω*dt;     # angular velocity of the center of mass
        vel = x.vel - x.μ/x.m*x.vel*dt + 1/x.m*[u.fx,u.fy]*dt;
        ω   = x.ω - 1/x.J*dot(x.r,[sin(x.θ),cos(x.θ)])*u.fx*dt + 1/x.J*dot(x.r,[cos(x.θ),-sin(x.θ)])*u.fy*dt + 1/x.J*u.tr*dt;
        m = x.m;
        J = x.J;
        r = x.r;
        μ = x.μ;
        t = round(x.t + dt, digits=5);
        return PhysManipulate2D(t,pos,θ,vel,ω,m,J,r,μ,2)
    end
end
function trans(model::TransModel_Manipulate2D,x::PhysManipulate2D{T},u::MControl2D{<:Real},dt::Real,
               Q::Matrix{Float64},rng::MersenneTwister)::PhysManipulate2D{T} where T <: Real
# Euler Approximation with stochastic disturbance.
    if x.t != u.t
        error(ArgumentError("PhysState and Control have inconsistent time parameters."))
    else
        vect = rand(rng,MvNormal(zeros(11),Q*dt));
        vect[1:2] += x.pos + x.vel*dt; # position of the center of mass.
        vect[3]   += x.θ + x.ω*dt;     # angular velocity of the center of mass
        vect[4:5] += x.vel - x.μ/x.m*x.vel*dt + 1/x.m*[u.fx,u.fy]*dt;
        vect[6]   += x.ω - 1/x.J*dot(x.r,[sin(x.θ),cos(x.θ)])*u.fx*dt + 1/x.J*dot(x.r,[cos(x.θ),-sin(x.θ)])*u.fy*dt + 1/x.J*u.tr*dt;
        vect[7]   += x.m;
        vect[8]   += x.J;
        vect[9:10] += x.r;
        vect[11]  += x.μ;
        t = round(x.t + dt, digits=5);
        return PhysManipulate2D(t,vect)
    end
end
function trans_jacobi(model::TransModel_Manipulate2D,x::PhysManipulate2D{T},u::MControl2D{<:Real}) where T <: Real
    if x.t != u.t
        error(ArgumentError("PhysState and Control have inconsistent time parameters."))
    else
        Jacobian = zeros(T,11,11);
        Jacobian[1,4] = 1.;
        Jacobian[2,5] = 1.;
        Jacobian[3,6] = 1.;
        Jacobian[4,4] = -x.μ/x.m;
        Jacobian[4,7] = x.μ/x.m^2*x.vel[1] - 1/x.m^2*u.fx;
        Jacobian[4,11]= -1/x.m*x.vel[1];
        Jacobian[5,5] = -x.μ/x.m;
        Jacobian[5,7] = x.μ/x.m^2*x.vel[2] - 1/x.m^2*u.fy;
        Jacobian[5,11] = -1/x.m*x.vel[2];
        Jacobian[6,3] = -1/x.J*dot(x.r,[cos(x.θ),-sin(x.θ)])*u.fx + 1/x.J*dot(x.r,[-sin(x.θ),-cos(x.θ)])*u.fy;
        Jacobian[6,8] = 1/x.J^2*dot(x.r,[sin(x.θ),cos(x.θ)])*u.fx - 1/x.J^2*dot(x.r,[cos(x.θ),-sin(x.θ)])*u.fy - 1/x.J^2*u.tr;
        Jacobian[6,9] = -1/x.J*sin(x.θ)*u.fx + 1/x.J*cos(x.θ)*u.fy;
        Jacobian[6,10] = -1/x.J*cos(x.θ)*u.fx - 1/x.J*sin(x.θ)*u.fy;
        return Jacobian
    end
end
function trans_jacobi_auto(model::TransModel_Manipulate2D,x::PhysManipulate2D{T},u::MControl2D{<:Real}) where T <: Real
    function f(xVec)
        x = PhysManipulate2D(x.t,xVec)
        return vec(trans(model,x,u,1.));
    end
    return ForwardDiff.jacobian(f,vec(x)) - Matrix(1.0I, 11, 11)
end
function trans_jacobi_euler(model::TransModel_Manipulate2D,x::PhysManipulate2D{T},u::MControl2D{<:Real},dt::Real) where T <: Real
    return Matrix{T}(1.0I, 11, 11) + trans_jacobi(model,x,u)*dt;
end
function trans_u_coeff(model::TransModel_Manipulate2D,x::PhysManipulate2D{T}) where T <: Real
    H = zeros(T,11,3);
    H[4,1] = 1/x.m;
    H[5,2] = 1/x.m;
    H[6,1] = -1/x.J*dot(x.r,[sin(x.θ),cos(x.θ)]);
    H[6,2] = 1/x.J*dot(x.r,[cos(x.θ),-sin(x.θ)]);
    H[6,3] = 1/x.J;
    return H
end


#### Observation Models ####
abstract type ObserveModel end

struct ObserveModel_Range <: ObserveModel
# Range-only measurements from relative positions.
end
function observe(model::ObserveModel_Range,p_robot::PhysPos,q_pos::Vector{<:Real})
    # Noiseless Observation
    return norm(p_robot.pos - q_pos)
end
function observe(model::ObserveModel_Range,p_robot::PhysPos,q_pos::Vector{<:Real},
                 R::Matrix{Float64},rng::MersenneTwister)
    # Observation with non-additive Gaussian noise.
    return norm(p_robot.pos - q_pos - rand(rng,MvNormal(zeros(p_robot.dim),R)))
end
function observe_jacobi(model::ObserveModel_Range,p_robot::PhysPos{T},q_pos::Vector{<:Real}) where T <: Real
    return transpose(q_pos - p_robot.pos)./(norm(q_pos - p_robot.pos) + 1e-15); # Prevent singularity.
end
function observe_jacobi_auto(model::ObserveModel_Range,p_robot::PhysPos{T},q_pos::Vector{<:Real}) where T <: Real
    return ForwardDiff.gradient(q_pos -> observe(model,p_robot,q_pos), q_pos)'
end


struct ObserveModel_Manipulate2D <: ObserveModel
end
function observe(model::ObserveModel_Manipulate2D,x::PhysManipulate2D{T},u::MControl2D{<:Real})::Vector{T} where T <: Real
    if x.t != u.t
        error(ArgumentError("PhysState and Control have inconsistent time parameters."))
    else
        px_robot = x.pos[1] + x.r[1]*cos(x.θ) - x.r[2]*sin(x.θ);
        py_robot = x.pos[2] + x.r[1]*sin(x.θ) + x.r[2]*cos(x.θ);
        θ = x.θ;
        vx_robot = x.vel[1] - x.ω*(x.r[1]*sin(x.θ) + x.r[2]*cos(x.θ));
        vy_robot = x.vel[2] + x.ω*(x.r[1]*cos(x.θ) - x.r[2]*sin(x.θ));
        ω = x.ω;
        α_robot = 1/x.J*u.tr - 1/x.J*(x.r[1]*sin(x.θ) + x.r[2]*cos(x.θ))*u.fx + 1/x.J*(x.r[1]*cos(x.θ) - x.r[2]*sin(x.θ))*u.fy;
        ax_robot = 1/x.m*(u.fx - x.μ*x.vel[1]) - α_robot*(x.r[1]*sin(x.θ) + x.r[2]*cos(x.θ)) - x.ω^2*(x.r[1]*cos(x.θ) - x.r[2]*sin(x.θ));
        ay_robot = 1/x.m*(u.fy - x.μ*x.vel[2]) + α_robot*(x.r[1]*cos(x.θ) - x.r[2]*sin(x.θ)) - x.ω^2*(x.r[1]*sin(x.θ) + x.r[2]*cos(x.θ));
        return [px_robot,py_robot,θ,vx_robot,vy_robot,ω,ax_robot,ay_robot,α_robot]
    end
end
function observe(model::ObserveModel_Manipulate2D,x::PhysManipulate2D{T},u::MControl2D{<:Real},
                 R::Matrix{Float64},rng::MersenneTwister)::Vector{T} where T <: Real
    obs = observe(model,x,u);
    obs =  obs + rand(rng,MvNormal(zeros(length(obs)),R));
    return obs
end
function observe_jacobi(model::ObserveModel_Manipulate2D,x::PhysManipulate2D{T},u::MControl2D) where T <: Real
    if x.t != u.t
        error(ArgumentError("PhysState and Control have inconsistent time parameters."))
    else
        Jacobian = zeros(T,9,11);
        Jacobian[1,1] = 1.;
        Jacobian[1,3] = x.r[1]*(-sin(x.θ)) - x.r[2]*cos(x.θ);
        Jacobian[1,9] = cos(x.θ);
        Jacobian[1,10] = -sin(x.θ);
        Jacobian[2,2] = 1.;
        Jacobian[2,3] = x.r[1]*cos(x.θ) + x.r[2]*(-sin(x.θ));
        Jacobian[2,9] = sin(x.θ);
        Jacobian[2,10] = cos(x.θ);
        Jacobian[3,3] = 1.;
        Jacobian[4,3] = -x.ω*x.r[1]*cos(x.θ) - x.ω*x.r[2]*(-sin(x.θ));
        Jacobian[4,4] = 1.;
        Jacobian[4,6] = -x.r[1]*sin(x.θ) - x.r[2]*cos(x.θ);
        Jacobian[4,9] = -x.ω*sin(x.θ);
        Jacobian[4,10] = -x.ω*cos(x.θ);
        Jacobian[5,3] = x.ω*x.r[1]*(-sin(x.θ)) - x.ω*x.r[2]*cos(x.θ);
        Jacobian[5,5] = 1.;
        Jacobian[5,6] = x.r[1]*cos(x.θ) - x.r[2]*sin(x.θ);
        Jacobian[5,9] = x.ω*cos(x.θ);
        Jacobian[5,10] = -x.ω*sin(x.θ);
        Jacobian[6,6] = 1.;
        Jacobian[9,3] = -1/x.J*(x.r[1]*cos(x.θ) + x.r[2]*(-sin(x.θ)))*u.fx + 1/x.J*(x.r[1]*(-sin(x.θ)) - x.r[2]*cos(x.θ))*u.fy;
        Jacobian[9,8] = -1/x.J^2*u.tr + 1/x.J^2*(x.r[1]*sin(x.θ) + x.r[2]*cos(x.θ))*u.fx - 1/x.J^2*(x.r[1]*cos(x.θ) - x.r[2]*sin(x.θ))*u.fy;
        Jacobian[9,9] = -1/x.J*sin(x.θ)*u.fx + 1/x.J*cos(x.θ)*u.fy;
        Jacobian[9,10] = -1/x.J*cos(x.θ)*u.fx - 1/x.J*sin(x.θ)*u.fy;
        α_robot = 1/x.J*u.tr - 1/x.J*(x.r[1]*sin(x.θ) + x.r[2]*cos(x.θ))*u.fx + 1/x.J*(x.r[1]*cos(x.θ) - x.r[2]*sin(x.θ))*u.fy;
        Jacobian[7,3] = -Jacobian[9,3]*(x.r[1]*sin(x.θ) + x.r[2]*cos(x.θ)) - α_robot*(x.r[1]*cos(x.θ) + x.r[2]*(-sin(x.θ))) - x.ω^2*(x.r[1]*(-sin(x.θ)) - x.r[2]*cos(x.θ));
        Jacobian[7,4] = -x.μ/x.m;
        Jacobian[7,6] = -2*x.ω*(x.r[1]*cos(x.θ) - x.r[2]*sin(x.θ));
        Jacobian[7,7] = -1/x.m^2*(u.fx - x.μ*x.vel[1]);
        Jacobian[7,8] = -Jacobian[9,8]*(x.r[1]*sin(x.θ) + x.r[2]*cos(x.θ));
        Jacobian[7,9] = -Jacobian[9,9]*(x.r[1]*sin(x.θ) + x.r[2]*cos(x.θ)) - α_robot*sin(x.θ) - x.ω^2*cos(x.θ);
        Jacobian[7,10] = -Jacobian[9,10]*(x.r[1]*sin(x.θ) + x.r[2]*cos(x.θ)) - α_robot*cos(x.θ) + x.ω^2*sin(x.θ);
        Jacobian[7,11] = -1/x.m*x.vel[1];
        Jacobian[8,3] = Jacobian[9,3]*(x.r[1]*cos(x.θ) - x.r[2]*sin(x.θ)) + α_robot*(x.r[1]*(-sin(x.θ)) - x.r[2]*cos(x.θ)) - x.ω^2*(x.r[1]*cos(x.θ) + x.r[2]*(-sin(x.θ)));
        Jacobian[8,5] = -x.μ/x.m;
        Jacobian[8,6] = -2*x.ω*(x.r[1]*sin(x.θ) + x.r[2]*cos(x.θ));
        Jacobian[8,7] = -1/x.m^2*(u.fy - x.μ*x.vel[2]);
        Jacobian[8,8] = Jacobian[9,8]*(x.r[1]*cos(x.θ) - x.r[2]*sin(x.θ));
        Jacobian[8,9] = Jacobian[9,9]*(x.r[1]*cos(x.θ) - x.r[2]*sin(x.θ)) + α_robot*cos(x.θ) - x.ω^2*sin(x.θ);
        Jacobian[8,10] = Jacobian[9,10]*(x.r[1]*cos(x.θ) - x.r[2]*sin(x.θ)) + α_robot*(-sin(x.θ)) - x.ω^2*cos(x.θ);
        Jacobian[8,11] = -1/x.m*x.vel[2];
        return Jacobian
    end
end
function observe_jacobi_auto(model::ObserveModel_Manipulate2D,x::PhysManipulate2D{T},u::MControl2D{<:Real}) where T <: Real
    function f(xVec)
        x = PhysManipulate2D(x.t,xVec);
        return vec(observe(model,x,u));
    end
    return ForwardDiff.jacobian(f,vec(x))
end
