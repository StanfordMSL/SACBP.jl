########################################
## File Name: belief_transition_models.jl
## Author: Haruki Nishimura (hnishimura@stanford.edu)
## Date Created: 2020/05/12
## Description: Belief Transition Models for SACBP
########################################

import Plots
using LinearAlgebra

abstract type BeliefTransModel <: TransModel end
abstract type GaussianFilter <: BeliefTransModel end
# abstract type DiscreteBayesianFilter <: BeliefTransModel end


function covObservePos(p::Vector{<:Real},qi::Vector{<:Real}) # State-dependent observation noise model used in multi-target localization problem.
    R_scale = 0.01*Matrix(1.0I, 2, 2);
    R_nominal = 0.001*Matrix(1.0I, 2, 2);
    dist = norm(p-qi);
    return R_nominal + dist*R_scale;
end


struct UKFPosRange <: GaussianFilter
# Discrete-time EKF model for belief transition in position estimation with range-only measurements.
end
function trans(model::UKFPosRange,b::BelMvNormal{T},p_robot::PhysPos{U},dt::Real,y::Real,
               Q::Matrix{<:Real}) where T <: Real where U <: Real
    d = Distributions.MvNormal(b);
    μ,Σ = d.μ, Matrix(d.Σ);

    # Predict.
    μ = μ;
    Σ = Σ + Q*dt;

    # Generate Sigma Points.
    n = 2 + 2;
    λ = 2.0;
    x0 = [μ;zeros(2)];
    x_sigma = [];
    push!(x_sigma,x0);
    sqrtm_P = Matrix(cholesky([Σ          zeros(2,2);
                               zeros(2,2) covObservePos(p_robot.pos,μ)]).L);
    for ii = 1:n
        push!(x_sigma,x0 + sqrt(n+λ)*sqrtm_P[:,ii])
    end
    for ii = 1:n
        push!(x_sigma,x0 - sqrt(n+λ)*sqrtm_P[:,ii])
    end
    w_sigma = 1/(2*(n+λ))*ones(2*n+1);
    w_sigma[1] = λ/(n+λ);
    y_sigma = Vector{Real}(undef, 2*n+1)
    for ii = 1:2*n+1
        y_sigma[ii] = norm(p_robot.pos - x_sigma[ii][1:2] + x_sigma[ii][3:4]);
    end
    y_pred_mean = dot(w_sigma,y_sigma);

    # Update
    σ_YY = dot(w_sigma,(y_sigma .- y_pred_mean).^2);
    Σ_XY = w_sigma[1]*(x_sigma[1][1:2] - μ)*(y_sigma[1] - y_pred_mean);
    for ii = 1:2*n
        Σ_XY += w_sigma[ii+1]*(x_sigma[ii+1][1:2] - μ)*(y_sigma[ii+1] - y_pred_mean);
    end
    K = 1/σ_YY*Σ_XY;
    μ = μ + K*(y - y_pred_mean);
    Σ = Σ - K*Σ_XY';
    Σ = (Σ + Σ')/2;
    return BelMvNormal(p_robot.t,μ,Σ)
end
function trans_jacobi_auto_p(model::UKFPosRange,b::BelMvNormal{T},p_robot::PhysPos{U},
                             dt::Real,y::Real,Q::Matrix{<:Real}) where T <: Real where U <: Real
    function f(pVec)
        # if isa(p_robot,PhysPos)
            return trans(model,b,PhysPos(p_robot.t,pVec),dt,y,Q).params
        # else
        #     return trans(model,b,PhysPosHead2D(p_robot.t,pVec),dt,y,Q).params
        # end
    end
    cfg = ForwardDiff.JacobianConfig(f,vec(p_robot),ForwardDiff.Chunk{2}());
    return ForwardDiff.jacobian(f,vec(p_robot),cfg)
    #return ForwardDiff.jacobian(f,vec(p_robot))
end
function trans_jacobi_auto_b(model::UKFPosRange,b::BelMvNormal{T},p_robot::PhysPos{U},
                             dt::Real,y::Real,Q::Matrix{<:Real}) where T <: Real where U <: Real
    function f(bVec)
        b1 = BelMvNormal(b.t,bVec)
        return trans(model,b1,p_robot,dt,y,Q).params
    end
    cfg = ForwardDiff.JacobianConfig(f,b.params,ForwardDiff.Chunk{6}());
    return ForwardDiff.jacobian(f,b.params,cfg)
    #return ForwardDiff.jacobian(f,b.params)
end
function trans(model::UKFPosRange,s::AugState{PhysPos{U},BelMvNormal{T}},dt::Real,y::Real,
               Q::Matrix{<:Real})::AugState{PhysPos{U},BelMvNormal{T}} where T <: Real where U <: Real
    b_new = trans(model,s.b,s.p,dt,y,Q);
    return AugState(s.p,b_new)
end
function trans_jacobi_auto_p(model::UKFPosRange,s::AugState{PhysPos{U},BelMvNormal{T}},
                             dt::Real,y::Real,Q::Matrix{<:Real}) where T <: Real where U <: Real
    return trans_jacobi_auto_p(model,s.b,s.p,dt,y,Q);
end
function trans_jacobi_auto_b(model::UKFPosRange,s::AugState{PhysPos{U},BelMvNormal{T}},
                             dt::Real,y::Real,Q::Matrix{<:Real}) where T <: Real where U <: Real
    return trans_jacobi_auto_b(model,s.b,s.p,dt,y,Q);
end
function trans(model::UKFPosRange,bvec::VecBelMvNormal{T},p_robot::PhysPos{U},dt::Real,yvec::Vector{<:Real},
               Q::Matrix{<:Real})::VecBelMvNormal{T} where T <: Real where U <: Real
    if length(bvec) != length(yvec)
        error(ArgumentError("Invalid number of observations."))
    else
        bvecNew = similar(bvec);
        for ii = 1:length(bvec)
            bvecNew[ii] = trans(model,bvec[ii],p_robot,dt,yvec[ii],Q)
        end
        return bvecNew
    end
end
function trans_jacobi_auto_p(model::UKFPosRange,bvec::VecBelMvNormal{T},
                             p_robot::PhysPos{U},dt::Real,yvec::Vector{<:Real},Q::Matrix{<:Real}) where T <: Real where U <: Real
    JArray = Vector{Matrix{T}}(undef, length(bvec));
    for ii = 1:length(bvec)
        JArray[ii] = trans_jacobi_auto_p(model,bvec[ii],p_robot,dt,yvec[ii],Q);
    end
    return JArray
end
function trans_jacobi_auto_b(model::UKFPosRange,bvec::VecBelMvNormal{T},
                             p_robot::PhysPos{U},dt::Real,yvec::Vector{<:Real},Q::Matrix{<:Real}) where T <: Real where U <: Real
    JArray = Vector{Matrix{T}}(undef, length(bvec));
    for ii = 1:length(bvec)
        JArray[ii] = trans_jacobi_auto_b(model,bvec[ii],p_robot,dt,yvec[ii],Q)
    end
    return JArray
end
function trans(model::UKFPosRange,s::AugState{PhysPos{U},VecBelMvNormal{T}},dt::Real,yvec::Vector{<:Real},
               Q::Matrix{<:Real})::AugState{PhysPos{U},VecBelMvNormal{T}} where T <: Real where U <: Real
    if length(s.b) != length(yvec)
        error(ArgumentError("Invalid number of observations."))
    else
        bvecNew = similar(s.b);
        for ii = 1:length(s.b)
            bvecNew[ii] = trans(model,s.b[ii],s.p,dt,yvec[ii],Q);
        end
        return AugState(s.p,bvecNew)
    end
end
function trans_jacobi_auto_p(model::UKFPosRange,s::AugState{PhysPos{U},VecBelMvNormal{T}},
                             dt::Real,yvec::Vector{<:Real},Q::Matrix{<:Real}) where T <: Real where U <: Real
    JArray = Vector{Matrix{T}}(undef, length(s.b));
    for ii = 1:length(s.b)
        JArray[ii] = trans_jacobi_auto_p(model,s.b[ii],s.p,dt,yvec[ii],Q);
    end
    return JArray
end
function trans_jacobi_auto_b(model::UKFPosRange,s::AugState{PhysPos{U},VecBelMvNormal{T}},
                             dt::Real,yvec::Vector{<:Real},Q::Matrix{<:Real}) where T <: Real where U <: Real
    JArray = Vector{Matrix{T}}(undef, length(s.b));
    for ii = 1:length(s.b)
        JArray[ii] = trans_jacobi_auto_b(model,s.b[ii],s.p,dt,yvec[ii],Q);
    end
    return JArray
end
function trans(model::UKFPosRange,b::BelMvNormal{T},p_robot::PhysPos{U},dt::Real,
               Q::Matrix{<:Real},rng::MersenneTwister)::Tuple{Real,BelMvNormal{T}} where T <: Real where U <: Real
    d = Distributions.MvNormal(b);
    μ,Σ = d.μ,Matrix(d.Σ);

    # Predict.
    μ = μ;
    Σ = Σ + Q*dt;

    # Generate Sigma Points.
    n = 2 + 2;
    λ = 2.0;
    x0 = [μ;zeros(2)];
    x_sigma = [];
    push!(x_sigma,x0);
    sqrtm_P = Matrix(cholesky([Σ          zeros(2,2);
                               zeros(2,2) covObservePos(p_robot.pos,μ)]).L);
    for ii = 1:n
        push!(x_sigma,x0 + sqrt(n+λ)*sqrtm_P[:,ii])
    end
    for ii = 1:n
        push!(x_sigma,x0 - sqrt(n+λ)*sqrtm_P[:,ii])
    end
    w_sigma = 1/(2*(n+λ))*ones(2*n+1);
    w_sigma[1] = λ/(n+λ);
    y_sigma = Vector{Real}(undef, 2*n+1)
    for ii = 1:2*n+1
        y_sigma[ii] = norm(p_robot.pos - x_sigma[ii][1:2] + x_sigma[ii][3:4]);
    end
    y_pred_mean = dot(w_sigma,y_sigma);

    # Update
    σ_YY = dot(w_sigma,(y_sigma .- y_pred_mean).^2);
    Σ_XY = w_sigma[1]*(x_sigma[1][1:2] - μ)*(y_sigma[1] - y_pred_mean);
    for ii = 1:2*n
        Σ_XY += w_sigma[ii+1]*(x_sigma[ii+1][1:2] - μ)*(y_sigma[ii+1] - y_pred_mean);
    end
    K = 1/σ_YY*Σ_XY;
    observeModel = ObserveModel_Range();
    y = observe(observeModel,p_robot,μ,Σ+covObservePos(p_robot.pos,μ),rng);
    μ = μ + K*(y - y_pred_mean);
    Σ = Σ - K*Σ_XY';
    Σ = (Σ + Σ')/2;
    return y,BelMvNormal(p_robot.t,μ,Σ)
end
function trans(model::UKFPosRange,s::AugState{PhysPos{U},BelMvNormal{T}},dt::Real,
               Q::Matrix{<:Real},rng::MersenneTwister)::Tuple{Real,AugState{PhysPos{U},BelMvNormal{T}}} where T <: Real where U <: Real
    y,b_new = trans(model,s.b,s.p,dt,Q,rng);
    return y,AugState(s.p,b_new)
end
function trans(model::UKFPosRange,bvec::VecBelMvNormal{T},p_robot::PhysPos{U},dt::Real,
               Q::Matrix{<:Real},rng::MersenneTwister)::Tuple{Vector{<:Real},VecBelMvNormal{T}} where T <: Real where U <: Real
    temp = map(b -> trans(model,b,p_robot,dt,Q,rng),bvec)
    yvec = Vector{Float64}(undef, length(bvec));
    bvecNew = similar(bvec);
    for ii = 1:length(bvec)
        yvec[ii] = temp[ii][1];
        bvecNew[ii] = temp[ii][2];
    end
    return yvec,bvecNew
end
function trans(model::UKFPosRange,s::AugState{PhysPos{U},VecBelMvNormal{T}},dt::Real,
               Q::Matrix{<:Real},rng::MersenneTwister)::Tuple{Vector{<:Real},AugState{PhysPos{U},VecBelMvNormal{T}}} where T <: Real where U <: Real
    temp = map(b -> trans(model,b,s.p,dt,Q,rng),s.b)
    yvec = Vector{Float64}(undef, length(s.b))
    bvecNew = similar(s.b);
    for ii = 1:length(s.b)
        yvec[ii] = temp[ii][1];
        bvecNew[ii] = temp[ii][2];
    end
    sNew = AugState(s.p,bvecNew);
    return yvec,sNew
end


struct CPredictManipulate2D <: GaussianFilter
# Linearized Continuous Gaussian Prediction model for belief transition in planar manipulation problem.
end
function trans(model::CPredictManipulate2D,b::BelMvNormal{T},u::MControl2D{<:Real},dt::Real,
               Q::Matrix{Float64},mode=1)::BelMvNormal{T} where T <: Real
# Continuous Time Transition. Euler Approximation.
    if b.t != u.t
        error(ArgumentError("BelState and Control have inconsistent time parameters."))
    else
        if mode!=1  # mode n != 1 is with naive euler approximation (for automatic jacobian computation with dt == 1.)
            transModel = TransModel_Manipulate2D();
            d = Distributions.MvNormal(b);
            μ,Σ = d.μ,Matrix(d.Σ);
            μMP = PhysManipulate2D(b.t,μ)
            A = trans_jacobi(transModel,μMP,u);
            Σ = Σ + (A*Σ + Σ*A' + Q)*dt; #This Euler Approximation is numerically unstable for large dt (e.g. dt == 0.01). Using Square-Root form.
            μ = vec(trans(transModel,PhysManipulate2D(b.t,μ),u,dt));
            t = round(b.t + dt,digits=5);
            return BelMvNormal(t,μ,Σ)
        else #mode 1 is with square-root form update.
            #println("In trans(): t = $(b.t), $(b.params[b.dim+1:end])")
            transModel = TransModel_Manipulate2D();
            try d = Distributions.MvNormal(b);
            catch
                # This block is solely for debugging and needs to be deleted before release.
                println("In trans(): t = $(b.t), μ = $(b.params[1:b.dim]),Σ = $(b.params[b.dim+1:end])")
                d = Distributions.MvNormal(b.params[1:b.dim],reshape(b.params[b.dim+1:end],11,11)+1e-5*Matrix(1.0I, 11, 11))
            end
            #d = Distributions.MvNormal(b);
            μ,Σ = d.μ,Matrix(d.Σ);
            μMP = PhysManipulate2D(b.t,μ)
            A = trans_jacobi(transModel,μMP,u);
            L = cholesky(Σ).L;
            L_inv = inv(L);
            L = L + (A*L + 1/2*Q*L_inv)*dt;
            Σ = Matrix(L*L');
            #A_discrete_approx = expm(A*dt);
            #Σ = A_discrete_approx*Σ*A_discrete_approx' + Q*dt; # Use Matrix Exponential and treat like a discrete-time system.
            #Σ = (Σ + Σ')/2;
            μ = vec(trans(transModel,PhysManipulate2D(b.t,μ),u,dt));
            #if μ[7] <= 0. || μ[8] <= 0. || μ[11] <= 0.
            #    println(μ)
            #end
            t = round(b.t + dt,digits=5);
            return BelMvNormal(t,μ,Σ)
        end
    end
end
function trans_jacobi(model::CPredictManipulate2D,b::BelMvNormal{T},u::MControl2D{<:Real},Q::Matrix{Float64}) where T <: Real
    function f(μ)
        d = Distributions.MvNormal(b);
        Σ = Matrix(d.Σ);
        A = trans_jacobi(TransModel_Manipulate2D(),PhysManipulate2D(b.t,μ),u);
        return vcat(A*Σ + Σ*A' + Q...);
    end
    Jacobian = zeros(T,11^2+11,11^2+11);
    A = trans_jacobi(TransModel_Manipulate2D(),PhysManipulate2D(b.t,b.params[1:11]),u);
    Jacobian[1:11,1:11] = A;
    for ii = 1:11
        Jacobian[12+11*(ii-1):11+11*ii,12+11*(ii-1):11+11*ii] += A;
        for jj = 1:11
            Jacobian[12+11*(ii-1):11+11*ii,12+11*(jj-1):11+11*jj] += diagm(A[ii,jj]*ones(11));
        end
    end
    cfg = ForwardDiff.JacobianConfig(f,b.params[1:11],ForwardDiff.Chunk{11}())
    Jacobian[12:end,1:11] = ForwardDiff.jacobian(f,b.params[1:11],cfg);
    return Jacobian
end
function trans_jacobi_auto(model::CPredictManipulate2D,b::BelMvNormal{T},u::MControl2D{<:Real},Q::Matrix{Float64}) where T <: Real
    function f(params)
        bel = BelMvNormal(b.t,params);
        return trans(model,bel,u,1.,Q,0).params - params;
    end
    cfg = ForwardDiff.JacobianConfig(f,b.params,ForwardDiff.Chunk{40}());
    return ForwardDiff.jacobian(f,b.params,cfg)
    #return ForwardDiff.jacobian(f,b.params)
end
function trans_u_coeff(model::CPredictManipulate2D,b::BelMvNormal{T},Q::Matrix{Float64}) where T <: Real
    uVecs = [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]];
    u0 = MControl2D(b.t,0.,0.,0.);
    H = Matrix{T}(undef,length(b.params),3);
    transModel = CPredictManipulate2D();
    for ii = 1:3
        u = MControl2D(b.t,uVecs[ii]);
        H[:,ii] = trans(transModel,b,u,1.,Q,0).params - trans(transModel,b,u0,1.,Q,0).params
    end
    return H
end
struct DUpdateManipulate2D <: GaussianFilter
# Discrete-time EKF Update model for belief transition in planar manipulation problem.
end
function trans(model::DUpdateManipulate2D,b::BelMvNormal{T},u::MControl2D{<:Real},y::Vector{<:Real},
               R::Matrix{Float64})::BelMvNormal{T} where T <: Real
# Instantaneous Transition (EKF Update Step)
    if b.t != u.t
        error(ArgumentError("BelState and Control have inconsistent time parameters."))
    else
        observeModel = ObserveModel_Manipulate2D();
        d = Distributions.MvNormal(b);
        μ,Σ = d.μ,Matrix(d.Σ);
        μMP = PhysManipulate2D(b.t,μ);
        C = observe_jacobi(observeModel,μMP,u);
        K = Σ*C'/(C*Σ*C' + R);
        y_pred_mean = observe(observeModel,μMP,u);
        if y[3] - y_pred_mean[3] > pi   # Correct angular difference.
            y[3] -= 2*pi;
        elseif y[3] - y_pred_mean[3] < -pi
            y[3] += 2*pi;
        end
        μ += K*(y - y_pred_mean);
        μ[3] = mod2pi(μ[3]);
        Σ -= K*C*Σ;
        Σ = (Σ + Σ')/2;
        t = b.t;
        return BelMvNormal(t,μ,Σ)
    end
end
function trans(model::DUpdateManipulate2D,b::BelMvNormal{T},u::MControl2D{<:Real},
               R::Matrix{Float64},rng::MersenneTwister)::Tuple{Vector{<:Real},BelMvNormal{T}} where T <: Real
    # Instantaneous Transition with sampled observation. (EKF Update Step)
    if b.t != u.t
        error(ArgumentError("BelState and Control have inconsistent time parameters."))
    else
        observeModel = ObserveModel_Manipulate2D();
        d = Distributions.MvNormal(b);
        μ,Σ = d.μ,Matrix(d.Σ);
        μMP = PhysManipulate2D(b.t,μ);
        C = observe_jacobi(observeModel,μMP,u);
        H = C*Σ*C' + R;
        H = (H+H')/2;
        K = Σ*C'/H;
        y = observe(observeModel,μMP,u,H,rng);
        y_pred_mean = observe(observeModel,μMP,u);
        if y[3] - y_pred_mean[3] > pi   # Correct angular difference.
            y[3] -= 2*pi;
        elseif y[3] - y_pred_mean[3] < -pi
            y[3] += 2*pi;
        end
        μ += K*(y - y_pred_mean);
        μ[3] = mod2pi(μ[3]);
        U = cholesky((H+H')/2).L; # Square-root form (Andrews, 1968)
        V = cholesky(R).L;
        L = cholesky(Σ).L;
        L -= Σ*C'/((U+V)*U')*C*L;
        Σ = Matrix(L*L');
        #Σ -= K*C*Σ;
        #Σ = (Σ + Σ')/2;
        #try MvNormal(zeros(11),Σ)
        #catch
        #    Σ += eye(11)*2*minimum(eig(Σ)[1]);
        #end
        t = b.t;
        return y,BelMvNormal(t,μ,Σ)
    end
end
function trans_jacobi_auto(model::DUpdateManipulate2D,b::BelMvNormal{T},u::MControl2D{<:Real},y::Vector{<:Real},R::Matrix{Float64}) where T <: Real
    function f(params)
        bel = BelMvNormal(b.t,params);
        return trans(model,bel,u,y,R).params
    end
    cfg = ForwardDiff.JacobianConfig(f,b.params,ForwardDiff.Chunk{40}());
    return ForwardDiff.jacobian(f,b.params,cfg)
    #return ForwardDiff.jacobian(f,b.params)
end



struct CDEKFManipulate2D <: GaussianFilter
# Continuous-Discrete EKF model for belief transition in planar manipulation problem.
end
function trans(model::CDEKFManipulate2D,b::BelMvNormal{T},u::MControl2D{<:Real},dtc::Real,dto::Real,y::Vector{<:Real},
               Q::Matrix{Float64},R::Matrix{Float64})::BelMvNormal{T} where T <: Real
    if b.t != u.t
        error(ArgumentError("BelState and Control have inconsistent time parameters."))
    else
        predictModel = CPredictManipulate2D();
        for ii = 1:Int64(dto/dtc)
            b = trans(predictModel,b,MControl2D(b.t,u.fx,u.fy,u.tr),dtc,Q);
        end
        updateModel = DUpdateManipulate2D();
        b = trans(updateModel,b,MControl2D(b.t,u.fx,u.fy,u.tr),y,R);
        return b
    end
end
function trans(model::CDEKFManipulate2D,b::BelMvNormal{T},u::MControl2D{<:Real},dtc::Real,dto::Real,
               Q::Matrix{Float64},R::Matrix{Float64},rng::MersenneTwister)::Tuple{Vector{<:Real},BelMvNormal{T}} where T <: Real
    if b.t != u.t
        error(ArgumentError("BelState and Control have inconsistent time parameters."))
    else
        predictModel = CPredictManipulate2D();
        for ii = 1:Int64(dto/dtc)
            b = trans(predictModel,b,MControl2D(b.t,u.fx,u.fy,u.tr),dtc,Q);
        end
        updateModel = DUpdateManipulate2D();
        y,b = trans(updateModel,b,MControl2D(b.t,u.fx,u.fy,u.tr),R,rng);
        return y,b
    end
end
