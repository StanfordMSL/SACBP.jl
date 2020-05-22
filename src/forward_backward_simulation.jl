########################################
## File Name: forward_backward_simulation.jl
## Author: Haruki Nishimura (hnishimura@stanford.edu)
## Date Created: 2020/05/14
## Description: Forward Backward Simulation for SACBP
########################################

abstract type SACSimulationModel end

struct SimulatePosRangeLocalization2D <: SACSimulationModel
    continuousTransModel::TransModel_Pos
    discreteTransModel::UKFPosRange
    costModel::CostPosRangeLocalization
    dtc::Float64
    dto::Float64
    dtexec::Float64
    Q::Matrix{Float64}
    Cu::Matrix{Float64}
end
function SimulatePosRangeLocalization2D(continuousTransModel::TransModel_Pos,
                                        discreteTransModel::UKFPosRange,
                                        dtc::Float64,dto::Float64,dtexec::Float64,
                                        Q::Matrix{Float64},Cu::Matrix{Float64})
    Int64(dto/dtc);  # Inexact Error Check.
    return SimulatePosRangeLocalization2D(continuousTransModel,discreteTransModel,CostPosRangeLocalization(),dtc,dto,dtexec,Q,Cu)
end
function simulateForward(simModel::SimulatePosRangeLocalization2D,
                         s_init::AugState{<:PhysPos,<:VecBelMvNormal},
                         UArray::Vector{<:PosControl},
                         rng::MersenneTwister)
    SArray = Vector{typeof(s_init)}(undef, length(UArray)+1);
    SArray_before_update = typeof(s_init)[];
    YArray = Vector{Float64}[];
    SArray[1] = s_init;
    o_c_ratio = Int64(simModel.dto/simModel.dtc);
    kk = 0;
    for t = 1:length(UArray)
        kk += 1;
        p = trans(simModel.continuousTransModel,SArray[t].p,UArray[t],simModel.dtc);
        bVec = similar(s_init.b);
        for ii = 1:length(bVec)
            bVec[ii] = BelMvNormal(p.t,SArray[t].b[ii].params)
        end
        SArray[t+1] = AugState(p.t,p,bVec);
        if kk == o_c_ratio
            push!(SArray_before_update,SArray[t]);
            yVec,SArray[t+1] = trans(simModel.discreteTransModel,SArray[t+1],simModel.dto,simModel.Q,rng);
            push!(YArray,yVec);
            kk = 0;
        end
    end
    YArray,SArray,SArray_before_update
end
function simulateForward(simModel::SimulatePosRangeLocalization2D,
                         nominalPolicy::GradientMultiTargetLocalizationPolicy,
                         s_init::AugState{<:PhysPos,<:VecBelMvNormal},
                         u_param_min::Vector{<:Real},
                         u_param_max::Vector{<:Real},
                         UArray::Vector{<:PosControl},
                         rng::MersenneTwister)
    UArrayClosed = similar(UArray);
    SArray = Vector{typeof(s_init)}(undef, length(UArray)+1);
    SArray_before_update = typeof(s_init)[];
    YArray = Vector{Float64}[];
    SArray[1] = s_init;
    UArrayClosed[1] = control_nominal(nominalPolicy,simModel.discreteTransModel,s_init,u_param_min,u_param_max,simModel.dto,simModel.Q);
    o_c_ratio = Int64(simModel.dto/simModel.dtc);
    kk = 0;
    for t = 1:length(UArray)
        kk += 1;
        p = trans(simModel.continuousTransModel,SArray[t].p,UArrayClosed[t],simModel.dtc);
        bVec = similar(s_init.b);
        for ii = 1:length(bVec)
            bVec[ii] = BelMvNormal(p.t,SArray[t].b[ii].params)
        end
        SArray[t+1] = AugState(p.t,p,bVec);
        if t != length(UArray)
            UArrayClosed[t+1] = PosControl(UArray[t+1].t,vec(UArrayClosed[t]));
        end
        if kk == o_c_ratio
            push!(SArray_before_update,SArray[t]);
            yVec,SArray[t+1] = trans(simModel.discreteTransModel,SArray[t+1],simModel.dto,simModel.Q,rng);
            if t != length(UArray)
                UArrayClosed[t+1] = control_nominal(nominalPolicy,simModel.discreteTransModel,SArray[t+1],u_param_min,u_param_max,simModel.dto,simModel.Q);
            end
            push!(YArray,yVec);
            kk = 0;
        end
    end
    YArray,SArray,SArray_before_update,UArrayClosed
end
function simulateBackward(simModel::SimulatePosRangeLocalization2D,
                          SArray::Vector{<:AugState{<:PhysPos,<:VecBelMvNormal}},
                          SArray_before_update::Vector{<:AugState{<:PhysPos,<:VecBelMvNormal}},
                          UArray::Vector{<:PosControl},
                          YArray::Vector{Vector{Float64}})
    CoSArray = similar(SArray);
    coSP_end = PhysPos(SArray[end].t,termCost_grad_p(simModel.costModel,SArray[end]));
    coSB_end = similar(SArray[end].b);
    for ii = 1:length(coSB_end)
        coSB_end[ii] = BelMvNormal(SArray[end].t,termCost_grad_b(simModel.costModel,SArray[end])[ii]);
    end
    CoSArray[end] = AugState(coSP_end,coSB_end);
    o_c_ratio = Int64(simModel.dto/simModel.dtc);
    kk = o_c_ratio;
    ll = length(SArray_before_update);
    for t = Iterators.reverse(1:length(UArray))
        if kk == o_c_ratio
            CoSArray[t] = cotrans(simModel.discreteTransModel,simModel.costModel,CoSArray[t+1],SArray_before_update[ll],YArray[ll],simModel.dto,simModel.Q);
            ll -= 1;
            kk = 1;
        else
            CoSArray[t] = cotrans(simModel.continuousTransModel,simModel.costModel,CoSArray[t+1],SArray[t+1],UArray[t+1],simModel.dtc);
            kk += 1;
        end
    end
    return CoSArray
end

function evaluateCost(simModel::SimulatePosRangeLocalization2D,
                      s_init::AugState{<:PhysPos,<:VecBelMvNormal},
                      UArray::Vector{<:PosControl},
                      rng::MersenneTwister)
    YArray,SArray,SArray_before_update = simulateForward(simModel,s_init,UArray,rng);
    C = termCost(simModel.costModel,SArray[end]);
    for t = 1:length(SArray)-1;
        C += instCost(simModel.costModel,SArray[t],UArray[t],simModel.Cu)*simModel.dtc
    end
    return C
end


struct SimulateManipulate2D  <: SACSimulationModel
    continuousTransModel::CPredictManipulate2D
    discreteTransModel::DUpdateManipulate2D
    costModel::CostManipulate2D
    dtc::Float64       # Discrete Time Step for Continuous Simulation
    dto::Float64       # Discrete Interval for Observations.
    dtexec::Float64
    Q::Matrix{Float64} # Symmetric PSD Matrix. Transition Covariance.
    R::Matrix{Float64} # Symmetric PSD Matrix. Observation Covariance.
    Cs::Matrix{Float64} # Symmetric PSD Matrix. State Cost Coefficient Matrix.
    Cu::Matrix{Float64} # Symmetric PD Matrix. Control Cost Coefficient Matrix
end
function SimulateManipulate2D(dtc::Real,dto::Real,dtexec::Float64,
                              Q::Matrix{Float64},R::Matrix{Float64},Cs::Matrix{Float64},Cu::Matrix{Float64})
    cTransModel = CPredictManipulate2D();
    dTransModel = DUpdateManipulate2D();
    costModel = CostManipulate2D();
    Int64(dto/dtc) # Inexact Error Check.
    return SimulateManipulate2D(cTransModel,dTransModel,costModel,dtc,dto,dtexec,Q,R,Cs,Cu)
end
function simulateForward(simModel::SimulateManipulate2D,s_init::BelMvNormal,UArray::Vector{<:MControl2D},rng::MersenneTwister)
    SArray = Vector{typeof(s_init)}(undef, length(UArray)+1);
    SArray_before_update = typeof(s_init)[];
    YArray = Vector{Float64}[];
    SArray[1] = s_init;
    o_c_ratio = Int64(simModel.dto/simModel.dtc);
    kk = 0;
    for t = 1:length(UArray)
        #d = Distributions.MvNormal(SArray[t]);
        kk += 1;
        #println("In simulateForward(): t = $(SArray[t].t), $(SArray[t].params[SArray[t].dim+1:end])")
        SArray[t+1] = trans(simModel.continuousTransModel,SArray[t],UArray[t],simModel.dtc,simModel.Q);
        try d = Distributions.MvNormal(SArray[t+1])
        catch
            println("In simulateForward() 1: t = $(SArray[t+1].t), $(SArray[t+1].params[SArray[t+1].dim+1:end])")
        end
        if kk == o_c_ratio
            push!(SArray_before_update,SArray[t]);
            y,SArray[t+1] = trans(simModel.discreteTransModel,SArray[t+1],MControl2D(SArray[t+1].t,vec(UArray[t])),simModel.R,rng);
            try d = Distributions.MvNormal(SArray[t+1])
            catch
                println("In simulateForward() 2: t = $(SArray[t+1].t), $(SArray[t+1].params[SArray[t+1].dim+1:end])")
            end
            push!(YArray,y);
            kk = 0;
        end
    end
    return YArray,SArray,SArray_before_update
end
function simulateForward(simModel::SimulateManipulate2D,
                         nominalPolicy::ManipulatePositionControlPolicy,
                         s_init::BelMvNormal,
                         u_param_min::Vector{<:Real},
                         u_param_max::Vector{<:Real},
                         posGain,
                         rotGain,
                         UArray::Vector{<:MControl2D},
                         rng::MersenneTwister)
    SArray = Vector{typeof(s_init)}(undef, length(UArray)+1);
    SArray_before_update = typeof(s_init)[];
    YArray = Vector{Float64}[];
    UArrayClosed = similar(UArray);
    SArray[1] = s_init;
    UArrayClosed[1] = control_nominal(nominalPolicy,s_init,u_param_min,u_param_max,posGain,rotGain);
    o_c_ratio = Int64(simModel.dto/simModel.dtc);
    kk = 0;
    for t = 1:length(UArray)
        #d = Distributions.MvNormal(SArray[t]);
        kk += 1;
        #println("In simulateForward(): t = $(SArray[t].t), $(SArray[t].params[SArray[t].dim+1:end])")
        SArray[t+1] = trans(simModel.continuousTransModel,SArray[t],UArrayClosed[t],simModel.dtc,simModel.Q);
        if t != length(UArray)
            UArrayClosed[t+1] = MControl2D(UArray[t+1].t,vec(UArrayClosed[t]));
        end
        try d = Distributions.MvNormal(SArray[t+1])
        catch
            println("In simulateForward() 1: t = $(SArray[t+1].t), $(SArray[t+1].params[SArray[t+1].dim+1:end])")
        end
        if kk == o_c_ratio
            push!(SArray_before_update,SArray[t]);
            y,SArray[t+1] = trans(simModel.discreteTransModel,SArray[t+1],MControl2D(SArray[t+1].t,vec(UArrayClosed[t])),simModel.R,rng);
            if t != length(UArray)
                UArrayClosed[t+1] = control_nominal(nominalPolicy,SArray[t+1],u_param_min,u_param_max,posGain,rotGain);
            end
            try d = Distributions.MvNormal(SArray[t+1])
            catch
                println("In simulateForward() 2: t = $(SArray[t+1].t), $(SArray[t+1].params[SArray[t+1].dim+1:end])")
            end
            push!(YArray,y);
            kk = 0;
        end
    end
    return YArray,SArray,SArray_before_update,UArrayClosed
end
function simulateBackward(simModel::SimulateManipulate2D,SArray::VecBelMvNormal,SArray_before_update::VecBelMvNormal,
                          UArray::Vector{<:MControl2D},YArray::Vector{Vector{Float64}})
    CoSArray = similar(SArray);
    CoSArray[end] = BelMvNormal(SArray[end].t,termCost_grad(simModel.costModel,SArray[end],simModel.Cs));
    o_c_ratio = Int64(simModel.dto/simModel.dtc);
    kk = o_c_ratio;
    ll = length(SArray_before_update);
    for t = Iterators.reverse(1:length(UArray))
        if kk == o_c_ratio
            CoSArray[t] = cotrans(simModel.discreteTransModel,simModel.costModel,CoSArray[t+1],SArray_before_update[ll],UArray[t],YArray[ll],simModel.R);
            ll -= 1;
            kk = 1;
        else
            CoSArray[t] = cotrans(simModel.continuousTransModel,simModel.costModel,CoSArray[t+1],SArray[t+1],UArray[t+1],simModel.dtc,simModel.Q,simModel.Cs);
            kk += 1;
        end
    end
    return CoSArray
end

function evaluateCost(simModel::SimulateManipulate2D,
                      s_init::BelMvNormal,
                      UArray::Vector{<:MControl2D},
                      rng::MersenneTwister)
    YArray,SArray,SArray_before_update = simulateForward(simModel,s_init,UArray,rng);
    C = termCost(simModel.costModel,SArray[end],simModel.Cs);
    for t = 1:length(SArray)-1;
        C += instCost(simModel.costModel,SArray[t],UArray[t],simModel.Cs,simModel.Cu)*simModel.dtc
    end
    return C
end
