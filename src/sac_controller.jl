########################################
## File Name: sac_controller.jl
## Author: Haruki Nishimura (hnishimura@stanford.edu)
## Date Created: 2020/05/18
## Description: SAC Controller for SACBP
########################################

using Convex
# using SCS
using ECOS
import Future
using Distributed

function getControlCoeffs(simModel::SimulatePosRangeLocalization2D,
                          s_init::AugState{<:PhysPos,<:VecBelMvNormal},
                          UArray::Vector{<:PosControl},
                          rng::MersenneTwister)
    YArray,SArray,SArray_before_update = simulateForward(simModel,s_init,UArray,rng);
    CoSArray = simulateBackward(simModel,SArray,SArray_before_update,UArray,YArray);
    CoSParams = map(CoS -> vec(CoS.p), CoSArray);
    HArray = map(s -> trans_u_coeff(simModel.continuousTransModel,s.p), SArray);
    CoeffArray = similar(CoSParams);
    for ii = 1:length(CoeffArray)
        CoeffArray[ii] = HArray[ii]'*CoSParams[ii];
    end
    #if myid() == 5
    #    println(vec(CoSArray[1].p)[1])
    #end
    return CoeffArray;
end
function getControlCoeffs(simModel::SimulatePosRangeLocalization2D,
                          nominalPolicy::GradientMultiTargetLocalizationPolicy,
                          s_init::AugState{<:PhysPos,<:VecBelMvNormal},
                          u_param_min::Vector{<:Real},
                          u_param_max::Vector{<:Real},
                          UArray::Vector{<:PosControl},
                          rng::MersenneTwister)
    YArray,SArray,SArray_before_update,UArrayClosed = simulateForward(simModel,nominalPolicy,s_init,u_param_min,u_param_max,UArray,rng);
    CoSArray = simulateBackward(simModel,SArray,SArray_before_update,UArrayClosed,YArray);
    CoSParams = map(CoS -> vec(CoS.p), CoSArray);
    HArray = map(s -> trans_u_coeff(simModel.continuousTransModel,s.p), SArray);
    CoeffArray = similar(CoSParams);
    NominalControlCostArray = Vector{Float64}(undef,length(CoeffArray)-1);
    for ii = 1:length(CoeffArray)
        CoeffArray[ii] = HArray[ii]'*CoSParams[ii];
        if ii != length(CoeffArray)
            NominalControlCostArray[ii] = dot(HArray[ii]'*CoSParams[ii],vec(UArrayClosed[ii]))+1/2*vec(UArrayClosed[ii])'*simModel.Cu*vec(UArrayClosed[ii]);
        end
    end
    #if myid() == 5
    #    println(vec(CoSArray[1].p)[1])
    #end
    return CoeffArray,NominalControlCostArray
end
function controlCoeffsExpected(simModel::SimulatePosRangeLocalization2D,  # This is slower than pmap() implementation.
                               s_init::AugState{<:PhysPos,<:VecBelMvNormal},
                               UArray::Vector{<:PosControl},
                               numSamples::Int64,
                               rng::MersenneTwister)
    rngVec = [rng; accumulate(Future.randjump, fill(big(10)^20, nprocs()-1), init=rng)] # Use randjump to get around unresolved issue with parallel RNG.
    cosMat = @distributed (+) for ii = 1:numSamples hcat(getControlCoeffs(simModel,s_init,UArray,rngVec[myid()])...) end
    return cosMat./numSamples
end

function controlCoeffsExpected(simModel::SimulatePosRangeLocalization2D,
                               nominalPolicy::GradientMultiTargetLocalizationPolicy,
                               s_init::AugState{<:PhysPos,<:VecBelMvNormal},
                               u_param_min::Vector{<:Real},
                               u_param_max::Vector{<:Real},
                               UArray::Vector{<:PosControl},
                               numSamples::Int64,
                               rng::MersenneTwister)
    rngVec = [rng; accumulate(Future.randjump, fill(big(10)^20, nprocs()-1), init=rng)] # Use randjump to get around unresolved issue with parallel RNG.
    hcat_coeffs(coeffs) = [hcat(coeffs[1]...),hcat(coeffs[2]...)];
    cosMat = @distributed (+) for ii = 1:numSamples hcat_coeffs(getControlCoeffs(simModel,nominalPolicy,s_init,u_param_min,u_param_max,UArray,rngVec[myid()])) end
    return cosMat[1]./numSamples,[cosMat[2]./numSamples...]
end

function optControlSchedule(simModel::SimulatePosRangeLocalization2D,
                            UArray::Vector{<:PosControl{T}},
                            coeffMat::Matrix{T}) where T <: Real
    UOptArray = similar(UArray);
    Cu_inv = inv(simModel.Cu);
    CostArray = Vector{T}(undef,length(UArray));
    for t = 1:length(UOptArray)
        coeffVec = coeffMat[:,t];
        u = -Cu_inv*coeffVec;
        CostArray[t] = -1/2*coeffVec'*Cu_inv*coeffVec;
        UOptArray[t] = PosControl(UArray[t].t,u);
    end
    return UOptArray,CostArray
end
function optControlSchedule(simModel::SimulatePosRangeLocalization2D,
                            UArray::Vector{<:PosControl{T}},
                            coeffMat::Matrix{T},
                            u_params_min::Vector{T},
                            u_params_max::Vector{T}) where T <: Real
    UOptArray = similar(UArray);
    Cu = simModel.Cu;
    Cu_inv = inv(Cu);
    CostArray = Vector{T}(undef,length(UArray));
    if isdiag(Cu)
        for t = 1:length(UOptArray)
            coeffVec = coeffMat[:,t];
            u_unconstrained = -Cu_inv*coeffVec;
            u_constrained = similar(u_unconstrained);
            for ii = 1:length(u_constrained)
                if u_unconstrained[ii] < u_params_min[ii]
                    u_constrained[ii] = u_params_min[ii];
                elseif u_unconstrained[ii] > u_params_max[ii]
                    u_constrained[ii] = u_params_max[ii];
                else
                    u_constrained[ii] = u_unconstrained[ii];
                end
            end
            UOptArray[t] = PosControl(UArray[t].t,vec(u_constrained));
            CostArray[t] = 1/2*u_constrained'*Cu*u_constrained + dot(u_constrained,coeffVec);
        end
        return UOptArray,CostArray
    else
        warn("Cost coefficient matrix not diagonal. Convex.jl is used for optimizing the control.")
        u = Variable(2);
        constraint = [u <= u_params_max, u >= u_params_min];
        for t = 1:length(UOptArray)
            coeffVec = coeffMat[:,t];
            cost = 1/2*quadform(u,Cu) + dot(u,coeffVec);
            prob = minimize(cost,constraint);
            solve!(prob,ECOSSolver(verbose=false),verbose=false);
            UOptArray[t] = PosControl(UArray[t].t,vec(evaluate(u)));
            CostArray[t] = prob.optval;
        end
        # Convex.clearmemory()
        return UOptArray,CostArray
    end
end

function determineControlTime(tcalc::Float64,
                              simModel::SimulatePosRangeLocalization2D,
                              UOptArray::Vector{<:PosControl},
                              CostArray::Vector{<:Real})
    start = time();
    t_current = UOptArray[1].t;
    t_allowed_min = round(t_current + tcalc*1.05 + simModel.dtexec,digits=5);  # Assuming the total t_calc is 1.05*t_calc so far.
    t_allowed_max = round(t_current + tcalc*1.05 + simModel.dto,digits=5); # This is required so that the control action is fully executed before the next control action takes over.
    index_allowed_min = length(filter(U -> U.t < t_allowed_min, UOptArray)) + 1;
    index_allowed_max = length(filter(U -> U.t <= t_allowed_max, UOptArray));
    index_chosen = findmin(CostArray[index_allowed_min:index_allowed_max])[2];
    tcalc += time() - start;
    return UOptArray[index_chosen+(index_allowed_min-1)],tcalc;
end
function determineControlTime(tcalc::Float64,
                              simModel::SimulatePosRangeLocalization2D,
                              UOptArray::Vector{<:PosControl},
                              CostArray::Vector{<:Real},
                              NominalControlCostArray::Vector{<:Real})
    start = time();
    t_current = UOptArray[1].t;
    t_allowed_min = round(t_current + tcalc*1.05 + simModel.dtexec,digits=5);  # Assuming the total t_calc is 1.05*t_calc so far.
    t_allowed_max = round(t_current + tcalc*1.05 + simModel.dto,digits=5); # This is required so that the control action is fully executed before the next control action takes over.
    index_allowed_min = length(filter(U -> U.t < t_allowed_min, UOptArray)) + 1;
    index_allowed_max = length(filter(U -> U.t <= t_allowed_max, UOptArray));
    index_chosen = findmin(CostArray[index_allowed_min:index_allowed_max]+NominalControlCostArray[index_allowed_min:index_allowed_max])[2];
    tcalc += time() - start;
    return UOptArray[index_chosen+(index_allowed_min-1)],tcalc;
end

function sacControlUpdate(simModel::SimulatePosRangeLocalization2D,
                          s_init::AugState{<:PhysPos,<:VecBelMvNormal},
                          UArray::Vector{<:PosControl},
                          u_params_min::Vector{<:Real},
                          u_params_max::Vector{<:Real},
                          numSamples::Int64, # For forward-backward simulation.
                          rng::MersenneTwister)
    coeffMat = controlCoeffsExpected(simModel,s_init,UArray,numSamples,rng);
    UOptSchedule,CostArray = optControlSchedule(simModel,UArray,coeffMat,u_params_min,u_params_max);
    tcalc = 1/4*simModel.dto; # Assumption for offline simulation.
    UOpt,tcalc = determineControlTime(tcalc,simModel,UOptSchedule,CostArray);
    UArrayNew = copy(UArray);
    ActTimes = Real[];
    for ii = 1:length(UArrayNew)
        if UArrayNew[ii].t >= round(UOpt.t - simModel.dtexec,digits=5) && UArrayNew[ii].t < round(UOpt.t,digits=5)
            push!(ActTimes,UArrayNew[ii].t);
            UArrayNew[ii] = PosControl(UArrayNew[ii].t,vec(UOpt));
        end
    end
    act_time_init,act_time_final = ActTimes[1],ActTimes[end];
    return UArrayNew,act_time_init,act_time_final,tcalc
end
function sacControlUpdate(simModel::SimulatePosRangeLocalization2D,
                          nominalPolicy::GradientMultiTargetLocalizationPolicy,
                          s_init::AugState{<:PhysPos,<:VecBelMvNormal},
                          UArray::Vector{<:PosControl},
                          u_param_min::Vector{<:Real},
                          u_param_max::Vector{<:Real},
                          numSamples::Int64, # For forward-backward simulation.
                          rng::MersenneTwister)
    coeffMat,NominalControlCostArray = controlCoeffsExpected(simModel,nominalPolicy,s_init,u_param_min,u_param_max,UArray,numSamples,rng);
    UOptSchedule,CostArray = optControlSchedule(simModel,UArray,coeffMat,u_param_min,u_param_max);
    tcalc = 1/4*simModel.dto; # Assumption for offline simulation.
    UOpt,tcalc = determineControlTime(tcalc,simModel,UOptSchedule,CostArray,NominalControlCostArray);
    UArrayNew = copy(UArray);
    ActTimes = Real[];
    for ii = 1:length(UArrayNew)
        if UArrayNew[ii].t >= round(UOpt.t - simModel.dtexec,digits=5) && UArrayNew[ii].t < round(UOpt.t,digits=5)
            push!(ActTimes,UArrayNew[ii].t);
            UArrayNew[ii] = PosControl(UArrayNew[ii].t,vec(UOpt));
        end
    end
    act_time_init,act_time_final = ActTimes[1],ActTimes[end];
    return UArrayNew,act_time_init,act_time_final,tcalc
end




function getControlCoeffs(simModel::SimulateManipulate2D,
                          s_init::BelMvNormal,
                          UArray::Vector{<:MControl2D},
                          rng::MersenneTwister)
    YArray,SArray,SArray_before_update = simulateForward(simModel,s_init,UArray,rng);
    CoSArray = simulateBackward(simModel,SArray,SArray_before_update,UArray,YArray);
    CoSParams = map(CoS -> CoS.params, CoSArray);
    HArray = map(s -> trans_u_coeff(simModel.continuousTransModel,s,simModel.Q), SArray);
    CoeffArray = similar(CoSParams);
    for ii = 1:length(CoeffArray)
        CoeffArray[ii] = HArray[ii]'*CoSParams[ii]
    end
    #if myid() == 5
    #    println(CoSArray[1].params[1])
    #end
    return CoeffArray;
end
function getControlCoeffs(simModel::SimulateManipulate2D,
                          nominalPolicy::ManipulatePositionControlPolicy,
                          s_init::BelMvNormal,
                          u_param_min::Vector{<:Real},
                          u_param_max::Vector{<:Real},
                          posGain,
                          rotGain,
                          UArray::Vector{<:MControl2D},
                          rng::MersenneTwister)
    YArray,SArray,SArray_before_update,UArrayClosed = simulateForward(simModel,nominalPolicy,s_init,u_param_min,u_param_max,posGain,rotGain,UArray,rng);
    CoSArray = simulateBackward(simModel,SArray,SArray_before_update,UArrayClosed,YArray);
    CoSParams = map(CoS -> CoS.params, CoSArray);
    HArray = map(s -> trans_u_coeff(simModel.continuousTransModel,s,simModel.Q), SArray);
    CoeffArray = similar(CoSParams);
    NominalControlCostArray = Vector{Float64}(undef,length(CoeffArray)-1);
    for ii = 1:length(CoeffArray)
        CoeffArray[ii] = HArray[ii]'*CoSParams[ii]
        if ii != length(CoeffArray)
            NominalControlCostArray[ii] = dot(HArray[ii]'*CoSParams[ii],vec(UArrayClosed[ii]))+1/2*vec(UArrayClosed[ii])'*simModel.Cu*vec(UArrayClosed[ii]);
        end
    end
    #if myid() == 5
    #    println(CoSArray[1].params[1])
    #end
    return CoeffArray,NominalControlCostArray
end
function controlCoeffsExpected(simModel::SimulateManipulate2D,
                               s_init::BelMvNormal,
                               UArray::Vector{<:MControl2D},
                               numSamples::Int64,
                               rng::MersenneTwister)
    rngVec = [rng; accumulate(Future.randjump, fill(big(10)^20, nprocs()-1), init=rng)] # Use randjump to get around unresolved issue with parallel RNG.
    cosMat = @distributed (+) for ii = 1:numSamples hcat(getControlCoeffs(simModel,s_init,UArray,rngVec[myid()])...) end
    return cosMat./numSamples
end
function controlCoeffsExpected(simModel::SimulateManipulate2D,
                               nominalPolicy::ManipulatePositionControlPolicy,
                               s_init::BelMvNormal,
                               u_param_min::Vector{<:Real},
                               u_param_max::Vector{<:Real},
                               posGain,
                               rotGain,
                               UArray::Vector{<:MControl2D},
                               numSamples::Int64,
                               rng::MersenneTwister)
    rngVec = [rng; accumulate(Future.randjump, fill(big(10)^20, nprocs()-1), init=rng)] # Use randjump to get around unresolved issue with parallel RNG.
    hcat_coeffs(coeffs) = [hcat(coeffs[1]...),hcat(coeffs[2]...)];
    cosMat = @distributed (+) for ii = 1:numSamples hcat_coeffs(getControlCoeffs(simModel,nominalPolicy,s_init,u_param_min,u_param_max,posGain,rotGain,UArray,rngVec[myid()])) end
    return cosMat[1]./numSamples,[cosMat[2]./numSamples...]
end
function optControlSchedule(simModel::SimulateManipulate2D,
                            UArray::Vector{MControl2D{T}},
                            coeffMat::Matrix{T}) where T <: Real
    UOptArray = similar(UArray);
    Cu_inv = inv(simModel.Cu);
    CostArray = Vector{T}(undef,length(UArray));
    for t = 1:length(UOptArray)
        coeffVec = coeffMat[:,t];
        u = -Cu_inv*coeffVec;
        CostArray[t] = -1/2*coeffVec'*Cu_inv*coeffVec;
        UOptArray[t] = MControl2D(UArray[t].t,u)
    end
    return UOptArray,CostArray
end
function optControlSchedule(simModel::SimulateManipulate2D,
                            UArray::Vector{MControl2D{T}},
                            coeffMat::Matrix{T},
                            u_params_min::Vector{T},
                            u_params_max::Vector{T}) where T <: Real
    UOptArray = similar(UArray);
    Cu = simModel.Cu;
    Cu_inv = inv(Cu);
    CostArray = Vector{T}(undef, length(UArray));
    if isdiag(Cu)
        for t = 1:length(UOptArray)
            coeffVec = coeffMat[:,t];
            u_unconstrained = -Cu_inv*coeffVec;
            u_constrained = similar(u_unconstrained);
            for ii = 1:length(u_constrained)
                if u_unconstrained[ii] < u_params_min[ii]
                    u_constrained[ii] = u_params_min[ii];
                elseif u_unconstrained[ii] > u_params_max[ii]
                    u_constrained[ii] = u_params_max[ii];
                else
                    u_constrained[ii] = u_unconstrained[ii];
                end
            end
            UOptArray[t] = MControl2D(UArray[t].t,vec(u_constrained));
            CostArray[t] = 1/2*u_constrained'*Cu*u_constrained + dot(u_constrained,coeffVec);
        end
        return UOptArray,CostArray
    else
        warn("Cost coefficient matrix not diagonal. Convex.jl is used for optimizing the control.")
        u = Variable(3);
        constraint = [u <= u_params_max, u >= u_params_min];
        for t = 1:length(UOptArray)
            coeffVec = coeffMat[:,t];
            cost = 1/2*quadform(u,Cu) + dot(u,coeffVec);
            prob = minimize(cost,constraint);
            solve!(prob,ECOSSolver(verbose=false),verbose=false);
            UOptArray[t] = MControl2D(UArray[t].t,vec(evaluate(u)));
            CostArray[t] = prob.optval;
        end
        # Convex.clearmemory()
        return UOptArray,CostArray
    end
end

function determineControlTime(tcalc::Float64,
                              simModel::SimulateManipulate2D,
                              UOptArray::Vector{<:MControl2D},
                              CostArray::Vector{<:Real})
    start = time();
    t_current = UOptArray[1].t;
    t_allowed_min = round(t_current + tcalc*1.05 + simModel.dtexec,digits=5);  # Assuming the total t_calc is 1.05*t_calc so far.
    t_allowed_max = round(t_current + tcalc*1.05 + simModel.dto,digits=5); # This is required so that the control action is fully executed before the next control action takes over.
    index_allowed_min = length(filter(U -> U.t < t_allowed_min, UOptArray)) + 1;
    index_allowed_max = length(filter(U -> U.t <= t_allowed_max, UOptArray));
    index_chosen = findmin(CostArray[index_allowed_min:index_allowed_max])[2];
    tcalc += time() - start;
    return UOptArray[index_chosen+(index_allowed_min-1)],tcalc;
end
function determineControlTime(tcalc::Float64,
                              simModel::SimulateManipulate2D,
                              UOptArray::Vector{<:MControl2D},
                              CostArray::Vector{<:Real},
                              NominalControlCostArray::Vector{<:Real})
    start = time();
    t_current = UOptArray[1].t;
    t_allowed_min = round(t_current + tcalc*1.05 + simModel.dtexec,digits=5);  # Assuming the total t_calc is 1.05*t_calc so far.
    t_allowed_max = round(t_current + tcalc*1.05 + simModel.dto,digits=5); # This is required so that the control action is fully executed before the next control action takes over.
    index_allowed_min = length(filter(U -> U.t < t_allowed_min, UOptArray)) + 1;
    index_allowed_max = length(filter(U -> U.t <= t_allowed_max, UOptArray));
    index_chosen = findmin(CostArray[index_allowed_min:index_allowed_max]+NominalControlCostArray[index_allowed_min:index_allowed_max])[2];
    tcalc += time() - start;
    return UOptArray[index_chosen+(index_allowed_min-1)],tcalc;
end

function sacControlUpdate(simModel::SimulateManipulate2D,
                          s_init::BelMvNormal,
                          UArray::Vector{<:MControl2D},
                          u_params_min::Vector{<:Real},
                          u_params_max::Vector{<:Real},
                          numSamples::Int64, # For forward-backward simulation.
                          rng::MersenneTwister)
    coeffMat = controlCoeffsExpected(simModel,s_init,UArray,numSamples,rng);
    UOptSchedule,CostArray = optControlSchedule(simModel,UArray,coeffMat,u_params_min,u_params_max);
    tcalc = 1/4*simModel.dto; # For offline simulation.
    UOpt,tcalc = determineControlTime(tcalc,simModel,UOptSchedule,CostArray);
    UArrayNew = copy(UArray);
    ActTimes = Real[];
    for ii = 1:length(UArrayNew)
        if UArrayNew[ii].t >= round(UOpt.t - simModel.dtexec,digits=5) && UArrayNew[ii].t < round(UOpt.t,digits=5)
            push!(ActTimes,UArrayNew[ii].t);
            UArrayNew[ii] = MControl2D(UArrayNew[ii].t,vec(UOpt));
        end
    end
    act_time_init,act_time_final = ActTimes[1],ActTimes[end];
    return UArrayNew,act_time_init,act_time_final,tcalc
end
function sacControlUpdate(simModel::SimulateManipulate2D,
                          nominalPolicy::ManipulatePositionControlPolicy,
                          s_init::BelMvNormal,
                          UArray::Vector{<:MControl2D},
                          u_param_min::Vector{<:Real},
                          u_param_max::Vector{<:Real},
                          posGain,
                          rotGain,
                          numSamples::Int64, # For forward-backward simulation.
                          rng::MersenneTwister)
    coeffMat,NominalControlCostArray = controlCoeffsExpected(simModel,nominalPolicy,s_init,u_param_min,u_param_max,posGain,rotGain,UArray,numSamples,rng);
    UOptSchedule,CostArray = optControlSchedule(simModel,UArray,coeffMat,u_param_min,u_param_max);
    tcalc = 1/4*simModel.dto; # For offline simulation.
    UOpt,tcalc = determineControlTime(tcalc,simModel,UOptSchedule,CostArray,NominalControlCostArray);
    UArrayNew = copy(UArray);
    ActTimes = Real[];
    for ii = 1:length(UArrayNew)
        if UArrayNew[ii].t >= round(UOpt.t - simModel.dtexec,digits=5) && UArrayNew[ii].t < round(UOpt.t,digits=5)
            push!(ActTimes,UArrayNew[ii].t);
            UArrayNew[ii] = MControl2D(UArrayNew[ii].t,vec(UOpt));
        end
    end
    act_time_init,act_time_final = ActTimes[1],ActTimes[end];
    return UArrayNew,act_time_init,act_time_final,tcalc
end
