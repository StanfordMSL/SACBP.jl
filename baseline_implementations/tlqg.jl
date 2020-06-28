try
    pyimport("sys")
catch e
    using PyCall
    @info "Python executable used by PyCall: $(pyimport("sys").executable)";
    # Append the code directory to python path
    # (see https://github.com/JuliaPy/PyCall.jl#troubleshooting)
    file_dir = @__DIR__;
    pushfirst!(PyVector(pyimport("sys")."path"),
               file_dir);
end

py"""
from tlqg import TLQG_PosRangeLocalization2D, TLQG_Manipulate2D
"""
import SACBP: BelMvNormal, MControl2D
using Distributions
using Printf


mutable struct TLQG_Planner_PosRangeLocalization2D
    planner::PyObject
    μ_array::Union{Nothing, Vector{Vector{Float64}}}
    Σ_array::Union{Nothing, Vector{Matrix{Float64}}}
    l_array::Union{Nothing, Vector{Vector{Float64}}}
    L_array::Union{Nothing, Vector{Matrix{Float64}}}
    kl_threshold::Float64 # kl threshold to trigger replanning
    t_last_plan::Union{Nothing, Float64} # last time when replanning occured
end

function TLQG_Planner_PosRangeLocalization2D(Th::Float64, # planning horizon [s]
                                             dto::Float64,
                                             Nt::Int64, # number of targets
                                             Q::Matrix{Float64},
                                             Cu::Matrix{Float64},
                                             u_param_min::Vector{Float64},
                                             u_param_max::Vector{Float64},
                                             kl_threshold::Float64)
    N = Int64(round(Th/dto, digits=5));
    Cs = diagm(vcat([10.0, 10.0], zeros(2*Nt)))
    Q_tlqg = zeros(2*Nt, 2*Nt)
    for ii = 1:Nt
        Q_tlqg[2*(ii-1)+1:2*ii, 2*(ii-1)+1:2*ii] = Q
    end
    planner = py"TLQG_PosRangeLocalization2D"(N, dto, Nt, Q_tlqg, Cs, Cu,
                                              u_param_min, u_param_max)
    return TLQG_Planner_PosRangeLocalization2D(planner, nothing, nothing, nothing, nothing,
                                               kl_threshold, nothing)
end

function get_kl_divergence(s1::AugState{<:PhysPos,<:VecBelMvNormal},
                           s2::AugState{<:PhysPos,<:VecBelMvNormal})
    Nt = length(s1.b)
    @assert length(s2.b) == Nt
    kl = 0.0;
    for ii = 1:Nt
        b1 = s1.b[ii]
        μ1, Σ1 = b1.params[1:b1.dim], reshape(b1.params[b1.dim+1:end],b1.dim,b1.dim);
        if !isposdef(Σ1)
            λ, V = eigen(Σ1)
            λ .= max.(1e-6, λ)
            Σ1 = Symmetric(V*diagm(λ)*V');
        end
        b2 = s2.b[ii]
        μ2, Σ2 = b2.params[1:b2.dim], reshape(b2.params[b2.dim+1:end],b2.dim,b2.dim);
        if !isposdef(Σ2)
            λ, V = eigen(Σ2)
            λ .= max.(1e-6, λ)
            Σ2 = Symmetric(V*diagm(λ)*V');
        end
        kl += 0.5*(tr(Σ2\Σ1) + (μ2 - μ1)'*(Σ2\(μ2 - μ1))
            - size(Σ2, 1) + logdet(Σ2) - logdet(Σ1))
    end
    return kl
end

function get_symmetric_kl(s1::AugState{<:PhysPos,<:VecBelMvNormal},
                          s2::AugState{<:PhysPos,<:VecBelMvNormal})
    symmetric_kl = 0.5*(get_kl_divergence(s1, s2) + get_kl_divergence(s2, s1))
    return symmetric_kl
end

function tlqgControlUpdate(tlqg::TLQG_Planner_PosRangeLocalization2D,
                           s::AugState{<:PhysPos,<:VecBelMvNormal},
                           UArray::Vector{<:PosControl};
                           verbose=true, online=true)
    dto = tlqg.planner.dto
    dim = s.b[1].dim;
    μ_current = vcat(s.p.pos, [b.params[1:dim] for b in s.b]...);
    Σ_current = zeros(length(μ_current)-s.p.dim, length(μ_current)-s.p.dim)
    for ii = 1:tlqg.planner.Nt
        Σ_current[dim*(ii-1)+1:dim*ii, dim*(ii-1)+1:dim*ii] = reshape(s.b[ii].params[dim+1:end], dim, dim)
    end
    replan_flag = false;
    if isnothing(tlqg.t_last_plan)
        # No planning done yet
        replan_flag = true
    else
        time_idx = Int64(round((s.t - tlqg.t_last_plan)/dto, digits=5)) + 1
        if time_idx > tlqg.planner.N
            # Reached the end of planning horizon
            if verbose
                println("Replanning since the end of planning horizon reached.")
            end
            replan_flag = true
        else
            # Symmetric KL larger than the replanning threshold
            ego = PhysPos(s.t, tlqg.μ_array[time_idx][1:dim]);
            bVec = [BelMvNormal(s.t, tlqg.μ_array[time_idx][dim+dim*(ii-1)+1:dim+dim*ii],
                                tlqg.Σ_array[time_idx][dim*(ii-1)+1:dim*ii, dim*(ii-1)+1:dim*ii])
                    for ii = 1:tlqg.planner.Nt];
            s_model = AugState(ego, bVec);
            kl_val = get_symmetric_kl(s_model, s)
            kl_str = @sprintf "%2.2f" kl_val
            if verbose
                println("KL: $(kl_str)")
            end
            kl_thresh_str = @sprintf "%2.2f" tlqg.kl_threshold
            if kl_val > tlqg.kl_threshold
                if verbose
                    println("Replanning since KL: $(kl_str) > $(kl_thresh_str).")
                end
                replan_flag = true
            end
        end
    end
    if replan_flag
        tlqg.μ_array, tlqg.Σ_array, tlqg.l_array, tlqg.L_array, elapsed =
            tlqg.planner.solve_tlqg(μ_current, Σ_current, verbose=false, online=online)
        for ii = 1:length(tlqg.Σ_array)
            tlqg.Σ_array[ii] = Symmetric(tlqg.Σ_array[ii])
            if !isposdef(tlqg.Σ_array[ii])
                @warn "At T = $(b.t)[s], tlqg.Σ_array[$(ii)] is not posdef. Fixing..."
                λ, V = eigen(tlqg.Σ_array[ii])
                λ .= max.(1e-6, λ)
                tlqg.Σ_array[ii] = Symmetric(V*diagm(λ)*V');
            end
        end
        tlqg.t_last_plan = s.t;
    else
        elapsed = nothing
    end
    time_idx = Int64(round((s.t - tlqg.t_last_plan)/dto, digits=5)) + 1
    if verbose
        println("Executing policy at time step $(time_idx) out of $(tlqg.planner.N)")
    end
    u_val = -tlqg.L_array[time_idx]*(μ_current - tlqg.μ_array[time_idx]) + tlqg.l_array[time_idx];
    # saturate control input
    for ii = 1:length(u_val)
        if u_val[ii] < tlqg.planner.u_param_min[ii]
            u_val[ii] = tlqg.planner.u_param_min[ii];
        elseif u_val[ii] > tlqg.planner.u_param_max[ii]
            u_val[ii] = tlqg.planner.u_param_max[ii];
        end
    end
    if verbose
        println(u_val)
    end
    UArray_new = copy(UArray)
    for ii = 1:length(UArray_new)
        if UArray_new[ii].t < round(s.t + dto,digits=5)
            UArray_new[ii] = PosControl(UArray_new[ii].t,u_val);
        end
    end
    return elapsed, UArray_new
end


mutable struct TLQG_Planner_Manipulate2D
    planner::PyObject
    μ_array::Union{Nothing, Vector{Vector{Float64}}}
    Σ_array::Union{Nothing, Vector{Matrix{Float64}}}
    l_array::Union{Nothing, Vector{Vector{Float64}}}
    L_array::Union{Nothing, Vector{Matrix{Float64}}}
    kl_threshold::Float64 # kl threshold to trigger replanning
    t_last_plan::Union{Nothing, Float64} # last time when replanning occured
end

function TLQG_Planner_Manipulate2D(Th::Float64, # planning horizon [s]
                                   dto::Float64,
                                   Q::Matrix{Float64}, R::Matrix{Float64},
                                   Cs::Matrix{Float64}, Cu::Matrix{Float64},
                                   u_param_min::Vector{Float64},
                                   u_param_max::Vector{Float64},
                                   posGain::Float64, rotGain::Float64,
                                   kl_threshold::Float64)
    x_target = [0.,0.,pi,0.,0.,0.];
    N = Int64(round(Th/dto, digits=5));
    planner = py"TLQG_Manipulate2D"(N, dto, Q, R, Cs, Cu, x_target,
                                    u_param_min, u_param_max, posGain, rotGain)
    return TLQG_Planner_Manipulate2D(planner, nothing, nothing, nothing, nothing,
                                     kl_threshold, nothing)
end

function get_kl_divergence(b1::BelMvNormal{Float64}, b2::BelMvNormal{Float64})
    μ1, Σ1 = b1.params[1:b1.dim], reshape(b1.params[b1.dim+1:end],b1.dim,b1.dim);
    if !isposdef(Σ1)
        λ, V = eigen(Σ1)
        λ .= max.(1e-6, λ)
        Σ1 = Symmetric(V*diagm(λ)*V');
    end
    μ2, Σ2 = b2.params[1:b2.dim], reshape(b2.params[b2.dim+1:end],b2.dim,b2.dim);
    if !isposdef(Σ2)
        λ, V = eigen(Σ2)
        λ .= max.(1e-6, λ)
        Σ2 = Symmetric(V*diagm(λ)*V');
    end
    kl = 0.5*(tr(Σ2\Σ1) + (μ2 - μ1)'*(Σ2\(μ2 - μ1))
       - size(Σ2, 1) + logdet(Σ2) - logdet(Σ1))
    return kl
end

function get_symmetric_kl(b1::BelMvNormal{Float64}, b2::BelMvNormal{Float64})
    symmetric_kl = 0.5*(get_kl_divergence(b1, b2) + get_kl_divergence(b2, b1))
    return symmetric_kl
end

function tlqgControlUpdate(tlqg::TLQG_Planner_Manipulate2D,
                           b::BelMvNormal{Float64},
                           UArray::Vector{<:MControl2D};
                           verbose=true, online=true)
    dto = tlqg.planner.dto
    μ_current, Σ_current = b.params[1:b.dim], reshape(b.params[b.dim+1:end],b.dim,b.dim);
    replan_flag = false;
    if isnothing(tlqg.t_last_plan)
        # No planning done yet
        replan_flag = true
    else
        time_idx = Int64(round((b.t - tlqg.t_last_plan)/dto, digits=5)) + 1
        if time_idx > tlqg.planner.N
            # Reached the end of planning horizon
            if verbose
                println("Replanning since the end of planning horizon reached.")
            end
            replan_flag = true
        else
            # Symmetric KL larger than the replanning threshold
            b_model = BelMvNormal(b.t, tlqg.μ_array[time_idx], tlqg.Σ_array[time_idx])
            kl_val = get_symmetric_kl(b_model, b)
            kl_str = @sprintf "%2.2f" kl_val
            if verbose
                println("KL: $(kl_str)")
            end
            kl_thresh_str = @sprintf "%2.2f" tlqg.kl_threshold
            if kl_val > tlqg.kl_threshold
                if verbose
                    println("Replanning since KL: $(kl_str) > $(kl_thresh_str).")
                end
                replan_flag = true
            end
        end
    end
    if replan_flag
        tlqg.μ_array, tlqg.Σ_array, tlqg.l_array, tlqg.L_array, elapsed =
            tlqg.planner.solve_tlqg(μ_current, Σ_current, verbose=false, online=online)
        for ii = 1:length(tlqg.Σ_array)
            tlqg.Σ_array[ii] = Symmetric(tlqg.Σ_array[ii])
            if !isposdef(tlqg.Σ_array[ii])
                @warn "At T = $(b.t)[s], tlqg.Σ_array[$(ii)] is not posdef. Fixing..."
                λ, V = eigen(tlqg.Σ_array[ii])
                λ .= max.(1e-6, λ)
                tlqg.Σ_array[ii] = Symmetric(V*diagm(λ)*V');
            end
        end
        tlqg.t_last_plan = b.t;
    else
        elapsed = nothing
    end
    time_idx = Int64(round((b.t - tlqg.t_last_plan)/dto, digits=5)) + 1
    if verbose
        println("Executing policy at time step $(time_idx) out of $(tlqg.planner.N)")
    end
    u_val = -tlqg.L_array[time_idx]*(μ_current - tlqg.μ_array[time_idx]) + tlqg.l_array[time_idx];
    # saturate control input
    for ii = 1:length(u_val)
        if u_val[ii] < tlqg.planner.u_param_min[ii]
            u_val[ii] = tlqg.planner.u_param_min[ii];
        elseif u_val[ii] > tlqg.planner.u_param_max[ii]
            u_val[ii] = tlqg.planner.u_param_max[ii];
        end
    end
    UArray_new = copy(UArray)
    for ii = 1:length(UArray_new)
        if UArray_new[ii].t < round(b.t + dto,digits=5)
            UArray_new[ii] = MControl2D(UArray_new[ii].t,u_val);
        end
    end
    return elapsed, UArray_new
end
