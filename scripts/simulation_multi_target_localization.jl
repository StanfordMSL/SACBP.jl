ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python")
using Distributed
# using 4 cores
if 4 - Distributed.nprocs() > 0
    Distributed.addprocs(4 - Distributed.nprocs());
end
@everywhere using SACBP
using SACBP
using LinearAlgebra
using Plots
pyplot()
using ProgressMeter
using Random
using JLD2
using FileIO

include(joinpath(@__DIR__, "../baseline_implementations/gradient_greedy.jl"))
include(joinpath(@__DIR__, "../baseline_implementations/mcts.jl"))
include(joinpath(@__DIR__, "../baseline_implementations/tlqg.jl"))

function get_initial_augmented_state(;Nt=20) # Number of Targets
    p_init = PhysPos(0.,[12.0,12.0]);
    b_init = [BelMvNormal(0.,zeros(2),300*Matrix(1.0I, 2, 2)) for ii = 1:Nt];
    s_init = AugState(p_init,b_init);
    return s_init
end

function get_target_motion_data(;Nt=20,    # Number of Targets,
                                 dtc=0.01, # Euler Approximation Interval [s]
                                 Q=0.1*Matrix(1.0I, 2, 2), # Process Noise for Each Target
                                 tf=200.0) # Simulation Horizon [s])
    try
        data_path = normpath(joinpath(@__DIR__, "..", "data",
                                      "multi_target_localization",
                                      "target_motion_data_localization.jld2"))
        data = FileIO.load(data_path)
        println("Loaded target motion data from $(data_path)")
        q_history = data["q_history"]
        @assert length(q_history) == Int64(round(tf/dtc,digits=5)) + 1
        @assert length(q_history[1]) == Nt
        @assert data["Q"] == Q "Q in loaded data does not match desired Q"
        return q_history
    catch e
        println("Data file does not exist. Generating taget motion data")
        rng = MersenneTwister(1234)  # RNG for generating Target Motion.

        # Target Motion Generation.
        q_init_g1 = [PhysPos(0.,rand(rng, 2).*10) for ii = 1:Int64(Nt/2)];
        q_init_g2 = [PhysPos(0.,rand(rng, 2).*10 + [20.,20.]) for ii = 1:Int64(Nt/2)];  # Initial target positions.
        q_init = vcat([q_init_g1,q_init_g2]...);
        q_history = [q_init];
        for t = 0.:dtc:tf-dtc
            push!(q_history,[trans(TransModel_Pos(),q,PosControl(t,zeros(2)),dtc,Q,rng) for q in q_history[end]])
        end
        data_path = normpath(joinpath(@__DIR__, "..", "data",
                                      "multi_target_localization",
                                      "target_motion_data_localization.jld2"))
        FileIO.save(data_path, "q_history", q_history, "Q", Q)
        println("Saved target motion data to $(data_path)")
        return q_history
    end
end

function visualizeSimEnv(s::AugState{<:PhysPos,<:VecBelMvNormal{T} where T<:Real},
                         q::Vector{<:PhysPos};plot_belief=true)
    plt = plot();
    plt = scatter!([qi.pos[1] for qi in q],[qi.pos[2] for qi in q],
                   aspect_ratio=1.0,
                   xlabel="x [m]", ylabel="y [m]", label="Targets",
                   color=:cyan,marker=(:star5,8.,1.0,stroke(0.5,:darkblue)),
                   size=(500,500))
    if plot_belief
        for bi in s.b
            plot_e_ellipse!(bi,0.99,plt);
        end
    end
    plt = scatter!((s.p.pos[1],s.p.pos[2]),
                  label="Robot", markershape = :circle, markersize = 7.0,
                  color=:red)
    xlims!((-10.,50.));
    ylims!((-10.,50.));
    return plt
end;

function visualize_results(s_history,q_history)
    plt = plot();
    Nt = length(q_history[1])

    qix_traj = [[q_history[t][ii].pos[1] for t = 1:length(q_history)] for ii = 1:Nt];
    qiy_traj = [[q_history[t][ii].pos[2] for t = 1:length(q_history)] for ii = 1:Nt];
    for ii = 1:Nt
        plt = plot!(qix_traj[ii],qiy_traj[ii],color=:paleturquoise,label="",
                    aspect_ratio=1.0);
        plot_e_ellipse!(s_history[end].b[ii],0.99,plt)
    end

    p_trajectory = hcat(map(s -> s.p.pos,s_history)...);
    plot!(p_trajectory[1,:],p_trajectory[2,:],color=:crimson,
          label="",size=(500,500))
    xlims!((-10.,50.));
    ylims!((-10.,50.));
    return plt,p_trajectory
end

function simulate(s_init::AugState{<:PhysPos,<:VecBelMvNormal},
                  q_history::Vector,
                  controlMode::String, rng_num::Int64,
                  simPosUKF::SimulatePosRangeLocalization2D;
                  numSamples=10, # for sacbp only
                  u_param_min::Vector{Float64}=[-2.0, -2.0],
                  u_param_max::Vector{Float64}=[2.0, 2.0],
                  Th::Float64=2.0, # planning horizon [s]
                  tf::Float64=200.0, # simulation horizon [s]
                  mcts_param::Union{Nothing, MCTSParams}=nothing,
                  tlqg::Union{Nothing, TLQG_Planner_PosRangeLocalization2D}=nothing,
                  animate=true, plotfig=true, verbose=true)
    rng = MersenneTwister(rng_num)
    @assert controlMode in ["greedy", "sacbp", "mcts", "tlqg", "tlqg_offline"];
    if animate
        # Animation Object.
        anim = Animation();
    end
    # Only for MCTS policies.
    if controlMode == "mcts"
        @assert !isnothing(mcts_param) "MCTS Parameters not given"
        amdp = AugMDP(simPosUKF, u_param_min, u_param_max, mcts_param.numActions)
        mctsSolverPlain = MCTS.DPWSolver(depth=Int64(Th/simPosUKF.dto),
                                         n_iterations=mcts_param.numSamples,
                                         exploration_constant=mcts_param.expConst,
                                         k_action=mcts_param.kAct,
                                         alpha_action=mcts_param.αAct,
                                         k_state=mcts_param.kState,
                                         alpha_state=mcts_param.αState,
                                         rng=rng);
        mctsPolicyPlain = solve(mctsSolverPlain,amdp);
    end

    ukfModel = simPosUKF.discreteTransModel;
    transPos = simPosUKF.continuousTransModel;
    dtc, dto, Q = simPosUKF.dtc, simPosUKF.dto, simPosUKF.Q;
    Nt = length(q_history[1])

    # Controller Compilation.
    U_default = [PosControl(round(dtc*t,digits=5),zeros(2)) for t = 0:Int64(Th/dtc)-1];
    if controlMode == "greedy"     # Greedy Heuristic.
        @time U_default = gradientControlUpdate(ukfModel,s_init,U_default,u_param_min,u_param_max,Q,dto)
    elseif controlMode == "sacbp" # SACBP.
        @time U_default = sacControlUpdate(simPosUKF,s_init,U_default,u_param_min,u_param_max,numSamples,rng,offline=true)[1]
    elseif controlMode == "mcts" # MCTS.
        @time U_default = mctsControlUpdate(mctsPolicyPlain,s_init,U_default,dto)
    elseif controlMode == "tlqg"
        @assert !isnothing(tlqg) "T-LQG Planner object not given"
        @time ~,U_default = tlqgControlUpdate(tlqg,s_init,U_default,verbose=verbose,online=true)
    elseif controlMode == "tlqg_offline"
        # offline == planner can take much longer time to compute NLP iterations (until convergence).
        # Only for collecting computation time statistics
        @assert !isnothing(tlqg) "T-LQG Planner object not given"
        @time ~,U_default = tlqgControlUpdate(tlqg,s_init,U_default,verbose=verbose,online=false)
    end
    println("Controller Compiled. Starting Simulation...")

    # Simulation.
    y_history = Vector{Float64}[];
    tcalc_true_history = Float64[];
    s_history = [s_init];
    u_history = PosControl{Float64}[];
    U_pool = copy(U_default);

    observeModel = ObserveModel_Range();
    numSteps = Int64(tf/dtc);
    nextSampleTime = dto;
    @showprogress 1 for t = 1:numSteps
        s = s_history[end];
        if round(t*dtc,digits=5) > nextSampleTime
        # Belief & Control Update.
            # Observation.
            q = q_history[t];
            yVec = Vector{Float64}(undef, Nt);
            for ii = 1:Nt
                yVec[ii] = observe(observeModel,s.p,q[ii].pos,covObservePos(s.p.pos,q[ii].pos),rng);
            end
            push!(y_history,yVec);
            # Belief Update.
            b_new = trans(ukfModel,map(bi -> BelMvNormal(s.p.t,bi.params),s.b),s.p,dto,yVec,Q);
            s_new = AugState(s.p,b_new);
            if verbose
                println((s_new.t,"Belief Updated."));
            end
            if animate
                plt = visualizeSimEnv(s_new,q);
                title!("Time: $(s.t) [s]")
                Plots.frame(anim);
            end
            # Control Update.
            if controlMode == "greedy";     # Greedy Heuristic.
                tcalc_true = @elapsed U_pool = gradientControlUpdate(ukfModel,s_new,U_pool,u_param_min,u_param_max,Q,dto)
                push!(tcalc_true_history,tcalc_true);
                if verbose
                    println((s.t,"Control Updated."));
                end
            elseif controlMode == "sacbp"; # SACBP.
                tcalc_true = @elapsed U_pool,act_time_init,act_time_final,tcalc_simulated = sacControlUpdate(simPosUKF,s_new,U_pool,u_param_min,u_param_max,numSamples,rng,offline=false);
                push!(tcalc_true_history,tcalc_true);
                if verbose
                    println((s.t,"Control Updated.","time_init=$(act_time_init)","time_final=$(act_time_final)"));
                end
            elseif controlMode == "mcts"; # MCTS.
                tcalc_true = @elapsed U_pool = mctsControlUpdate(mctsPolicyPlain,s_new,U_pool,dto);
                push!(tcalc_true_history,tcalc_true);
                if verbose
                    println((s.t,"Control Updated."));
                end
            elseif controlMode == "tlqg" # T-LQG
                tcalc_true, U_pool = tlqgControlUpdate(tlqg,s_new,U_pool,verbose=verbose,online=true)
                if !isnothing(tcalc_true)
                    push!(tcalc_true_history,tcalc_true);
                end
                if verbose
                    println((s.t,"Control Updated."));
                end
            elseif controlMode == "tlqg_offline" # T-LQG, offline
                tcalc_true, U_pool = tlqgControlUpdate(tlqg,s_new,U_pool,verbose=verbose,online=false)
                if !isnothing(tcalc_true)
                    push!(tcalc_true_history,tcalc_true);
                end
                if verbose
                    println((s.t,"Control Updated."));
                end
            end
            # Control Execution.
            u = popfirst!(U_pool);
            push!(u_history,u);
            p_new = trans(transPos,s.p,u,dtc);
            s_new = AugState(p_new,map(bi -> BelMvNormal(p_new.t,bi.params),b_new));
            push!(s_history,s_new);
            #println(s_new.p)

            # Append nominal control.
            u_nominal = PosControl(round(s.t+Th,digits=5),zeros(2));
            push!(U_pool,u_nominal);
            nextSampleTime = round(nextSampleTime+dto,digits=5);
        else
            #q = q_history[t];
            #if t%5 == 0
            #    plt = visualizeSimEnv(s,q);
            #    title!("Time: $(s.t) [s]")
            #    frame(anim);
            #end
            # No Control Update.
            # Control Execution.
            u = popfirst!(U_pool);
            push!(u_history,u);
            p_new = trans(transPos,s.p,u,dtc);
            s_new = AugState(p_new,map(bi -> BelMvNormal(p_new.t,bi.params),s.b));
            push!(s_history,s_new);
            #println(s_new.p);

            # Append nominal control.
            u_nominal = PosControl(round(s.t+Th,digits=5),zeros(2));
            push!(U_pool,u_nominal);
        end
    end
    s = s_history[end];
    # Observation.
    q = q_history[numSteps+1];
    yVec = Vector{Float64}(undef, Nt);
    for ii = 1:Nt
        yVec[ii] = observe(observeModel,s.p,q[ii].pos,covObservePos(s.p.pos,q[ii].pos),rng);
    end
    push!(y_history,yVec);
    # Belief Update.
    b_new = trans(ukfModel,map(bi -> BelMvNormal(s.p.t,bi.params),s.b),s.p,dto,yVec,Q);
    s_new = AugState(s.p,b_new);
    if verbose
        println((s_new.t,"Belief Updated."));
    end
    if animate
        plt = visualizeSimEnv(s_new,q);
        title!("Time: $(s.t) [s]")
        Plots.frame(anim);
    end
    push!(s_history,s_new);
    log_dir = normpath(joinpath(@__DIR__, "../data",
                                 "multi_target_localization"));
    if plotfig
        fig_path = joinpath(log_dir, "tracking_traj_$(controlMode)_tf_$(tf)_rng_$(rng_num).png")
        plt = visualize_results(s_history, q_history);
        if controlMode == "greedy"
            title = "Greedy"
        elseif controlMode == "sacbp"
            title = "SACBP (Ours)"
        elseif controlMode == "mcts"
            title = "MCTS-DPW"
        elseif controlMode == "tlqg"
            title = "T-LQG"
        elseif controlMode == "tlqg_offline"
            title = "T-LQG (Offline)"
        end
        plot!(title=title)
        savefig(fig_path)
    end
    if animate
        gif(anim,joinpath(log_dir, "$(controlMode)_tf_$(tf)_rng_$(rng_num).gif"),fps=50)
    end
    return y_history,tcalc_true_history,s_history,u_history,U_pool
end

function simulate_main(controlMode::String; rng_num::Int64,
                       animate=true, plotfig=true, verbose=true)
    # SAC Controller Parameters.
    dtc = 0.01;         # Euler Approximation Interval.
    dto = 0.2;          # Observation & Control Update Interval.
    Th = 2.0;           # Planning Horizon.
    dtexec = 0.8*dto;   # Control Execution Duration.
    Q = 0.1*Matrix(1.0I, 2, 2);     # Process Noise for Each Target.
    Cu = 0.1*Matrix(1.0I, 2, 2);    # Control Cost Coefficient.
    u_param_min = [-2.0,-2.0];  # Lower Limit for PosControl
    u_param_max = [2.0,2.0];    # Upper Limit for PosControl
    transPos = TransModel_Pos();
    ukfModel = UKFPosRange();
    simPosUKF = SimulatePosRangeLocalization2D(transPos,ukfModel,dtc,dto,dtexec,Q,Cu);
    numSamples = 10;     # Number of Monte Carlo Samples.

    # MCTS Parameters
    numActions = 5;
    # numSamples_mcts = 15;
    numSamples_mcts = 25;
    expConst = 10.0;
    kAct = 10.0;
    αAct = 0.3;
    kState = 3.0;
    αState = 0.1;
    mcts_param = MCTSParams(numActions, numSamples_mcts, expConst, kAct, αAct, kState, αState);

    # Simulation Parameters.
    Nt = 20;     # Number of Targets.
    tf = 200.;    # Simulation Time.

    # T-LQG Planner
    Th_tlqg = Th;
    kl_threshold = 1e6;
    tlqg = TLQG_Planner_PosRangeLocalization2D(Th_tlqg,dto,Nt,Q,Cu,u_param_min,u_param_max,kl_threshold)

    # Initial Augmented State
    s_init = get_initial_augmented_state(Nt=Nt);

    # Target Motion Generation
    q_history = get_target_motion_data(Nt=Nt, dtc=dtc, Q=Q,tf=tf);
    if plotfig
        frame = visualizeSimEnv(s_init, q_history[1], plot_belief=false)
        init_fig_path = normpath(joinpath(@__DIR__, "../data",
                                          "multi_target_localization",
                                          "tracking_initial.pdf"))
        savefig(init_fig_path);
    end

    y_history,tcalc_true_history,s_history,u_history,U_pool =
        simulate(s_init, q_history, controlMode, rng_num,
                 simPosUKF, numSamples=numSamples,
                 u_param_min=u_param_min, u_param_max=u_param_max,
                 Th=Th, tf=tf, mcts_param=mcts_param, tlqg=tlqg,
                 animate=animate, plotfig=plotfig, verbose=verbose);
    println("Average control computation time: $(round(mean(tcalc_true_history), digits=3)) [s]")
    return y_history,tcalc_true_history,s_history,u_history,U_pool
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("1. Simulation with Greedy Controller")
    simulate_main("greedy", rng_num=169, animate=true, plotfig=true, verbose=false);
    println("2. Simulation with SACBP Controller")
    simulate_main("sacbp", rng_num=169, animate=true, plotfig=true, verbose=false);
    println("3. Simulation with MCTS-DPW Controller")
    simulate_main("mcts", rng_num=169, animate=true, plotfig=true, verbose=false);
    println("4. Simulation with T-LQG Controller")
    simulate_main("tlqg", rng_num=169, animate=true, plotfig=true, verbose=false);
end
