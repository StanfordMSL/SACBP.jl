using Distributed
# using 6 cores
if 6 - Distributed.nprocs() > 0
    Distributed.addprocs(6 - Distributed.nprocs());
end
@everywhere using SACBP
using SACBP
using LinearAlgebra
using Plots
pyplot()
using ProgressMeter
using Printf
using Random
using JLD2
using FileIO

include(joinpath(@__DIR__, "../baseline_implementations/p_control.jl"))
include(joinpath(@__DIR__, "../baseline_implementations/mcts.jl"))
include(joinpath(@__DIR__, "../baseline_implementations/belief_ilqg.jl"))

function get_initial_true_state()
    px0_true = 1.5;    # True px(0)
    py0_true = 1.;     # True py(0)
    θ0_true = 11*pi/6.;# True theta(0) in (0,2*pi)
    vx0_true = 0.;     # True vx(0)
    vy0_true = 0.;     # True vy(0)
    ω0_true = 0.;      # True ω(0)
    m_true = 2.0;      # True mass
    a_true = 0.5;      # Height of the Plate.
    b_true = 1.0;      # Width of the Plate.
    J_true = 1/3*m_true*(a_true^2 + b_true^2); # Moment of Inertia
    rx_true = 0.25;    # Arm Length rx
    ry_true = 0.25;    # Arm Length ry
    μ_true = 4.0;      # Linear Friction Coefficient
    x_init = PhysManipulate2D(0.,[px0_true,py0_true],θ0_true,[vx0_true,vy0_true],ω0_true,m_true,J_true,[rx_true,ry_true],μ_true);
    return x_init, a_true, b_true
end

function get_initial_belief_state()
    μ_init = [4.,4.,pi/4,0.1,-0.1,pi/10,6.,2.,0.3,0.1,7.0];
    Σ_init = diagm([10.,10.,(pi/2)^2,2.,2.,(pi/4)^2,1.,1.0,1.0,1.0,5.0]);
    b_init = BelMvNormal(0.,μ_init,Σ_init);
    return b_init
end

function visualizeSimEnv(px::Real,py::Real,θ::Real,vx::Real,vy::Real,ω::Real,
                         a::Real,b::Real,rx::Real,ry::Real,dto=0.0;
                         plot_velocity=false)
    plt = plot();
    Vertices = [[-b/2,a/2],[-b/2,-a/2],[b/2,-a/2],[b/2,a/2]];
    xVertices = map(v -> v[1], Vertices);
    yVertices = map(v -> v[2], Vertices);
    VRotated = map(v -> [cos(θ) -sin(θ); sin(θ) cos(θ)]*v + [px, py], Vertices);
    xRotated = map(v -> v[1], VRotated);
    yRotated = map(v -> v[2], VRotated);
    Robot = [cos(θ) -sin(θ); sin(θ) cos(θ)]*[rx, ry] + [px, py];
    plot!(xRotated,yRotated,seriestype=:shape,color=:cyan,linecolor=:darkcyan,alpha=0.4,aspect_ratio=1.0,label="Object",xlabel="x [m]",ylabel="y [m]");
    scatter!((px,py),color=:darkcyan,markersize=5.0,label="C.M.")
    scatter!((Robot[1],Robot[2]),color=:red,markersize=10.0,label="Robot")
    if plot_velocity
        plot!([px,px+dto*vx],[py,py+dto*vy],line=:arrow,color=:green,width=2.0,label="Lin. Vel.")
        plot!([px + min(a,b)/4*cos(pi/2+ii) for ii = range(0.,stop=5*dto*ω,length=10)],
              [py + min(a,b)/4*sin(pi/2+ii) for ii = range(0.,stop=5*dto*ω,length=10)],line=:arrow,color=:orange,width=2.0,label="Ang. Vel.");
    end
    plot!(xVertices,yVertices,seriestype=:shape,color=:black,linecolor=:black,alpha=0.1,label="Goal",size=(500,400))
    xlims!((-1.,2.3))
    ylims!((-1.,1.5))
    return plt
end

function simulate(b_init::BelMvNormal{<:Real},
                  x_init::PhysManipulate2D{<:Real},
                  a_true::Real, b_true::Real,
                  controlMode::String, rng_num::Int64,
                  simModel::SimulateManipulate2D;
                  numSamples=10, # for sacbp only
                  u_param_min::Vector{Float64}=[-3.0, -3.0, -3.0],
                  u_param_max::Vector{Float64}=[3.0, 3.0, 3.0],
                  posGain::Real,
                  rotGain::Real,
                  Th::Float64=2.0, # planning horizon [s]
                  tf::Float64=20.0, # simulation horizon [s]
                  mcts_param::Union{Nothing, MCTSParams}=nothing,
                  ilqgPolicy::Union{Nothing, Dict{String,Any}}=nothing,
                  animate=true, plotfig=true, verbose=true)
    rng = MersenneTwister(rng_num)
    @assert controlMode in ["sacbp", "mcts", "ilqg"];
    if animate
        # Animation Object.
        anim = Animation();
    end
    # Only for MCTS policies.
    if controlMode == "mcts"
        @assert !isnothing(mcts_param) "MCTS Parameters not given"
        bmdp = BelMDP(simModel, u_param_min, u_param_max, mcts_param.numActions)
        mctsSolverPControl = MCTS.DPWSolver(depth=Int64(Th/simModel.dto),
                                            n_iterations=mcts_param.numSamples,
                                            exploration_constant=mcts_param.expConst,
                                            k_action=mcts_param.kAct,
                                            alpha_action=mcts_param.αAct,
                                            k_state=mcts_param.kState,
                                            alpha_state=mcts_param.αState,
                                            rng=rng,
                                            estimate_value=RolloutEstimator(PControl(u_param_min, u_param_max, posGain, rotGain)));
        mctsPolicyPControl = solve(mctsSolverPControl,bmdp);
    end

    contTransModel = simModel.continuousTransModel
    discTransModel = simModel.discreteTransModel
    dtc, dto = simModel.dtc, simModel.dto
    Q, R, Cs, Cu = simModel.Q, simModel.R, simModel.Cs, simModel.Cu

    # Controller Compilation.
    U_default = [MControl2D(round(dtc*t,digits=5),zeros(3)) for t = 0:Int64(Th/dtc)-1];
    if controlMode == "sacbp" # SACBP (with PControl as nominal)
        @time U_default = pControlUpdate(simModel,b_init,U_default,u_param_min,u_param_max,posGain,rotGain);
        @time U_default = sacControlUpdate(simModel,b_init,U_default,u_param_min,u_param_max,numSamples,rng)[1];
    elseif controlMode == "mcts" # MCTS (with PControl as nominal)
        @time U_default = mctsControlUpdate(mctsPolicyPControl,b_init,U_default,dto)
    elseif controlMode == "ilqg" # Belief iLQG (initialized with PControl)
        @assert !isnothing(ilqgPolicy) "iLQG Policy file not given"
        @time U_default = ilqgControlUpdate(ilqgPolicy,b_init,U_default,u_param_min,u_param_max,dto,dto)
    end
    println("Controller Compiled. Starting Simulation...")

    # Simulation.
    y_history = Vector{Float64}[];
    tcalc_true_history = Float64[];
    x_history = [x_init];
    b_history = [b_init];
    u_history = MControl2D{Float64}[];
    U_pool = copy(U_default);

    observeModel = ObserveModel_Manipulate2D();
    numSteps = Int64(tf/dtc);
    nextSampleTime = dto;
    @showprogress 1 for t = 1:numSteps
        if round(t*dtc,digits=5) > nextSampleTime
        # Belief & Control Update.
            x = x_history[end];
            if animate
                plt = visualizeSimEnv(x.pos[1],x.pos[2],x.θ,x.vel[1],x.vel[2],x.ω,a_true,b_true,x.r[1],x.r[2],dto,plot_velocity=true)
                time_str = @sprintf "%3.2f" x.t
                title!("Time: $(time_str) [s]")
                Plots.frame(anim);
            end
            u = u_history[end];
            # Observation.
            yVec = observe(observeModel,x,MControl2D(x.t,vec(u)),R,rng);
            push!(y_history,yVec);
            # Belief Update.
            b = pop!(b_history);
            b_new = trans(discTransModel,b,MControl2D(b.t,vec(u)),yVec,R);
            push!(b_history,b_new);
            if verbose
                println((b_new.t,"Belief Updated."));
            end
            # Control Update
            if controlMode == "sacbp";  # SACBP (with PControl)
                tcalc_true = @elapsed U_pool = pControlUpdate(simModel,b_new,U_pool,u_param_min,u_param_max,posGain,rotGain);
                tcalc_true += @elapsed U_pool,act_time_init,act_time_final,tcalc_simulated = sacControlUpdate(simModel,b_new,U_pool,u_param_min,u_param_max,numSamples,rng,offline=true);
                push!(tcalc_true_history,tcalc_true);
                if verbose
                    println((b.t,"Control Updated.","time_init=$(act_time_init)","time_final=$(act_time_final)"));
                end
            elseif controlMode == "mcts" # MCTS (with PControl)
                tcalc_true = @elapsed U_pool = mctsControlUpdate(mctsPolicyPControl,b_new,U_pool,dto);
                push!(tcalc_true_history,tcalc_true);
                if verbose
                    println((s.t,"Control Updated."));
                end
            elseif controlMode == "ilqg" # Belief iLQG (with PControl)
                tcalc_true = @elapsed U_pool = ilqgControlUpdate(ilqgPolicy,b_new,U_pool,u_param_min,u_param_max,dto,dto);
                push!(tcalc_true_history,tcalc_true);
                if verbose
                    println((s.t,"Control Updated."));
                end
            end
            # Control Execution.
            u = popfirst!(U_pool);
            push!(u_history,u);
            x_new = trans(TransModel_Manipulate2D(),x,u,dtc);
            push!(x_history,x_new);
            b_new = trans(contTransModel,b_new,MControl2D(b.t,vec(u)),dtc,Q);
            push!(b_history,b_new);

            # Append nominal control.
            u_nominal = MControl2D(round(b.t+Th,digits=5),zeros(3));
            push!(U_pool,u_nominal);
            nextSampleTime = round(nextSampleTime+dto,digits=5);
        else
            # No Control Update.
            b = b_history[end];
            # Control Execution.
            x = x_history[end];
            if animate && round(x.t/0.02, digits=5) == round(x.t/0.02)
                plt = visualizeSimEnv(x.pos[1],x.pos[2],x.θ,x.vel[1],x.vel[2],x.ω,a_true,b_true,x.r[1],x.r[2],dto,plot_velocity=true)
                time_str = @sprintf "%3.2f" x.t
                title!("Time: $(time_str) [s]")
                Plots.frame(anim);
            end
            u = popfirst!(U_pool);
            push!(u_history,u);
            x_new = trans(TransModel_Manipulate2D(),x,u,dtc);
            push!(x_history,x_new);
            b_new = trans(contTransModel,b,u,dtc,Q);
            push!(b_history,b_new);

            # Append nominal control.
            u_nominal = MControl2D(round(b.t+Th,digits=5),zeros(3));
            push!(U_pool,u_nominal);
        end
    end
    # Observation.
    x = x_history[end];
    if animate
        plt = visualizeSimEnv(x.pos[1],x.pos[2],x.θ,x.vel[1],x.vel[2],x.ω,a_true,b_true,x.r[1],x.r[2],dto,plot_velocity=true)
        time_str = @sprintf "%3.2f" x.t
        title!("Time: $(time_str) [s]")
        Plots.frame(anim);
    end
    u = u_history[end];
    # Observation.
    yVec = observe(observeModel,x,MControl2D(x.t,vec(u)),R,rng);
    push!(y_history,yVec);
    # Belief Update.
    b = pop!(b_history);
    b_new = trans(discTransModel,b,MControl2D(x.t,vec(u)),yVec,R);
    push!(b_history,b_new);
    if verbose
        println((b_new.t,"Belief Updated."));
    end
    log_dir = normpath(joinpath(@__DIR__, "../data",
                                "manipulation_with_unknown_parameters"));
    if animate
        gif(anim,joinpath(log_dir, "$(controlMode)_tf_$(tf)_rng_$(rng_num).gif"),fps=50)
        #gif(anim,joinpath(log_dir, "$(controlMode)_tf_$(tf)_rng_$(rng_num).gif"),fps=5)
    end
    return y_history,tcalc_true_history,b_history,x_history,u_history,U_pool
end

function simulate_main(controlMode::String; rng_num::Int64,
                       animate=true, plotfig=true, verbose=true)
    # PControl Parameters
    posGain = 1.0;
    rotGain = 0.5;

    # SAC Controller Parameters.
    dtc = 0.01;        # Euler Approximation Interval.
    dto = 0.2;         # Observation & Control Update Interval.
    Th = 2.0;          # Planning Horizon.
    dtexec = 0.2*dto;  # Control Execution Duration.
    Q = diagm([0.05, 0.05, 1.0*pi/180, 0.05, 0.05, 1.0*pi/180., 0., 0., 0., 0., 0.]); # state: [px, py, θ, vx, vy, ω, m, J, rx, ry, μ]
    R = diagm([0.1, 0.1, 5.0*pi/180, 0.1, 0.1, 5.0*pi/180., 0.2, 0.2, 10.0*pi/180.]) # observation: [px, py, θ, vx, vy, ω, ax, ay(linear acceleration), α(angular acceleration)]
    Cs = diagm([20.0, 20.0, 20.0, 15.0, 15.0, 15.0, 0., 0., 0., 0., 0.]); # State Cost Coefficient.
    Cu = diagm([0.1, 0.1, 0.1]);  # Control Cost Coefficient.
    u_param_min = [-3.,-3.,-3.];  # Lower Limit for PosControl
    u_param_max = [3.,3.,3.];     # Upper Limit for PosControl
    contTransModel = CPredictManipulate2D();
    discTransModel = DUpdateManipulate2D();
    costModel = CostManipulate2D();
    simModel = SimulateManipulate2D(contTransModel,discTransModel,costModel,dtc,dto,dtexec,Q,R,Cs,Cu);
    numSamples = 10;     # Number of Monte Carlo Samples for SACBP.

    # MCTS Parameters
    numActions = 7;             # Number of Actions per Coordinate.
    numSamples_mcts = 240;
    expConst = 100.0            # Exploration Constant.
    kAct = 8.0;                 # Action k.
    αAct = 0.2;                 # Action α.
    kState = 8.0;               # State k.
    αState = 0.2;               # State α.
    mcts_param = MCTSParams(numActions, numSamples_mcts, expConst, kAct, αAct, kState, αState);

    # iLQG Policy
    ilqg_policy_path = normpath(joinpath(@__DIR__, "../baseline_implementations",
                                         "ilqg_pcontrol_policy_20sec_2.jld2"));
    ilqgPolicy = FileIO.load(ilqg_policy_path);

    # Simulation Parameters
    tf = 20.;    # Simulation Time.

    x_init, a_true, b_true = get_initial_true_state();
    if plotfig
        frame = visualizeSimEnv(x_init.pos[1],x_init.pos[2],x_init.θ,
                                x_init.vel[1],x_init.vel[2],x_init.ω,
                                a_true, b_true, x_init.r[1], x_init.r[2])
        init_fig_path = normpath(joinpath(@__DIR__, "../data",
                                          "manipulation_with_unknown_parameters",
                                          "manipulation_initial.pdf"))
        savefig(init_fig_path);
    end

    b_init = get_initial_belief_state();

    y_history,tcalc_true_history,b_history,x_history,u_history,U_pool =
        simulate(b_init, x_init, a_true, b_true, controlMode, rng_num,
                 simModel, numSamples=numSamples,
                 u_param_min=u_param_min, u_param_max=u_param_max,
                 posGain=posGain, rotGain=rotGain,
                 Th=Th, tf=tf, mcts_param=mcts_param,
                 ilqgPolicy=ilqgPolicy,
                 animate=animate, plotfig=plotfig, verbose=verbose);
    println("Average control computation time: $(round(mean(tcalc_true_history), digits=3)) [s]")
    return y_history,tcalc_true_history,b_history,x_history,u_history,U_pool, Cs, Cu
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("1. Simulation with SACBP Controller")
    simulate_main("sacbp", rng_num=202, animate=true, plotfig=true, verbose=false);
    println("2. Simulation with MCTS-DPW Controller")
    simulate_main("mcts", rng_num=202, animate=true, plotfig=true, verbose=false);
    println("3. Simulation with Belief iLQG Controller")
    simulate_main("ilqg", rng_num=202, animate=true, plotfig=true, verbose=false);
end
