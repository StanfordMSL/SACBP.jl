import Distributions
using Statistics

include(joinpath(@__DIR__, "simulation_multi_target_localization.jl"))

function getMetricHistory(s_history)
    # Evaluation Metric: Worst Entropy
    MetricArray = Float64[];
    for t = 1:length(s_history)
        bVec = Distributions.MvNormal.(s_history[t].b)
        push!(MetricArray,maximum(Distributions.entropy.(bVec)));
    end
    return MetricArray
end

function evaluate()
    safe_rngnums = [];
    rngnum = 133;
    num_successful_cases = 0;

    cost_greedy = Array{Float64}[];
    tcalc_greedy = Array{Float64}[];
    cost_sacbp = Array{Float64}[];
    tcalc_sacbp = Array{Float64}[];
    cost_mcts = Array{Float64}[];
    tcalc_mcts = Array{Float64}[];

    time_idx_array = [];

    while num_successful_cases < 20
        println("------- RNG Seed: $(rngnum) & #Cases Tested: $(num_successful_cases) -------")
        println("1. Simulation with Greedy Controller")
        try
            y_history,tcalc_true_history,s_history,u_history,U_pool =
            simulate_main("greedy",rng_num=rngnum, animate=false, plotfig=false, verbose=false);
            push!(cost_greedy,getMetricHistory(s_history));
            push!(tcalc_greedy,tcalc_true_history);
            if isempty(time_idx_array)
                push!(time_idx_array, unique([s.t for s in s_history]))
            end
        catch
            rngnum += 1
            continue
        end
        println("2. Simulation with SACBP Controller")
        try
            y_history,tcalc_true_history,s_history,u_history,U_pool =
            simulate_main("sacbp",rng_num=rngnum, animate=false, plotfig=false, verbose=false);
            push!(cost_sacbp,getMetricHistory(s_history));
            push!(tcalc_sacbp,tcalc_true_history);
            if isempty(time_idx_array)
                push!(time_idx_array, unique([s.t for s in s_history]))
            end
        catch
            pop!(cost_greedy)
            pop!(tcalc_greedy)
            rngnum += 1
            continue
        end
        println("3. Simulation with MCTS Controller")
        try
            y_history,tcalc_true_history,s_history,u_history,U_pool =
            simulate_main("mcts",rng_num=rngnum, animate=false, plotfig=false, verbose=false);
            push!(cost_mcts,getMetricHistory(s_history));
            push!(tcalc_mcts,tcalc_true_history);
            if isempty(time_idx_array)
                push!(time_idx_array, unique([s.t for s in s_history]))
            end
        catch
            pop!(cost_greedy)
            pop!(tcalc_greedy)
            pop!(cost_sacbp)
            pop!(tcalc_sacbp)
            rngnum += 1
            continue
        end
        append!(safe_rngnums,rngnum)
        rngnum += 1;
        num_successful_cases += 1;
    end

    tcalc_greedy = vcat(tcalc_greedy...);
    tcalc_sacbp = vcat(tcalc_sacbp...);
    tcalc_mcts = vcat(tcalc_mcts...);

    data_dir = normpath(joinpath(@__DIR__, "../data",
                                 "multi_target_localization"));
    # For ergodic control, the ErgodicControl.jl package is not compatible with Julia v1.3
    # Therefore, we use previously collected data with Julia v0.6 for the WAFR paper.
    ergodic_data = FileIO.load(joinpath(data_dir, "statistical_data_localization_openloop_ergodic.jld2"));
    cost_ergodic, tcalc_ergodic = ergodic_data["cost_ergodic"], ergodic_data["tcalc_ergodic"]

    FileIO.save(joinpath(data_dir, "statistical_data_localization_openloop.jld2"),
                "cost_greedy",cost_greedy,"tcalc_greedy",tcalc_greedy,
                "cost_sacbp",cost_sacbp,"tcalc_sacbp",tcalc_sacbp,
                "cost_mcts",cost_mcts,"tcalc_mcts",tcalc_mcts,
                "cost_ergodic",cost_ergodic,"tcalc_ergodic",tcalc_ergodic)

    # Plot Stats Data
    cost_std_mcts = [std([cost[ii] for cost in cost_mcts]) for ii = 1:length(cost_mcts[1])]
    cost_std_sacbp = [std([cost[ii] for cost in cost_sacbp]) for ii = 1:length(cost_sacbp[1])]
    cost_std_greedy = [std([cost[ii] for cost in cost_greedy]) for ii = 1:length(cost_greedy[1])];
    cost_std_ergodic = [std([cost[ii] for cost in cost_ergodic]) for ii = 1:length(cost_ergodic[1])];

    MetricHistoryGreedy = mean(cost_greedy)
    plot(time_idx_array,MetricHistoryGreedy[1:end-1],xlabel="Time [s]",ribbon=cost_std_greedy,fillalpha=0.3,ylabel="Worst Entropy Value",label="Greedy",color=:dimgrey,linewidth=2.,linestyle=:dot)
    MetricHistoryMCTS = mean(cost_mcts)
    plot!(time_idx_array,MetricHistoryMCTS[1:end-1],label="MCTS-DPW",ribbon=cost_std_mcts,color=:lightcoral,fillalpha=0.3,linewidth=2.,linestyle=:dash)
    MetricHistoryErgodic = mean(cost_ergodic)
    plot!(time_idx_array,MetricHistoryErgodic[1:end-1],label="Ergodic",ribbon=cost_std_ergodic,color=:darkorange,fillalpha=0.3,linewidth=2.,linestyle=:dashdot)
    MetricHistorySACBP = mean(cost_sacbp)
    plot!(time_idx_array,MetricHistorySACBP[1:end-1],label="SACBP",ribbon=cost_std_sacbp,fillalpha=0.3,color=:darkblue,linewidth=2.,size=(400,400))
    savefig(joinpath(data_dir, "tracking_costs.pdf"))

    # Plot Computation Time Comparison
    tcalcTrueArray = [tcalc_greedy,tcalc_mcts,tcalc_ergodic,tcalc_sacbp]
    meanTcalcTrueArray = mean.(tcalcTrueArray);
    stdTcalcTrueArray = std.(tcalcTrueArray);
    bar([1,2,3,4],meanTcalcTrueArray,yerror=stdTcalcTrueArray,linecolor=:black,marker=stroke(4.0,:black, :dash),c=[:dimgrey,:lightcoral,:darkorange,:darkblue],label="",size=(400,400));
    xticks!([1,2,3,4],["Greedy","MCTS-DPW","Ergodic","SACBP"]);
    plot!([0.5,4.5],[3/4*0.2,3/4*0.2],width=2,label="Targeted",color=:darkred)
    ylabel!("Control Computation Time [s]")
    savefig(joinpath(data_dir, "tracking_time.pdf"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    evaluate();
end
