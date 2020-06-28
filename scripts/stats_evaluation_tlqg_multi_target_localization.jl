ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python")
include(joinpath(@__DIR__, "stats_evaluation_multi_target_localization.jl"))

function evaluate()
    data_dir = normpath(joinpath(@__DIR__, "../data",
                                 "multi_target_localization"));
    data = load(joinpath(data_dir, "statistical_data_localization_openloop.jld2"))
    ergodic_data = load(joinpath(data_dir, "statistical_data_localization_openloop_ergodic.jld2"));

    safe_rngnums = ergodic_data["safe_rngnums"];

    cost_tlqg = Array{Float64}[]
    tcalc_tlqg = Array{Float64}[]
    tcalc_tlqg_offline = Array{Float64}[]

    time_idx_array = [];

    num_successful_cases = 0;

    for rngnum in safe_rngnums
        println("------- RNG Seed: $(rngnum) & #Cases Tested: $(num_successful_cases) -------")
        println("4. Simulation with T-LQG Controller")
        y_history,tcalc_true_history,s_history,u_history,U_pool =
        simulate_main("tlqg",rng_num=rngnum, animate=false, plotfig=false, verbose=false);
        push!(cost_tlqg,getMetricHistory(s_history));
        push!(tcalc_tlqg,tcalc_true_history);
        if isempty(time_idx_array)
            push!(time_idx_array, unique([s.t for s in s_history]))
        end

        num_successful_cases += 1
    end

    println("5. Simulation with T-LQG Controller (Offline, for Computation Time Statistics Only)")
    rngnum = safe_rngnums[1]
    ~,tcalc_true_history,~,~,~ =
    simulate_main("tlqg_offline",rng_num=rngnum, animate=false, plotfig=false, verbose=false);
    push!(tcalc_tlqg_offline,tcalc_true_history);

    tcalc_tlqg = vcat(tcalc_tlqg...);
    tcalc_tlqg_offline = vcat(tcalc_tlqg_offline...);

    FileIO.save(joinpath(data_dir, "statistical_data_tlqg_localization_openloop.jld2"),
            "cost_tlqg", cost_tlqg, "tcalc_tlqg", tcalc_tlqg,
            "tcalc_tlqg_offline", tcalc_tlqg_offline,
            "safe_rngnums", safe_rngnums, "time_idx_array", time_idx_array);

    cost_greedy = data["cost_greedy"]
    cost_sacbp = data["cost_sacbp"]
    cost_mcts = data["cost_mcts"]
    cost_ergodic = ergodic_data["cost_ergodic"]
    tcalc_greedy = data["tcalc_greedy"]
    tcalc_sacbp = data["tcalc_sacbp"]
    tcalc_mcts = data["tcalc_mcts"]
    tcalc_ergodic = ergodic_data["tcalc_ergodic"];

    # Plot Stats Data
    cost_std_mcts = [std([cost[ii] for cost in cost_mcts]) for ii = 1:length(cost_mcts[1])]
    cost_std_sacbp = [std([cost[ii] for cost in cost_sacbp]) for ii = 1:length(cost_sacbp[1])]
    cost_std_greedy = [std([cost[ii] for cost in cost_greedy]) for ii = 1:length(cost_greedy[1])];
    cost_std_ergodic = [std([cost[ii] for cost in cost_ergodic]) for ii = 1:length(cost_ergodic[1])];
    cost_std_tlqg = [std([cost[ii] for cost in cost_tlqg]) for ii = 1:length(cost_tlqg[1])];

    MetricHistoryMCTS = mean(cost_mcts)
    plot(time_idx_array,MetricHistoryMCTS[1:end-1],label="MCTS-DPW",ribbon=cost_std_mcts,color=:lightcoral,fillalpha=0.2,linewidth=2.,linestyle=:dash)
    MetricHistoryTLQGOnline = mean(cost_tlqg);
    plot!(time_idx_array,MetricHistoryTLQGOnline[1:end-1],label="T-LQG",color=:darkgreen,ribbon=cost_std_tlqg,fillalpha=0.1,linewidth=2.,linestyle=:dot)
    MetricHistoryErgodic = mean(cost_ergodic)
    plot!(time_idx_array,MetricHistoryErgodic[1:end-1],label="Ergodic",ribbon=cost_std_ergodic,color=:darkorange,fillalpha=0.2,linewidth=2.,linestyle=:dashdot)
    MetricHistoryGreedy = mean(cost_greedy)
    plot!(time_idx_array,MetricHistoryGreedy[1:end-1],xlabel="Time [s]",ribbon=cost_std_greedy,fillalpha=0.2,ylabel="Worst Entropy Value",label="Greedy",color=:dimgrey,linewidth=2.,linestyle=:dot)
    MetricHistorySACBP = mean(cost_sacbp)
    plot!(time_idx_array,MetricHistorySACBP[1:end-1],label="SACBP (Ours)",ribbon=cost_std_sacbp,fillalpha=0.2,color=:darkblue,linewidth=2.,size=(400,400))
    savefig(joinpath(data_dir, "tracking_costs_with_tlqg.pdf"))

    # Plot Computation Time Comparison
    tcalcTrueArray = [tcalc_greedy,tcalc_mcts,tcalc_tlqg,tcalc_ergodic,tcalc_sacbp]
    meanTcalcTrueArray = mean.(tcalcTrueArray);
    stdTcalcTrueArray = std.(tcalcTrueArray);
    bar([1,2,3,4,5],meanTcalcTrueArray,yerror=stdTcalcTrueArray,linecolor=:black,marker=stroke(1.5,:black, :dash),c=[:dimgrey,:lightcoral,:palegreen3,:darkorange,:royalblue3],label="",size=(500,400));
    xticks!([1,2,3,4,5],["Greedy","MCTS-DPW","T-LQG","Ergodic","SACBP (Ours)"]);
    plot!([0.5,5.5],[3/4*0.2,3/4*0.2],width=2,label="Targeted",color=:darkred,ylim=(-0.03,0.8))
    ylabel!("Online Replanning Time [s]")
    savefig(joinpath(data_dir, "tracking_time_with_tlqg.pdf"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    evaluate();
end
