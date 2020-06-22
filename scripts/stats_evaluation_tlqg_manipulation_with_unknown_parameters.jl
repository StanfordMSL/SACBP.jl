ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python")
include(joinpath(@__DIR__, "stats_evaluation_manipulation_with_unknown_parameters.jl"))

function evaluate()
    data_dir = normpath(joinpath(@__DIR__, "../data",
                                 "manipulation_with_unknown_parameters"));
    data = load(joinpath(data_dir, "statistical_data_manipulation.jld2"))
    safe_rngnums = data["safe_rngnums"];

    cost_tlqg = Array{Float64}[]
    tcalc_tlqg = Array{Float64}[]
    tcalc_tlqg_offline = Array{Float64}[]

    time_idx_array = [];

    num_successful_cases = 0;

    for rngnum in safe_rngnums
        println("------- RNG Seed: $(rngnum) & #Cases Tested: $(num_successful_cases) -------")
        println("4. Simulation with T-LQG Controller")
        y_history,tcalc_true_history,b_history,x_history,u_history,U_pool,Cs,Cu =
        simulate_main("tlqg",rng_num=rngnum, animate=false, plotfig=false, verbose=false);
        push!(cost_tlqg,getMetricHistory(x_history));
        push!(tcalc_tlqg,tcalc_true_history);
        if isempty(time_idx_array)
            push!(time_idx_array, unique([x.t for x in x_history]))
        end

        println("5. Simulation with T-LQG Controller (Offline, for Computation Time Statistics Only)")
        ~,tcalc_true_history,~,~,~,~,~,~ =
        simulate_main("tlqg_offline",rng_num=rngnum, animate=false, plotfig=false, verbose=false);
        push!(tcalc_tlqg_offline,tcalc_true_history);
        if isempty(time_idx_array)
            push!(time_idx_array, unique([x.t for x in x_history]))
        end

        num_successful_cases += 1
    end

    tcalc_tlqg = vcat(tcalc_tlqg...);
    tcalc_tlqg_offline = vcat(tcalc_tlqg_offline...);

    FileIO.save(joinpath(data_dir, "statistical_data_tlqg_manipulation.jld2"),
            "cost_tlqg", cost_tlqg, "tcalc_tlqg", tcalc_tlqg,
            "tcalc_tlqg_offline", tcalc_tlqg_offline,
            "safe_rngnums", safe_rngnums, "time_idx_array", time_idx_array);

    cost_sacbp = data["cost_sacbp"]
    cost_mcts = data["cost_mcts"]
    cost_ilqg = data["cost_ilqg"]
    tcalc_sacbp = data["tcalc_sacbp"]
    tcalc_mcts = data["tcalc_mcts"]
    tcalc_ilqg = data["tcalc_ilqg"];

    # Plot Stats Data
    cost_std_sacbp = [std([cost[ii] for cost in cost_sacbp]) for ii = 1:length(cost_sacbp[1])]
    cost_std_mcts = [std([cost[ii] for cost in cost_mcts]) for ii = 1:length(cost_mcts[1])]
    cost_std_ilqg = [std([cost[ii] for cost in cost_ilqg]) for ii = 1:length(cost_ilqg[1])];
    cost_std_tlqg = [std([cost[ii] for cost in cost_tlqg]) for ii = 1:length(cost_tlqg[1])];

    CostHistorySACBP = mean(cost_sacbp)
    CostHistoryMCTS = mean(cost_mcts)
    plot(time_idx_array[1][1:end-1],CostHistoryMCTS,label="MCTS-DPW",color=:lightcoral,ribbon=cost_std_mcts,fillalpha=0.2,linewidth=2.,linestyle=:dash);
    CostHistoryiLQG = mean(cost_ilqg)
    CostHistoryTLQGOnline = mean(cost_tlqg);
    plot!(time_idx_array[1][1:end-1],CostHistoryTLQGOnline,label="T-LQG",color=:darkgreen,ribbon=cost_std_tlqg,fillalpha=0.1,linewidth=2.,linestyle=:dot)
    plot!(time_idx_array[1][1:end-1],CostHistoryiLQG,label="Belief iLQG",color=:darkorange,ribbon=cost_std_ilqg,fillalpha=0.2,linewidth=2.,linestyle=:dashdot)
    plot!(time_idx_array[1][1:end-1],CostHistorySACBP,label="SACBP (Ours)",color=:darkblue,ribbon=cost_std_sacbp,fillalpha=0.2,linewidth=2.,
          xlabel="Time [s]",ylabel="Residual Norm",size=(400,400),ylim=(0.,5.))
    savefig(joinpath(data_dir, "manipulation_costs_with_tlqg.pdf"))

    tcalcTrueArray = [tcalc_mcts,tcalc_tlqg,tcalc_sacbp]
    meanTcalcTrueArray = mean.(tcalcTrueArray);
    stdTcalcTrueArray = std.(tcalcTrueArray);

    bar([1,2,3],meanTcalcTrueArray,yerror=stdTcalcTrueArray,linecolor=:black,marker=stroke(1.5,:black, :dash),c=[:lightcoral,:palegreen3,:royalblue3],label="");
    xticks!([1,2,3],["MCTS-DPW","T-LQG","SACBP (Ours)"])
    plot!([0.5,3.5],[3/4*0.2,3/4*0.2],width=2.,label="Targeted",size=(300,400),color=:darkred,ylim=(-0.03,0.8))
    ylabel!("Online Replanning Time [s]")
    savefig(joinpath(data_dir, "manipulation_time_with_tlqg.pdf"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    evaluate();
end
