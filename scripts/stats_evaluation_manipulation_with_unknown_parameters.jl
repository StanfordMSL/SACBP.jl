import Distributions
using Statistics

include(joinpath(@__DIR__, "simulation_manipulation_with_unknown_parameters.jl"))

function state_res_norm(x::PhysManipulate2D)
    xVec = vec(x)[1:6];
    xVec_target = [0.,0.,pi,0.,0.,0.];
    return norm(xVec - xVec_target)
end;

function getMetricHistory(x_history)
    CostArray = Float64[];
    for t = 1:length(x_history)-1
        push!(CostArray,state_res_norm(x_history[t]))
    end
    #push!(CostArray,termCost_true(CostManipulate2D(),x_history[end],Cs));
    return CostArray
end

function evaluate()
    safe_rngnums = [];
    rngnum = 191;
    num_successful_cases = 0;

    cost_sacbp = Array{Float64}[];
    tcalc_sacbp = Array{Float64}[];
    cost_mcts = Array{Float64}[];
    tcalc_mcts = Array{Float64}[];
    cost_ilqg = Array{Float64}[];
    tcalc_ilqg = Array{Float64}[];

    time_idx_array = [];

    while num_successful_cases < 20
        println("------- RNG Seed: $(rngnum) & #Cases Tested: $(num_successful_cases) -------")
        println("1. Simulation with SACBP Controller")
        try
            y_history,tcalc_true_history,b_history,x_history,u_history,U_pool,Cs,Cu =
            simulate_main("sacbp",rng_num=rngnum, animate=false, plotfig=false, verbose=false);
            push!(cost_sacbp,getMetricHistory(x_history));
            push!(tcalc_sacbp,tcalc_true_history);
            if isempty(time_idx_array)
                push!(time_idx_array, unique([x.t for x in x_history]))
            end
        catch
            rngnum += 1
            continue
        end
        println("2. Simulation with MCTS Controller")
        try
            y_history,tcalc_true_history,b_history,x_history,u_history,U_pool,Cs,Cu =
            simulate_main("mcts",rng_num=rngnum, animate=false, plotfig=false, verbose=false);
            push!(cost_mcts,getMetricHistory(x_history));
            push!(tcalc_mcts,tcalc_true_history);
            if isempty(time_idx_array)
                push!(time_idx_array, unique([x.t for x in x_history]))
            end
        catch
            pop!(cost_sacbp)
            pop!(tcalc_sacbp)
            rngnum += 1
            continue
        end
        println("3. Simulation with Belief iLQG Controller")
        try
            y_history,tcalc_true_history,b_history,x_history,u_history,U_pool,Cs,Cu =
            simulate_main("ilqg",rng_num=rngnum, animate=false, plotfig=false, verbose=false);
            push!(cost_ilqg,getMetricHistory(x_history));
            push!(tcalc_ilqg,tcalc_true_history);
            if isempty(time_idx_array)
                push!(time_idx_array, unique([x.t for x in x_history]))
            end
        catch
            pop!(cost_sacbp)
            pop!(tcalc_sacbp)
            pop!(cost_mcts)
            pop!(tcalc_mcts)
            rngnum += 1
            continue
        end
        append!(safe_rngnums,rngnum)
        rngnum += 1;
        num_successful_cases += 1;
    end

    tcalc_sacbp = vcat(tcalc_sacbp...);
    tcalc_mcts = vcat(tcalc_mcts...);
    tcalc_ilqg = vcat(tcalc_ilqg...);

    data_dir = normpath(joinpath(@__DIR__, "../data",
                                 "manipulation_with_unknown_parameters"));

    FileIO.save(joinpath(data_dir, "statistical_data_manipulation.jld2"),
                "cost_sacbp",cost_sacbp,"tcalc_sacbp",tcalc_sacbp,
                "cost_mcts",cost_mcts,"tcalc_mcts",tcalc_mcts,
                "cost_ilqg",cost_ilqg,"tcalc_ilqg",tcalc_ilqg,
                "safe_rngnums",safe_rngnums)

    # Plot Stats Data
    cost_std_sacbp = [std([cost[ii] for cost in cost_sacbp]) for ii = 1:length(cost_sacbp[1])]
    cost_std_mcts = [std([cost[ii] for cost in cost_mcts]) for ii = 1:length(cost_mcts[1])]
    cost_std_ilqg = [std([cost[ii] for cost in cost_ilqg]) for ii = 1:length(cost_ilqg[1])];
    # # In this problem, cost is w.r.t. the true state history.
    CostHistoryMCTS = mean(cost_mcts)
    plot(time_idx_array[1][1:end-1],CostHistoryMCTS,xlabel="Time [s]",ribbon=cost_std_mcts,fillalpha=0.3,ylabel="Residual Norm",label="MCTS-DPW",color=:lightcoral,linewidth=2.,linestyle=:dash);
    CostHistoryiLQG = mean(cost_ilqg)
    plot!(time_idx_array[1][1:end-1],CostHistoryiLQG,label="Belief iLQG",color=:darkorange,ribbon=cost_std_ilqg,fillalpha=0.3,linewidth=2.,linestyle=:dashdot)
    CostHistorySACBP = mean(cost_sacbp)
    plot!(time_idx_array[1][1:end-1],CostHistorySACBP,label="SACBP",color=:darkblue,ribbon=cost_std_sacbp,fillalpha=0.3,linewidth=2.,size=(400,400),ylim=(0.,5.))
    savefig(joinpath(data_dir, "manipulation_costs.pdf"))
    # Plot Computation Time Comparison
    tcalcTrueArray = [tcalc_ilqg,tcalc_mcts,tcalc_sacbp]
    meanTcalcTrueArray = mean.(tcalcTrueArray);
    stdTcalcTrueArray = std.(tcalcTrueArray);
    bar([1,2,3],meanTcalcTrueArray,yerror=stdTcalcTrueArray,linecolor=:black,marker=stroke(4.0,:black, :dash),c=[:darkorange,:lightcoral,:darkblue],label="");
    xticks!([1,2,3],["Belief iLQG","MCTS-DPW","SACBP"]);
    plot!([0.5,3.5],[3/4*0.2,3/4*0.2],width=2.,label="Targeted",size=(300,400),color=:darkred,ylim=(-0.03,1.1))
    ylabel!("Control Computation Time [s]")
    savefig(joinpath(data_dir, "manipulation_time.pdf"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    evaluate();
end
