# SACBP.jl
Stochastic Sequential Action Control for Continuous-Time Belief Space Planning in Julia.

## Julia Version
* v1.3
* Distributions, ForwardDiff, Plots, Convex, ECOS, POMDPs, MCTS, PyCall

## External Dependencies
* Python v3.6+ (only for running [T-LQG](https://ieeexplore.ieee.org/document/7989080))
- Python requirements are found in python_requirements.txt

## Note on PyCall Setup for testing T-LQG
- [Specify the python version (3.6) in PyCall.jl](https://github.com/JuliaPy/PyCall.jl#specifying-the-python-version).
