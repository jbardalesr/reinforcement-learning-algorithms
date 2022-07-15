using Random, Distributions
using PyPlot

function upper_confidence(k_arm::Int64, runs::Int64, time::Int64, c::Float64)
    # k-armed bandit with upper-confidence
    collect_reward = zeros(Float64, (runs, time))
    # collect_op_action = zeros(Float64, (runs, time))
    action_list = collect(1:k_arm)
    for run=1:runs
        q_true = rand(Normal(0, 1), k_arm)
        # optimal_action = argmax(q_true, dims=1)
        q_estimated = zeros(Float64, k_arm)
        action_count = zeros(Int64, k_arm)
        for t=1:time
            action = argmax(q_estimated + c*sqrt.(log(t)./action_count))
            reward = rand(Normal(q_true[action], 1))
            action_count[action] += 1
            q_estimated[action] += (reward - q_estimated[action])/action_count[action]
            collect_reward[run, t] = reward
        end
    end
    sum(collect_reward, dims=1)/runs
end

time = 1000
runs = 2000
c = 2.0
k_arm = 10

collected_reward = upper_confidence(k_arm, runs, time, c)
plot(1:time, vec(collected_reward), linestyle="-")
title("Upper Confidence")
xlabel("steps")
ylabel("Average reward")
legend(["c=$c"])
show()