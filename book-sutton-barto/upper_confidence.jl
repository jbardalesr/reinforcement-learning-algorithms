using Random, Distributions

function upper_confidence(k_arm::Int64, epsilon::Float64, runs::Int64, time::Int64)
    # collect_reward = zeros(Float64, (runs, time))
    # collect_op_action = zeros(Float64, (runs, time))

    action_list = collect(1:10)
    
    for run=1:runs
        q_true = rand(Normal(0, 1), k_arm)
        # optimal_action = argmax(q_true, dims=1)

        q_estimated = zeros(Float64, k_arm)
        action_count = zeros(Int64, k_arm)

        for t=1:time
            if rand(Uniform(0, 1)) < 1 - epsilon
                action = argmax(q_estimated)
            else
                action = sample(action_list)
            end
            reward = rand(Normal(q_true[action], 1))
            action_count[action] = action_count[action] + 1

            q_estimated[action] += (reward - q_estimated[action])/action_count[action]
            
            println(reward)
        end
    end 
end

upper_confidence(10, 0.1, 1, 1000)
