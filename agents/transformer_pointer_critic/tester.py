from agents.double_pointer_critic.optimum_solver import solver

def test(env, agent):
    data = env.convert_to_ortools_input()
    solver(data)

    # for game in range(20):

    #     rewards = 0

    #     current_state = env.reset()
    #     # print(f"Step {episode} | State {current_state} | Reward {rewards}")

    #     actions, _, _ = agent.act(current_state)

    #     _, rewards, _, _ = env.multiple_steps(actions)
    #     rewards = sum(rewards)
        
    #     env.print_stats()

    #     print(f"Game {game} finished with reward: {rewards}")