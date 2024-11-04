from ChainReaction_environment.env.ChainReaction_environment import ChainReactionEnvironment

env = ChainReactionEnvironment(render_mode=None)
env.reset(seed=42)

for agent in env.agent_iter(max_iter=100): #pygame auto close in 10 secs
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()
    print(f'Action: {action} by agent: {agent}')
    env.step(action)

env.close()