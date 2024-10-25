from ChainReaction_environment.env.ChainReaction_environment import ChainReactionEnvironment

env = ChainReactionEnvironment(render_mode="human")
env.reset(seed=42)
i=0
for agent in env.agent_iter(max_iter=3): #pygame auto close in 10 secs
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()

    env.step(action)

env.close()