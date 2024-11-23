from ChainReaction_environment.env.ChainReaction_environment import ChainReactionEnvironment
env = ChainReactionEnvironment(render_mode='human')
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        action = info
    print(f'Action: {action} by agent: {agent}')
    env.step(action)

env.close()