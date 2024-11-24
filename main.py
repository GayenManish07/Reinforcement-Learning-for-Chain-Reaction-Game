from Algorithms.PPO.ChainReaction_environment.env.ChainReaction_environment import ChainReactionEnvironment
import torch as T
import numpy as np
from Algorithms.PPO.network import ActorNetwork
env = ChainReactionEnvironment(render_mode='human')

env.reset()

model=ActorNetwork(0.0003,'P!')

model.load_state_dict(T.load('Algorithms\\PPO\\checkpoints\\P1\\actor_torch_ppo', map_location=T.device('cpu')))
model.eval()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        if agent=='P1':
            
            action = info
            env.step(action)
        else:
            observation=env.board
            observation=T.tensor(observation,dtype=T.float).unsqueeze(dim=0)
            action =model.forward(observation)
            action=np.argmax(action)
            print('___________________________')
            print(action)
            print('___________________________')
            env.step(action)
        
    print(f'Action: {action} by agent: {agent}')


env.close()