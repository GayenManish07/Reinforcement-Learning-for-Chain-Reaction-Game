from ChainReaction_environment.env.ChainReaction_environment import ChainReactionEnvironment
import torch as T
import numpy as np
from Algorithms.PPO.network import ActorNetwork
#from MADDPG.networks import ActorNetwork
var=1 
env = ChainReactionEnvironment(render_mode='human')
env.reset()   
if var==0:    

    model=ActorNetwork(0.01,1024,'P!','chkpt')
    #model=ActorNetwork(0.0003,'P!')# PPO
    #MADDPG\\checkpoints\\P0_actor
    model.load_state_dict(T.load('MADDPG\\checkpoints\\P0_actor', map_location=T.device('cpu')))
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
                action=np.argmax(action.detach().numpy())
                print('___________________________')
                print(action)
                print('___________________________')
                env.step(action)
            
        print(f'Action: {action} by agent: {agent}')


    env.close()

else:


    model=ActorNetwork(0.03,'P!')# PPO

    model.load_state_dict(T.load('Algorithms\\PPO\\checkpoints\\P2\\actor_torch_ppo', map_location=T.device('cpu')))
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
