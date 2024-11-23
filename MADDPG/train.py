import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from ChainReaction_environment.env.ChainReaction_environment import ChainReactionEnvironment
env = ChainReactionEnvironment(render_mode=None)
env.reset()

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

if __name__ == '__main__':
    #scenario = 'simple'
    
    env.reset()
    observation=env.board

    actor_dims = []
    for i in range(2):

        actor_dims.append(observation)
    critic_dims = sum(actor_dims)

    critic_dims = critic_dims.shape

    # action space is a list of arrays, assume each agent has same action space
    n_actions = 100#env.action_space[0].n
    maddpg_agents = MADDPG( env, chkpt_dir='checkpoints/',
                           fc1=1024, fc2=1024,  
                           alpha=0.01, beta=0.01
                           )

    memory = MultiAgentReplayBuffer(100000, critic_dims, actor_dims, 
                        n_actions, 2, batch_size=1024)

    PRINT_INTERVAL = 10
    N_GAMES = 10
    MAX_STEPS = 100
    total_steps = 0
    score_history = []
    evaluate = False
    best_score = [0,0]#{'P1':0,'P2':0}

    if evaluate:
        maddpg_agents.load_checkpoint()

    for i in range(N_GAMES):
        env.reset()
        obs= env.board
        score = [0,0]
        done = False
        episode_steps = 0

        while not done:
            if evaluate:
                env.render()
                #time.sleep(0.1) # to slow down the action for the video
            actions = maddpg_agents.choose_action(obs)
            #print(done)
            print(f'Action: {actions} by agent: {env.agent_selection}')
            if episode_steps%2==0:
                env.step(actions[0])
            else:
                env.step(actions[1])
            obs_=env.board
            reward=env.rewards

            done=any(env.terminations.values())

            state = obs#obs_list_to_state_vector(obs)
            state_ = obs_#obs_list_to_state_vector(obs_)

            #print(done)
            if episode_steps >= MAX_STEPS:
                done = True

            memory.store_transition(obs, state, actions, reward, obs_, state_, done)

            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            obs = obs_

            score = list(x+y for x,y in zip(score,reward.values()))

            total_steps += 1
            episode_steps += 1
            


        score_history.append(score)


        avg_score=[sum(col) / len(col) for col in zip(*score_history)]
        #avg_score = {'P1':np.mean(score_history[0][0][-1:]),'P2':np.mean(score_history[0][1][-1:])}

        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:

            print(f"Episode {i}: Average Player Scores: {avg_score}")
