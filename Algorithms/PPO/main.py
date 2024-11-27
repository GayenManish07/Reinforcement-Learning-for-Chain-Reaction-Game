from ChainReaction_environment.env.ChainReaction_environment import ChainReactionEnvironment
import numpy as np
from agent import Agent
from utils import plot_learning_curve,plot_learning_curve2

if __name__ == '__main__':
    env = ChainReactionEnvironment()
    N = 30
    batch_size = 32
    n_epochs = 4
    alpha = 0.05
    agent = Agent(batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs)

    n_games = 3000

#P:\MARL_project\Reinforcement-Learning-for-Chain-Reaction-Game\Algorithms\PPO\plots
    figure_file1 = 'plots/agent_steps_ppo.png'
    figure_file2 = 'plots/agent1_ppo.png'
    figure_file3 = 'plots/agent2_ppo.png'
    best_score1= 0
    best_score2= 0
    score_history1 = []
    score_history2 = []
    learn_iters = 0
    avg_score = 0
    n_steps = 0
    qu_step={}
    resume = True
    if resume:
        agent.load_models()
    for i in range(n_games):
        env.reset()
        observation = env.board
        done = False
        score = 0
        steps=0
        while not done:
            if steps%2==0:
                action, prob, val = agent.choose_action(observation,'P1')
            else:
                action, prob, val = agent.choose_action(observation,'P2')

            env.step(action)

            observation_=env.board
            if steps%2==0:
                reward=env.rewards['P1']
            else:
                reward=env.rewards['P2']

            score += reward
            if steps%2==0:
                agent.remember(observation, action, prob, val, reward, done,'P1')
            else:
                agent.remember(observation, action, prob, val, reward, done,'P2')
            if n_steps % N == 0:
                agent.learn('P1')
                learn_iters += 1
            elif n_steps % N == 1:
                agent.learn('P2')
                learn_iters += 1
            observation = observation_
            done=any(env.terminations.values())
            if done:
                print(f'Player {env.agent_selection} won!')
                qu_step[i]=steps
            if steps%2==0:
                score_history1.append(score)
            else:
                score_history2.append(score)
            avg_score1 = np.mean(score_history1[-100:])
            avg_score2 = np.mean(score_history2[-100:])
            steps +=1
            n_steps += 1
        if i%100 ==0:
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score Player 1: %.1f' % avg_score1,'avg score Player 2: %.1f' % avg_score2,
                'episode_steps',steps,'time_steps', n_steps, 'learning_steps', learn_iters)
    x1 = [i+1 for i in range(len(score_history1))]
    x2 = [i+1 for i in range(len(score_history2))]
    plot_learning_curve2(qu_step.keys(), qu_step.values(), figure_file1)
    plot_learning_curve(x1, score_history1, figure_file2)
    plot_learning_curve(x2, score_history2, figure_file3)

