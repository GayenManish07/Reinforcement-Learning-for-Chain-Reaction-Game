import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class OUActionNoise(object):
    def __init__(self,mu, sigma=0.15, theta=0.2 ,dt=1e-2,x0=None):
        self.mu=mu
        self.sigma=sigma
        self.theta=theta
        self.dt=dt
        self.x0=x0
        self.reset()
    def __call__(self):
        x=self.x_prev +self.theta*(self.mu-self.x0)+ self.dt \
        +self.sigma*np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
        self.x_prev=x
        return x
    def reset(self):
        self.x_prev=self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    

class ReplayBuffer(object):
    def __init__(self,input_shape,max_size,n_actions):
        self.mem_size=max_size
        self.mem_cntr=0
        self.state_mem=np.zeros((self.mem_size,*input_shape))
        self.new_state_mem=np.zeros((self.mem_size,*input_shape))
        self.action_mem=np.zeros((self.mem_size,n_actions))
        self.reward_mem=np.zeros(self.mem_size)
        self.terminal_mem=np.zeros(self.mem_size,dtype=np.float32)

    def store_transition(self,state,action,reward,state_,done):
        index= self.mem_cntr%self.mem_size
        self.state_mem[index]=state
        self.action_mem[index]=action
        self.reward_mem[index]=reward
        self.new_state_mem[index]=state_
        self.terminal_mem=1-done
        self.mem_cntr+=1
        
    
    def sample_buffer(self,batch_size):
        max_mem=min(self.mem_cntr,self.mem_size)
        batch=np.random.choice(max_mem,batch_size)

        state=self.state_mem[batch]
        actions=self.action_mem[batch]
        reward=self.reward_mem[batch]
        new_states=self.new_state_mem[batch]
        terminal_state=self.terminal_mem[batch]
        return state ,actions, reward, new_states,terminal_state
    
class CriticNetwork(nn.Module):

    def __init__(self,beta,n_actions,input_dims,fc1_dims,fc2_dims,ckpt='checkpoints/ddpg/'):

        super(CriticNetwork,self).__init__()
        self.input_dims=input_dims
        self.fc1_dims=fc1_dims
        self.fc2_dims=fc2_dims
        self.actions=n_actions
        self.checkpoin_file=os.path.join(ckpt+'ddpg.pth')
        self.fc1=nn.linear(*self.input_dims,self.fc1_dims)
        f1=1./np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data,-f1,f1)
        nn.init.uniform_(self.fc1.bias.data,-f1,f1)
        self.bn1=nn.LayerNorm(self.fc1_dims)
        self.fc2-nn.Linear(self.fc1_dims,self.fc2_dims)
        f2=1./np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data,-f2,f2)
        nn.init.uniform_(self.fc2.bias.data,-f2,f2)
        self.bn2=nn.LayerNorm(self.fc2_dims)
        f3=0.003
        self.action_value=nn.Linear(self.actions,self.fc2_dims)
        self.q=nn.Linear(fc2_dims,1)
        nn.init.uniform(self.fc3.weight.data,-f3,f3)
        nn.init.uniform(self.fc3.bias.data,-f3,f3)
        self.optimizer=optim.AdamW(self.parameters(),lr=beta)
        self.device=torch.device('cuda' if torch.cuda.is_avaiable() else 'cpu')
        self.to(self.device)

    def forward(self,state,action):
        state_value=self.fc1(state)
        state_value=self.bn1(state_value)
        state_value=F.relu(state_value)
        state_value=self.fc2(state_value)
        state_value=self.bn2(state_value)
        action_value=F.relu(self.action_value(action))
        state_action_value=F.relu(torch.add(state_value,action_value))
        state_action_value=self.q(state_action_value)
        return state_action_value


    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self,alpha,n_actions,input_dims,fc1_dims,fc2_dims,ckpt='checkpoints/ddpg/'):

        super(CriticNetwork,self).__init__()
        self.input_dims=input_dims
        self.fc1_dims=fc1_dims
        self.fc2_dims=fc2_dims
        self.actions=n_actions
        self.checkpoin_file=os.path.join(ckpt+'ddpg.pth')
        self.fc1=nn.linear(*self.input_dims,self.fc1_dims)
        f1=1./np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data,-f1,f1)
        nn.init.uniform_(self.fc1.bias.data,-f1,f1)
        self.bn1=nn.LayerNorm(self.fc1_dims)
        self.fc2-nn.Linear(self.fc1_dims,self.fc2_dims)
        f2=1./np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data,-f2,f2)
        nn.init.uniform_(self.fc2.bias.data,-f2,f2)
        self.bn2=nn.LayerNorm(self.fc2_dims)
        f3=0.003
        self.mu=nn.Linear(self.fc2,self.actions)
        nn.init.uniform_(self.mu.weight.data,-f3,f3)
        nn.init.uniform_(self.mu.bias.data,-f3,f3)
        self.optimizer=optim.AdamW(self.parameters(),lr=alpha)
        self.device=torch.device('cuda' if torch.cuda.is_avaiable() else 'cpu')
        self.to(self.device)
    
    def forward(self,state):
        x=self.fc1(state)
        x=self.bn1(x)
        x=F.relu(x)
        x=self.fc2(x)
        x=self.bn1(x)
        x=F.relu(x)
        x=self.mu(x)
        x=torch.tanh(x)
        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class Agent(object):
    def __init__(self,alpha,beta,input_dims,tau,env,n_actions,gamma=0.99,max_size=1000000, layer1_size=400,layer2_size=300, batch_size=64):
        self.tau=tau
        self.gamma=gamma
        self.memory=ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size=batch_size
        self.actor=ActorNetwork(alpha, input_dims, layer1_size,layer2_size, n_actions=n_actions,name='Actor')
        self.critic=CriticNetwork(beta, input_dims, layer1_size,layer2_size, n_actions=n_actions,name='Critic')
        self.target_actor=ActorNetwork(alpha, input_dims, layer1_size,layer2_size, n_actions=n_actions,name='TargetActor')
        self.target_critic=CriticNetwork(beta, input_dims, layer1_size,layer2_size, n_actions=n_actions,name='TargetCritic')
        self.noise = OUActionNoise(mu=np.zeros(n_actions))
        self.update_network_parameters(tau=1)
    def choose_action(self,observation):
        self.actor.eval()
        observation=torch.tensor(observation, dtype=torch.float).to(self.actor.device)
        mu=self.actor.forward(observation).to(self.actor.device)
        mu_prime=mu+ torch.tensor(self.noise,dtype=float).to(self.actor.device)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr<self.batch_size:
            return
        state, action ,reward ,new_state,done=self.memory.sample_buffer(self.batch_size)
        reward = torch.tensor(reward, dtype=torch.float).to(self.critic.device)
        done = torch.tensor(done).to(self.critic.device)
        new_state = torch.tensor(new_state, dtype=torch.float).to(self.critic.device)
        action = torch.tensor(action, dtype=torch.float).to(self.critic.device)
        state = torch.tensor(state, dtype=torch.float).to(self.critic.device)
        
        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()
        target_actions = self.target_actor.forward(new_state)
        critic_value_ = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j])
        target = torch.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self,tau=None):
        if tau is None:
            tau=self.tau
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                      (1-tau)*target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                      (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)
        
        

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def check_actor_params(self):
        current_actor_params = self.actor.named_parameters()
        current_actor_dict = dict(current_actor_params)
        original_actor_dict = dict(self.original_actor.named_parameters())
        original_critic_dict = dict(self.original_critic.named_parameters())
        current_critic_params = self.critic.named_parameters()
        current_critic_dict = dict(current_critic_params)
        print('Checking Actor parameters')

        for param in current_actor_dict:
            print(param, torch.equal(original_actor_dict[param], current_actor_dict[param]))
        print('Checking critic parameters')
        for param in current_critic_dict:
            print(param, torch.equal(original_critic_dict[param], current_critic_dict[param]))
        input()
       






       


    

    


        