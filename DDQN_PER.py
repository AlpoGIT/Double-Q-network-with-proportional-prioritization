from ReplayBufferClass import ReplayBuffer
from unityagents import UnityEnvironment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import random

# Unity environment
# Change My_Path: e.g. C:\\Users\AL\Documents\GitHub\Example\
env = UnityEnvironment(file_name="My_Path\Banana.exe")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
action_size = brain.vector_action_space_size


# default device for torch (default is gpu)
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")


# Neural network class: state -> action values
class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed,\
       fc1_units=128, fc2_units=128):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define Q, Qhat and set Qhat = Q
Q = QNetwork(37,4,0).to(device)
Qhat = QNetwork(37,4,0).to(device)
Qhat.load_state_dict(Q.state_dict())


#initialize replay memory to capacity N
N = 6400*2
batch_size = 64
memory = ReplayBuffer(4, N, batch_size, 0)


#tzero = torch.zeros(batch_size,1).detach().to(device)


# Definition of the optimizer for gradient descent
optim = torch.optim.Adam(Q.parameters(), lr = 0.001)


#------hyperparameters------#

gamma = 0.99#r + gamma*Qhat(s',argmax_a' Q(s',a'))
M = 600     #number of episodes
C = 4       #from time to time, Qhat = Q
T = 1000    #max duration of 1 episode
K = 4       #replay period

SMALL = 0.0001  #P(i)~(|TD_error|+SMALL)^\alpha

# epsilon-greedy parameters
eps = 1.
eps_0 = 1.      #initial epsilon
eps_f = 0.1     #final epsilon
tau = 0.01      #soft update of weights, should prevent oscillations

# In the orignal paper, alpha~0.7 and beta_i~0.5
alpha = 0.8     #P(i)~(|TD_error|+SMALL)^\alpha
beta_i = 0.7    #w_i =(1/(N*P(i)))^\beta
beta_f = 1.

# clipping [-1,1] is used in a 'custom loss function'
p_max = np.array([1.+SMALL]) #initial priority with max value in [-1,1]
w = deque(maxlen = N)        #Importance sampling weights

# Scores
score_window = deque(maxlen = 100)
scores = []
max_score = 0
score = 0

#
batch_size_array = np.arange(batch_size)

# the agent internal time
internal_counter = 0

def choose_action(state, eps):
    #epsilon-greedy action: with probability eps; choose a random action,
    #                       otherwise, choose an optimal action.
    #input: state
    #       epsilon
    #ouput: action

        rd = np.random.rand()
        if rd < eps:
            action = np.random.choice(np.arange(action_size))
        else:
            tstate = torch.tensor(state, dtype = torch.float).to(device)
            action = np.argmax(Q(tstate).detach().cpu().numpy())
        return  int(action)


def PER_loss(input, target, weights):
    #Custom loss: prioritized replay introduces a bias,
    #             corrected with the importance-sampling weights.
    #input: input -- Q
    #       target -- r + gamma*Qhat(s', argmax_a' Q(s',a'))
    #       weights -- importance sampling weights
    #output:loss -- unbiased loss

    loss = torch.clamp((input-target),-1,1)
    loss = loss**2
    loss = torch.sum(weights*loss)
    return loss


#------Main loop------#
# The agent trains for M episodes

for episode in range(1,M+1):

    # initialize the score
    max_score = np.maximum(score, max_score)
    score = 0       


    # epsilon behavior
    # after the training, agent is almost greedy
    #eps = (eps_f-eps_0)*(episode/M) + eps_0 #linearly decreasing
    #eps = eps_0*np.exp((episode/M)*np.log(eps_f/eps_0)) #exp decreasing
    eps = np.maximum(0.05, eps*0.992) #power law


    # linearly increasing from beta_i~0.5 to beta_f = 1
    beta = (beta_f-beta_i)*(episode-1)/(M-1) + beta_i
    

    # reset the environment and get the current state
    env_info = env.reset(train_mode=True)[brain_name] 
    state = env_info.vector_observations[0]



    #------One episode------#
    #one episode lasts 1000 steps or until the environment resets

    for t in range(1,T+1):

        #choose epsilon-greedy action from the current state
        action = choose_action(state, eps)


        #the environment's feedback
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        

        #store transition in D with max priority
        memory.add(state, action, reward, next_state, done, p_max)
        

        #accumulate experience before learning and learn periodically
        if (internal_counter > 2*batch_size   ) & (internal_counter % K == 0):

            #sample transitions from memory
            states, actions, rewards, next_states, dones, priorities, indices = memory.sample()


            #compute importance sampling weight, before the update
            w.append(priorities)
            p_max = np.max(w)
            np_w = (np.sum(w)/(len(w)*priorities)).reshape(-1,1)**beta
            np_w /= p_max

            # decoupled actions a_ for double DQN, should be argmax from Q
            # compute TD error for loss function
            a_ = Q(next_states).argmax(dim = 1)
            y = (rewards + gamma*\
                Qhat(next_states).detach()[batch_size_array,a_].unsqueeze(1).to(device)\
                *(1-dones))
            local = Q(states).gather(1,actions)


            #for priorities
            y_p = (rewards + gamma*\
                Qhat(next_states).detach().max(1)[0].unsqueeze(1)\
                *(1-dones))
            

            #update priorities p_j
            with torch.no_grad():
                tw = torch.tensor(np_w).detach().float().to(device)
                p = torch.abs(y_p-local)
                p = (p.cpu().numpy()+ SMALL)**alpha
                for j, idx in enumerate(indices):
                    memory.tree_memory.update(idx, p[j])

            #perform gradient descent, annealing the biais from sampling ~ P(i)
            #also clip errors [-1,1] in PER_loss

            loss = PER_loss(local, y, tw)
            optim.zero_grad()
            loss.backward()
            optim.step()

        #after the gradient descent, from time to time, update Qhat = Q
            if internal_counter % C == 0:
                for target_param, local_param in zip(Qhat.parameters(), Q.parameters()):
                        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

            
        state = next_state
        internal_counter += 1
        score += reward

        if done:
            break


    # Print scores, save weights
    scores.append(score)
    score_window.append(score)
    print("\r{}/{}\taverage score: {:.2f}\teps: {:.2f}\tmax_score: {:.2f}"\
        .format(episode,M,np.mean(score_window), eps, max_score), end = "")
    if episode%100 == 0:
        print("\r{}/{}\taverage score: {:.2f}\teps: {:.2f}\tmax_score: {:.2f}"\
            .format(episode,M,np.mean(score_window), eps, max_score))
        torch.save(Q.state_dict(), 'checkpoint.pth')
        np.savetxt('scores.txt', scores, fmt='%d')
    if np.mean(score_window) > 13:
        print("\nSolved!")
        torch.save(Q.state_dict(), 'checkpoint.pth')
        np.savetxt('scores.txt', scores, fmt='%d')
        break
env.close()


# plot and save scores
score_window = deque(maxlen = 100)
average = []
#scores = np.loadtxt('scores.txt')
goal = np.array([13 for i in scores])
for x in scores:
    score_window.append(x)
    average.append(np.mean(score_window))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores, label = 'Score')
plt.plot(average, 'r-', label = 'Average over the last 100 episodes',linewidth = 2)
plt.plot(goal, 'k--',label = 'Goal', linewidth = 2)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.legend()

plt.savefig('scores_plot.png', bbox_inches='tight')
plt.show()
