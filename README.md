# Double Q-network with proportional prioritization

A double Q-network with proportional prioritization is implemented in order to solve the *Project 1: Navigation* from <https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation>.

## Environment

Collecting a yellow banana (resp. a blue banana) gives a reward of 1 (resp. -1). The state space is a continuous 37 dimensional space. The action space is discrete $\mathcal A =\{0,1,2,3\}$ with

* 0 - move forward
* 1 - move backward
* 2 - turn left
* 3 - turn right.
  
The task is episodic and the environment is considered solved when the agent get an average score of 13 over 100 consecutive episodes.

## Installing the environment

* Download the environment:
  * Linux: **[download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)**
  * Mac OSX: **[download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)**
  * Windows (32-bit): **[download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)**
  * Windows (64-bit): **[download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)**
* Unzip
* In the code, import the UnityEnvironment as follow (the file_name should target the reader's own *Banana.exe*):

```python
from unityagents import UnityEnvironment
env = UnityEnvironment(file_name="C:\\Users\AL\Documents\GitHub\deep-reinforcement-learning\p1_navigation\Banana_Windows_x86_64\Banana.exe")
```

## Instructions

*ReplayBufferClass* contains two classes:

* the SumTree class from [AI-blog/SumTree.py](https://github.com/jaromiru/AI-blog/blob/master/SumTree.py)
* a modified version of the ReplayBuffer class from Udacity's DQN implementation [deep-reinforcement-learning/dqn/solution/dqn_agent.py](https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn).

Run the *DDQN_PER* python file in order to train the agent. After being trained over 600 episodes or if the problem is solved, the code will plot the scores and the average score over the last 100 episodes. It will save the neural network weights in *checkpoint.pth* and the scores in *scores.txt*. During the execution of the program, the code writes the current episode, the average score over the last 100 episodes, the current epsilon (used in the $\epsilon-$greedy policy) and the maximum score. The agent should be able to solve the environment in approximatively 400 episodes (hidden layer unit number = 32, plain fully connected NN with two hidden layers)

```dos
100/600 average score: 1.53     eps: 0.45       max_score: 9.00
200/600 average score: 7.73     eps: 0.20       max_score: 15.00
300/600 average score: 11.08    eps: 0.09       max_score: 20.00
400/600 average score: 12.98    eps: 0.05       max_score: 22.00
402/600 average score: 13.07    eps: 0.05       max_score: 22.00
Solved!
```
