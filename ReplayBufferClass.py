import numpy as np
from collections import deque, namedtuple
import random
import torch

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])



class ReplayBuffer:
    """Fixed-size buffer to store experience tuples"""

    def __init__(self, action_size, buffer_size, batch_size, seed):

        self.action_size = action_size  
        self.batch_size = batch_size
        #store in a sum tree
        self.tree_memory = SumTree(buffer_size)

    
    def add(self, state, action, reward, next_state, done, TD_error):
        """Add a new experience to memory."""
        e = tuple((state, action, reward, next_state, done, TD_error))


        #priority = TD_error #priority of the transition
        self.tree_memory.add(TD_error, e)



    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = []
        segment = self.tree_memory.total()/self.batch_size #total error
        

        for i in range(self.batch_size):
            a = segment*i
            b = segment*(i+1)
            rd = np.round(np.random.uniform(a,b),6) #should fix some problems with idx
            idx, priority, data = self.tree_memory.get(rd)
           
            experiences.append( data + (priority,) + (idx,) )

        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float().to(device)
        priorities = np.array([e[6] for e in experiences])
        indices =np.array([e[7] for e in experiences])
  
        return (states, actions, rewards, next_states, dones, priorities, indices)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
