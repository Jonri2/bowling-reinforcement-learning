import os
import random
import torch
import numpy as np
from collections import deque, namedtuple
from model import Linear_QNet
from trainer import QTrainer

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

E_FACTOR = 1.5

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


# https://www.youtube.com/watch?v=6pJBPPrDO40
class Agent:

    def __init__(self, filename=None, filter_pred=False, epsilon_start=0):
        self.n_games = 0
        self.epsilon = 0
        self.epsilon_start = epsilon_start
        self.gamma = 0.999
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(13, 256, 7)
        self.target = Linear_QNet(13, 256, 7)
        self.target.eval()
        self.target.load_state_dict(self.model.state_dict())
        if filename is not None:
            self.model.load_state_dict(torch.load(
                os.path.join('./model', filename)))
        self.trainer = QTrainer(self.model, self.target,
                                lr=LEARNING_RATE, gamma=self.gamma)
        self.filter = filter_pred

    def make_decision(self, state, legal_decisions):
        if state.count(1) == 0:
            return -1
        self.epsilon = self.epsilon_start - self.n_games
        if random.randint(0, int(self.epsilon_start*E_FACTOR)) < self.epsilon:
            decision = random.randint(0, 6)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            if self.filter:
                masked_prediction = torch.tensor(
                    [-1000000]*7, dtype=torch.float)
                for i, p in enumerate(prediction):
                    if i in legal_decisions:
                        masked_prediction[i] = p
                decision = torch.argmax(masked_prediction).item()
            else:
                decision = torch.argmax(prediction).item()
        return decision

    def train_short_memory(self, transition):
        self.trainer.train_step(transition, False)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        batch = Transition(*zip(*mini_sample))

        self.trainer.train_step(batch)

    def remember(self, transition):
        self.memory.append(transition)
