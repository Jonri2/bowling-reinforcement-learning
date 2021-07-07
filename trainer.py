import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

PRINT_STEPS = 20


class QTrainer:
    def __init__(self, model, target, lr, gamma):
        self.model = model
        self.target = target
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()
        self.steps = 0

    # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    def train_step(self, batch, is_long=True):
        state_batch = torch.tensor(batch.state, dtype=torch.float)
        action_batch = torch.tensor(batch.action, dtype=torch.int64)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float)
        pred = self.model(state_batch)

        if is_long:
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.bool)
            non_final_next_states = torch.tensor([s for s in batch.next_state
                                                  if s is not None], dtype=torch.float)
            state_action_values = pred.gather(
                1, action_batch.unsqueeze(1)).transpose(0, 1)[0]
            next_state_values = torch.zeros(len(batch.next_state))
            next_state_values[non_final_mask] = self.target(
                non_final_next_states).max(1)[0].detach()
            self.steps += 1
            self.target.load_state_dict(self.model.state_dict())
        else:
            state_action_values = pred[action_batch]
            next_state_values = self.target(torch.tensor(
                batch.next_state, dtype=torch.float)).max().detach() if batch.next_state is not None else torch.tensor(0)

        Q_new = reward_batch + self.gamma * next_state_values
        loss = self.criterion(state_action_values, Q_new)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()

        if self.steps % PRINT_STEPS == 0 and is_long:
            print("Model:", pred[0])
            print("Target:", self.target(non_final_next_states)[0])
            print("State Action Values:", state_action_values[0].item())
            print("Next State Values:", next_state_values[0].item())
            print("Action:", action_batch[0].item())
            print("Reward:", reward_batch[0].item())
            print("Q:", Q_new[0].item())
            print("Loss:", loss.item())
