import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from game import Bowling_Game
from agent import Agent, Transition

DEBUG = False
SAVE_FREQUENCY = 1000
PRINT_FREQUENCY = 1000
NUM_FRAMES_IN_GAME = 10
EPSILON = 5000
LOAD_FILE_NAME = None  # '48000_96_99#.pth'


class Plotter:
    def __init__(self):
        self.plot_scores = []
        self.plot_mean_scores = []
        self.agent = Agent(LOAD_FILE_NAME, epsilon_start=EPSILON)
        self.game = Bowling_Game()
        self.game_score = 0
        self.num_frames = 0
        self.rolls = 0
        self.score = 0
        self.decision_count = np.array([0]*7)
        self.frame_transitions = []
        self.legal_count = 0
        plt.ion()

    def plot(self):
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(self.plot_scores)
        plt.plot(self.plot_mean_scores)
        plt.ylim(ymin=0)
        plt.text(len(self.plot_scores)-1,
                 self.plot_scores[-1], str(self.plot_scores[-1]))
        plt.text(len(self.plot_mean_scores)-1,
                 self.plot_mean_scores[-1], str(self.plot_mean_scores[-1]))
        plt.show(block=False)
        plt.pause(.1)

    def save(self, mean_score, legal_percent):
        num_games_thousands = (
            self.agent.n_games - 1) // SAVE_FREQUENCY + 1
        new_file_name = "{games}_{score}_{legal}.pth".format(
            games=num_games_thousands * SAVE_FREQUENCY, score=int(mean_score), legal=int(legal_percent))
        self.decision_count = np.array([0]*7)
        self.rolls = 0
        self.legal_count = 0

        self.agent.model.save(new_file_name)

    def finish_frame(self, old_state, new_state):
        if DEBUG:
            print("State:", new_state)
            print("Score:", self.score)
        last_transition = self.frame_transitions[-1]
        self.frame_transitions[-1] = Transition(
            last_transition.state, last_transition.action, last_transition.reward, next_state=None)
        for transition in self.frame_transitions:
            self.agent.remember(transition)
        self.game_score += self.score
        self.game.reset()
        self.num_frames += 1
        self.score = 0
        self.frame_transitions = []

        if self.num_frames % NUM_FRAMES_IN_GAME == 0:
            self.finish_game()

    def finish_game(self):
        self.agent.train_long_memory()
        self.agent.n_games += 1
        self.plot_scores.append(self.game_score)
        mean_score = sum(
            self.plot_scores[-SAVE_FREQUENCY:]) / SAVE_FREQUENCY if self.agent.n_games > SAVE_FREQUENCY else sum(self.plot_scores) / self.agent.n_games
        self.plot_mean_scores.append(mean_score)
        self.plot()
        self.game_score = 0
        legal_percent = (self.legal_count/self.rolls)*100

        if self.agent.n_games % SAVE_FREQUENCY == 0:
            self.save(mean_score, legal_percent)

    def get_reward(self, old_state, new_state):
        old_pins = old_state[:10]
        new_pins = new_state[:10]
        return self.game.calculate_score(
            new_pins) - self.game.calculate_score(old_pins)

    def train(self):
        old_state = self.game.get_state()
        final_decision = self.agent.make_decision(
            old_state, self.game.get_legal_decisions(old_state))
        legal_decisions = self.game.get_legal_decisions(old_state)

        if final_decision in legal_decisions:
            self.game.remove_pins(final_decision)
            self.game.roll_dice()
            new_state = self.game.get_state()
            self.legal_count += 1
            reward = self.get_reward(old_state, new_state)
            self.score += reward
            self.decision_count[final_decision] += 1
        else:
            if DEBUG:
                print("Illegal Move!")
            reward = 0
            new_state = None
            self.game.roll_dice()

        transition = Transition(
            old_state, final_decision, reward, new_state)
        self.frame_transitions.append(transition)
        # self.agent.train_short_memory(transition)
        self.rolls += 1

        if DEBUG:
            print("State:", old_state)
            print("Legal Decisions:", legal_decisions)
            print("Final Decision:", final_decision)
            print("Reward:", reward)

        if self.game.is_done():
            self.finish_frame(old_state, new_state)

        if self.rolls % PRINT_FREQUENCY == 0 and self.rolls != 0:
            print("Decision Distribution:", self.decision_count/self.rolls)
            print("Legal Moves:", self.legal_count/self.rolls)


p = Plotter()
while True:
    p.train()
