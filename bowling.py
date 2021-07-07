from agent import Agent
from game import Bowling_Game


class Bowling_Competition:
    def __init__(self, player1, player2, num_games):
        self.players = [player1, player2]
        self.scores = [[], []]
        self.game = Bowling_Game()
        self.trials = 0
        self.num_games = num_games

    def compete(self):
        dice = []
        for i in range(self.num_games * 2):
            player_id = self.trials % 2
            rolls = 0
            if player_id == 1:
                self.game.set_dice(dice[rolls])

            while not self.game.is_done():
                state = self.game.get_state()
                legal_decisions = self.game.get_legal_decisions(state)
                decision = self.players[player_id].make_decision(
                    state, legal_decisions)
                self.game.remove_pins(decision)

                if player_id == 0:
                    dice.append(state[10:])
                    self.game.roll_dice()
                else:
                    rolls += 1
                    if rolls < len(dice):
                        self.game.set_dice(dice[rolls])
                    else:
                        self.game.roll_dice()

            state = self.game.get_state()
            if player_id == 0:
                dice.append(state[10:])
            else:
                dice = []

            self.scores[self.trials %
                        2].append(self.game.calculate_score(state[:10]))
            self.trials += 1
            self.game.reset()
        print(sum(self.scores[0]*10) / (self.trials / 2),
              sum(self.scores[1]*10) / (self.trials / 2))


class Jon_Agent:
    def __init__(self):
        pass

    def make_decision(self, state, legal_decisions):
        pins = state[:10]
        dice = state[10:]

        if 6 in legal_decisions and pins.count(0) >= 6:
            return 6

        for d in [3, 4, 5]:
            if d in legal_decisions and pins.count(0) >= 7:
                return d

        if (2 in legal_decisions and 10 >= dice[2] >= 7) or (not 0 in legal_decisions and not 1 in legal_decisions):
            return 2
        elif 0 in legal_decisions:
            return 0
        else:
            return 1


comp = Bowling_Competition(Agent("48000_96_99#.pth", True),
                           Jon_Agent(), 10000)
comp.compete()
