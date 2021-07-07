import random


class Bowling_Game:
    def __init__(self):
        self.reset()

    def roll_dice(self):
        self.dice = [random.randint(1, 6), random.randint(1, 6)]
        self.dice.append(self.dice[0] + self.dice[1])
        self.dice.sort()

    def get_pins(self):
        return self.pins

    def get_dice(self):
        return self.dice

    def get_state(self):
        return self.pins + self.dice

    def set_dice(self, dice):
        self.dice = dice

    def remove_pins(self, decision):
        if decision in [0, 3, 4, 6]:
            self.pins[self.dice[0] - 1] = 0
        if decision in [1, 3, 5, 6]:
            self.pins[self.dice[1] - 1] = 0
        if decision in [2, 4, 5, 6]:
            self.pins[self.dice[2] - 1] = 0

    def calculate_score(self, pins):
        score = pins.count(0) if pins.count(0) < 9 else 20
        score += 5 if pins.count(0) == 10 else 0
        return score

    def reset(self):
        self.pins = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.roll_dice()

    def is_done(self):
        return not sum([self.pins[d - 1] for d in self.dice if d <= 10])

    def get_legal_decisions(self, state):
        choices = []
        available_dice = [False]*3
        pins = state[:10]
        dice = state[10:]

        for i, die in enumerate(dice):
            if die <= 10 and pins[die - 1] == 1:
                choices.append(i)
                available_dice[i] = True

        if available_dice[0] and available_dice[1]:
            choices.append(3)
        if available_dice[0] and available_dice[2]:
            choices.append(4)
        if available_dice[1] and available_dice[2]:
            choices.append(5)
        if available_dice[0] and available_dice[1] and available_dice[2]:
            choices.append(6)
        return choices
