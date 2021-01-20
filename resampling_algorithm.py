

class ResamplingAlgorithm():
    def __init__(self, error_rates, pop_size, n_games):
        self.error_rates = error_rates
        self.pop_size = pop_size
        self.n_games = n_games

        # Moving proportions of puzzles
        self.puzzles_proportions = {error_rate: 1/len(error_rates) for error_rate in error_rates}
        # Moving average success on puzzles
        self.fail_count = {error_rate: 0 for error_rate in error_rates}

    def reset(self):
        # Reset the puzzles failure counting
        # Very important to modify the same object in memory via a for loop
        for error_rate in self.error_rates:
            self.fail_count[error_rate] = 0

    def update(self):
        # Update the puzzle proportions
        print("Fail count", self.fail_count)

        average_fails = {error_rate: 0 for error_rate in self.error_rates}
        for error_rate in self.error_rates:
            # Average success over the whole generation
            average_fails[error_rate] = self.fail_count[error_rate]/self.pop_size/(self.n_games*self.puzzles_proportions[error_rate])

        total_fail = sum(average_fails.values())

        for error_rate in self.error_rates:
            if average_fails[error_rate] / total_fail * self.n_games < 1:
                self.puzzles_proportions[error_rate] = 1 / self.n_games
            else:
                self.puzzles_proportions[error_rate] = average_fails[error_rate] / total_fail

        print("New Puzzle proportion: {}".format(self.puzzles_proportions))
