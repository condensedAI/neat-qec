"""
Runs evaluation functions in parallel subprocesses
in order to evaluate multiple genomes at once.
"""
from abstract_parallel_evaluator import AbstractParallelEvaluator
from resampling_algorithm import ResamplingAlgorithm

from game import ToricCodeGame
from simple_feed_forward import SimpleFeedForwardNetwork
from neat.nn import FeedForwardNetwork
from config import GameMode

# This is the object copied on each subprocess
# It contains the essential variables
class FitnessEvaluator(object):
    def __init__(self, config):
        self.error_rates = config["Training"]["error_rates"]
        self.error_mode = config["Training"]["error_mode"]
        self.reward_mode = config["Training"]["reward_mode"]
        self.n_games = config["Training"]["n_games"]

        self.game = ToricCodeGame(config)

    def __del__(self):
        self.game.close()

    def get(self, genome, config, puzzles_proportions):
        net = SimpleFeedForwardNetwork.create(genome, config)

        fitness = {error_rate: 0 for error_rate in self.error_rates}
        n_puzzles = {error_rate: int(self.n_games*puzzles_proportions[error_rate]) for error_rate in self.error_rates}
        fail_count = {error_rate: 0 for error_rate in self.error_rates}

        for error_rate in self.error_rates:
            for i in range(n_puzzles[error_rate]):
                result = self.game.play(net, error_rate, self.error_mode, self.reward_mode, GameMode["TRAINING"])["fitness"]
                fitness[error_rate] += result

                # Count the number of fails for the resampling learner
                fail_count[error_rate] += 1 - result

        return sum(fitness.values()) / len(self.error_rates) / self.n_games, fail_count


# The existence of this class lies on the necessity
# to share the fail_counts dictionary for the resampling algorithm
# because we want to average results over the population at a given generation

class ParallelEvaluatorResampling(AbstractParallelEvaluator):
    def __init__(self, num_workers, config, savedir, file_id, global_test_set=True, timeout=None):
        super().__init__(num_workers, config, savedir, file_id, global_test_set, timeout)

        # Resampling
        self.resampling = ResamplingAlgorithm(config["Training"]["error_rates"], config["Population"]["pop_size"], config["Training"]["n_games"])

        self.fitness_evaluator = FitnessEvaluator(config)

    def evaluate(self, genomes, config):
        self.resampling.reset()

        jobs = []
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.fitness_evaluator.get, (genome, config, self.resampling.puzzles_proportions)))

        # assign the fitness back to each genome
        # TODO: the best genome per generation is calculated twice (also in population)
        best = None # Best genome of the generation
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            genome.fitness, detailed_results = job.get(timeout=self.timeout)
            for error_rate in self.resampling.error_rates:
                self.resampling.fail_count[error_rate] += detailed_results[error_rate]

            if best is None or genome.fitness > best.fitness:
                best = genome

        # Update puzzle proportions
        self.resampling.update()

        if self.global_test_set:
            self.test(best, config)
