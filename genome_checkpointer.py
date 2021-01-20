import pickle

from neat.reporting import BaseReporter

class GenomeCheckpointer(BaseReporter):
    def __init__(self, generation_interval=100, filename_prefix='neat-checkpoint-'):

        self.generation_interval = generation_interval
        self.filename_prefix = filename_prefix

        self.current_generation = None
        self.last_generation_checkpoint = -1

    def start_generation(self, generation):
        self.current_generation = generation

    def post_evaluate(self, config, population, species_set, best_genome):
        checkpoint_due = False

        if (self.generation_interval is not None):
            dg = self.current_generation - self.last_generation_checkpoint
            if dg >= self.generation_interval:
                checkpoint_due = True

        if checkpoint_due:
            self.save_checkpoint(best_genome, self.current_generation)
            self.last_generation_checkpoint = self.current_generation

    def save_checkpoint(self, best_genome, generation):
        """ Save the current simulation best genome. """
        filename = '{0}{1}'.format(self.filename_prefix, generation)
        print("Saving best genome to {0}".format(filename))

        print(best_genome)

        with open(filename, 'wb') as f:
            pickle.dump(best_genome, f)
