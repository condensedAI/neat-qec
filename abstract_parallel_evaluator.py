"""
Runs evaluation functions in parallel subprocesses
in order to evaluate multiple genomes at once.
"""
from multiprocessing import Pool
import copy
import pandas as pd

from test_set import TestSet

#
class AbstractParallelEvaluator(object):
    def __init__(self, num_workers, config, savedir, file_id, global_test_set=True, timeout=None):
        self.num_workers = num_workers
        self.timeout = timeout
        self.pool = Pool(num_workers)

        # Keeping track of the best genome evaluated on the exact same test set
        self.global_test_set = global_test_set
        if global_test_set:
            self.best_genome = None
            self.best_genome_test_score = 0
            self.test_set = TestSet(config, n_games=4000)
            self.recording = pd.DataFrame(columns=self.test_set.error_rates)
            self.gen_counter = 0
            self.savedir = savedir
            self.file_id = file_id

    def __del__(self):
        self.pool.close() # should this be terminate?
        self.pool.join()

    def test(self, generation_best, config):
        # Evaluate the best genome of the generation on the test set
        self.gen_counter += 1
        test_score, details = self.test_set.evaluate(self.pool, generation_best, config)
        self.recording.loc[self.gen_counter] = details
        self.recording.to_csv("%s/recording_best-%s.csv"%(self.savedir, self.file_id))
        print("Current generation (%i) best test score: %0.2f"%(self.gen_counter, test_score))
        if test_score > self.best_genome_test_score:
            # Make sure to do a deep copy
            self.best_genome = copy.copy(generation_best)
            self.best_genome_test_score = test_score
            print("NEW BEST with %0.2f"%test_score)
