import neat
from datetime import datetime
import pickle
import os
import json
import visualize

# Project imports
from parallel_evaluator import ParallelEvaluator
from parallel_evaluator_resampling import ParallelEvaluatorResampling
from transplantation import transplantate_population
from initialize import initialize_population
from genome_checkpointer import GenomeCheckpointer
from config import generate_config_file, TrainingMode

class Population():
    def __init__(self, config):
        self.config = config
        self.d = config["Physics"]["distance"]
        self.training_mode = config["Training"]["training_mode"]
        self.n_generations = config["Training"]["n_generations"]

    def evolve(self, savedir, n_cores=1, loading_file=None, initialize_file=None, transplantation_file=None, verbose=0):
        time_id = datetime.now()

        if loading_file is None:
            # Generate configuration file
            # TODO: No need to generate population config file each time
            generate_config_file(savedir, self.config)

            population_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 savedir+"/population-config")

            # Create the population, which is the top-level object for a NEAT run.
            p = neat.Population(population_config)

            # Transplantation of genomes in the initial population
            if not transplantation_file is None:
                transplantate_population(p=p,
                              transplantation_file=transplantation_file,
                              config_rec=population_config.genome_config,
                              size_rec=self.config["Physics"]["distance"],
                              neat_config=self.config)

            elif not initialize_file is None:
                initialize_population(p=p,
                                initialize_file=initialize_file)

        else:
            p = neat.Checkpointer.restore_checkpoint(loading_file)

        if verbose:
            # Add a stdout reporter to show progress in the terminal.
            p.add_reporter(neat.StdOutReporter(True))

        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        file_id = time_id.strftime('%Y-%m-%d_%H-%M-%S')
        p.add_reporter(neat.Checkpointer(generation_interval=100,
                                         time_interval_seconds=None,
                                         filename_prefix="%s/checkpoint-%s-"%(savedir, file_id)))
        p.add_reporter(GenomeCheckpointer(generation_interval=100,
                                         filename_prefix="%s/checkpoint-best-genome-%s-"%(savedir, file_id)))
        # TODO: checkpointer is cleaner for reporting best genome performance
        #p.add_reporter(Test)


        if self.training_mode == TrainingMode["RESAMPLING"]:
            pe = ParallelEvaluatorResampling(num_workers=n_cores,
                                             global_test_set=True,
                                             config=self.config,
                                             savedir=savedir,
                                             file_id=file_id)
        else:
            pe = ParallelEvaluator(num_workers=n_cores,
                                   global_test_set=True,
                                   config=self.config,
                                   savedir=savedir,
                                   file_id=file_id)

        w = p.run(pe.evaluate, self.n_generations)
        #print("Check best test scores: %.2f vs %.2f"%(pe.test_set.evaluate(w, population_config), pe.best_genome_test_score))
        winner = pe.best_genome

        # Display the winning genome.
        print('\nBest genome on global test set:\n{!s}'.format(winner))

        if verbose > 1:
            # Show output of the most fit genome against training data.
            visualize.plot_stats(stats, ylog=False, view=True)

        # Saving and closing
        stats.save_genome_fitness(filename="%s/genome.fitness.%s.csv"%(savedir, time_id.strftime('%Y-%m-%d_%H-%M-%S')))
        stats.save_species_count(filename="%s/species.count.%s.csv"%(savedir, time_id.strftime('%Y-%m-%d_%H-%M-%S')))
        stats.save_species_fitness(filename="%s/species.fitness.%s.csv"%(savedir, time_id.strftime('%Y-%m-%d_%H-%M-%S')))

        # Save the winner
        with open("%s/winner.genome.%s.pkl"%(savedir, time_id.strftime('%Y-%m-%d_%H-%M-%S')), 'wb') as f:
            pickle.dump(winner, f)


        return winner.fitness
