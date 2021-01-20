import pickle
from datetime import datetime
import argparse, json
import os
import pandas as pd
import numpy as np
from multiprocessing import Pool
import multiprocessing

import neat

from config import GameMode, RewardMode, check_config
from neat.nn import FeedForwardNetwork
from game import ToricCodeGame
from simple_feed_forward import SimpleFeedForwardNetwork
from config import generate_config_file
from transplantation import transplantate



def evaluate(file, error_rates, n_games, n_jobs,
            verbose, file_suffix='', transfer_to_distance=None,
            without_illegal_actions=False):
    time_id = datetime.now()

    print(without_illegal_actions)
    # Load the corresponding config files
    savedir = file[:file.rfind("/")]

    if not os.path.exists("%s/config.json"%savedir):
        raise ValueError("Configuration file does not exist (%s)."%("%s/config.json"%savedir))

    with open("%s/config.json"%savedir) as f:
        config = json.load(f)

    config = check_config(config)

    # Load the genome to be evaluated
    if not os.path.exists(file):
        raise ValueError("Genome file does not exist.")

    with open(file, "rb") as f:
        genome = pickle.load(f)

    if not os.path.exists("%s/population-config"%savedir):
        raise ValueError("Population configuration file does not exist.")

    population_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "%s/population-config"%savedir)


    if transfer_to_distance is None:
        net = SimpleFeedForwardNetwork.create(genome, population_config)
        code_distance = config["Physics"]["distance"]
    elif transfer_to_distance > config["Physics"]["distance"]:

        generate_config_file(savedir, config, transfer_to_distance)

        pop_config_transferred = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             savedir+"/population-config-temp-d"+str(transfer_to_distance))

        new_genome = pop_config_transferred.genome_type(0)
        new_genome.configure_new(pop_config_transferred.genome_config)
        new_genome.connections = {}
        new_genome.nodes = {}

        transplantate(pop_config_transferred.genome_config, new_genome, transfer_to_distance, genome, config["Physics"]["distance"], config)
        net = SimpleFeedForwardNetwork.create(new_genome, pop_config_transferred)
        code_distance = transfer_to_distance

    # DIRTY: To ensure that samples are generated according to transfer_to_distance
    config["Physics"]["distance"] = code_distance

    ## (PARALLEL) EVALUATION LOOP
    fitness = []
    results={"fitness":[], "error_rate":[], "outcome":[], "nsteps":[], "initial_qubits_flips":[]}

    # with statement to close properly the parallel processes
    with Pool(n_jobs) as pool:
        # Game evaluation
        for error_rate in error_rates:
            fitness.append(0)

            jobs=[]
            for i in range(n_games):
                jobs.append(pool.apply_async(get_fitness, (net, config, error_rate, without_illegal_actions)))

            for job in jobs:
                output, errors_id = job.get(timeout=None)

                fitness[-1] += output["fitness"]
                for k, v in output.items():
                    results[k].append(v)
                results["initial_qubits_flips"].append(errors_id)

            fitness[-1] /= n_games
            print("Evaluation on error_rate=%.2f is done, %.2f success."%(error_rate, fitness[-1]))

        elapsed = datetime.now() - time_id
        print("Total running time:", elapsed.seconds,":",elapsed.microseconds)

        # Always overwrite the result of evaluation
        # Synthesis report
        if transfer_to_distance is not None:
            file_suffix+=".transfered_distance%i"%transfer_to_distance
        if without_illegal_actions:
            file_suffix+=".no_illegal_action"

        savefile = "%s_evaluation.ngames=%i.%s.csv"%(file.replace(".pkl", ""), n_games, file_suffix)
        if os.path.exists(savefile):
            print("Deleting evaluation file %s"%savefile)
            os.remove(savefile)

        print([error_rates, fitness])
        df = pd.DataFrame(list(zip(error_rates, fitness)), columns=["error_rate", "mean_fitness"])
        df.to_csv(savefile)

        # Detailed report
        savefile = "%s_detailed_results_evaluation.ngames=%i.%s.csv"%(file.replace(".pkl", ""), n_games, file_suffix)
        if os.path.exists(savefile):
            print("Deleting evaluation file %s"%savefile)
            os.remove(savefile)

        pd.DataFrame.from_dict(results).to_csv(savefile)

    return error_rates, fitness

def get_fitness(net, config, error_rate, without_illegal_actions=False, seed=None):
    # We need to create a different game object for each thread
    game = ToricCodeGame(config)
    res = game.play(net, error_rate, RewardMode["BINARY"],
                    GameMode["EVALUATION"], seed, without_illegal_actions)
    initial_errors = ['1' if i in game.env.initial_qubits_flips else '0' for i in game.env.state.qubit_pos]
    error_id = int(''.join(initial_errors),2)
    return res, error_id

if __name__ == "__main__":
    # Parse arguments passed to the program (or set defaults)
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", nargs="+", help="Genome file to load and evaluate")
    parser.add_argument("--errorRates", type=float, nargs="+", default=np.arange(0.01, 0.21, 0.01), help="Qubit error rate")
    #parser.add_argument("--errorMode", type=int, choices=[0,1], default=0, help="Error generation mode")
    parser.add_argument("-n", "--numPuzzles", type=int, default=1000, help="Number of syndrome configurations to solve per individual")
    #parser.add_argument("--maxSteps", type=int, default=1000, help="Number of maximum qubits flips to solve syndromes")
    parser.add_argument("--withoutIllegalActions", default=False, action="store_true", help="Without illegal actions")
    parser.add_argument("-j", "--numParallelJobs", type=int, default=1, help="Number of jobs launched in parallel")
    parser.add_argument("--id", default="", help="File additional id")
    parser.add_argument("--transferToDistance", type=int, choices=[3,5,7,9,11], help="Hyperneat: Toric code distance to evaluate on")
    parser.add_argument("-v", "--verbose", type=int, choices=[0,1,2], default=0, help="Level of verbose output (higher is more)")
    args = parser.parse_args()

    for file in args.file:
        evaluate(file, args.errorRates,  args.numPuzzles, args.numParallelJobs, args.verbose, args.id, args.transferToDistance, args.withoutIllegalActions)
