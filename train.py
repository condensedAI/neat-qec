from datetime import datetime
import argparse, json
from glob import glob
import os
from numpy import argmax

# Project imports
from population import Population
from config import *

def simulate(config, savedir, n_jobs, loading_mode, initialize_file, transplantation_file, verbose):
    time_id = datetime.now()

    if not savedir is None:
        savedir = "output/%s"%savedir.replace("output/", "")
    ckpt_file = None

    # In case the directory already exist and a configuration file also
    # We just load it
    if savedir is not None and os.path.exists(savedir):
        with open("%s/config.json"%savedir, "r") as f:
            config = json.load(f)

        # TODO: Allow for modifications of config when loading (like training longer, ect..)

        # Verify checkpoints exist
        if loading_mode:
            if len(glob("%s/checkpoint-2020*"%savedir)) == 0:
                raise ValueError("No checkpoint to load.")
            else:
                ckpts=glob("%s/checkpoint-2020*"%savedir)
                print(list(zip(ckpts, map(os.path.getmtime, ckpts))))
                ckpt_file=ckpts[argmax(list(map(os.path.getmtime, ckpts)))]
                print("Loading the last edited checkpoint %s"%ckpt_file)

    # Otherwise we create it
    # Along with a directory regardless of whether a directory has been given
    else:
        if savedir is None:
            savedir = "output/default-%s"%time_id.strftime('%Y-%m-%d_%H-%M-%S')

        os.mkdir(savedir)
        # Save the configuration dict
        with open("%s/config.json"%savedir, 'w') as f:
            json.dump(config, f, indent=4)

    print(config)
    population = Population(config)

    results = population.evolve(savedir, n_jobs, ckpt_file, initialize_file, transplantation_file, verbose)

    elapsed = datetime.now() - time_id
    print("Total running time:", elapsed.seconds,":",elapsed.microseconds)

    return results

if __name__ == "__main__":
    # Parse arguments passed to the program (or set defaults)
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--saveDir", help="Config file to load (overrides other settings)")
    parser.add_argument("-j", "--numParallelJobs", type=int, default=1, help="Number of jobs launched in parallel")
    parser.add_argument("-v", "--verbose", type=int, choices=[0,1,2], default=0, help="Level of verbose output (higher is more)")
    parser.add_argument("--load", default=False, action="store_true", help="Loading an already existing population")
    parser.add_argument('--transplantate', help="Genome file to transplantate the population")
    parser.add_argument('--initialize', help="Genome file to initialize the population")

    parser.add_argument("-L", "--distance", type=int, choices=[3,5,7,9,11], help="Toric Code Distance")
    parser.add_argument("--errorRates", type=float, nargs="+", help="Qubit error rate")
    #parser.add_argument("--errorMode", type=int, choices=[0,1], help="Error generation mode")
    parser.add_argument("--errorModel", type=int, choices=[0,1], help="Error model")

    #
    #parser.add_argument('--networkType', help="Type of NN to evolve")
    #parser.add_argument('--rotationInvariantDecoder', type=int, choices=[0,1], help="Exploiting rotation symmetry")
    parser.add_argument("--rotationInvariantDecoder", default=True, action='store_true')
    parser.add_argument("--noRotationInvariantDecoder", dest='rotationInvariantDecoder', action='store_false')
    parser.add_argument("--trainingMode", type=int, choices=[0,1], help="Training mode")
    #parser.add_argument("--rewardMode", type=int, choices=[0,1], help="Reward mode")
    parser.add_argument("--numGenerations", type=int, help="Number of simulated generations")
    parser.add_argument("--initialConnection", nargs="+", help="Initial connection of the initial NN in the population")
    parser.add_argument("--numPuzzles", type=int, help="Number of syndrome configurations to solve per individual")
    parser.add_argument("--maxSteps", type=int, help="Number of maximum qubits flips to solve syndromes")
    parser.add_argument("--epsilon", type=float, help="Epsilon of the greedy search among perspectives results")
    #parser.add_argument("--substrateType", type=int, choices=[0, 1], help="Substrate type for hyperNEAT")
    parser.add_argument("--populationSize", type=int, help="Size of the population")
    parser.add_argument("--connectAddProb", type=float, help="Probability of adding a new connection")
    parser.add_argument("--addNodeProb", type=float, help="Probability of adding a new node")
    parser.add_argument("--weightMutateRate", type=float, help="Mutation rate for the weights")
    parser.add_argument("--weightMutatePower", type=float, help="Mutation power for the weights")
    parser.add_argument("--biasMutateRate", type=float, help="Mutation rate for the bias")
    parser.add_argument("--biasMutatePower", type=float, help="Mutation power for the bias")
    parser.add_argument("--compatibilityDisjointCoefficient", type=float, help="Weight on the number of disjoint genes used when calculating genome distance")
    parser.add_argument("--compatibilityWeightCoefficient", type=float, help="Weight on the L2 distance between connections weights used when calculating genome distance")
    parser.add_argument("--compatibilityThreshold", type=float, help="Distance threshold to form species")
    parser.add_argument("--speciesElitism", type=int, help="Minimal number of species")
    #parser.add_argument("--activationMutateRate", type=float, help="Activation function mutate rate")
    parser.add_argument("--activationOptions", nargs='+', help="Set of activation functions")
    parser.add_argument("--activationDefault", help="Default activation functions")
    parser.add_argument("--memory", action='store_true')
    parser.add_argument("--noMemory", dest='memory', action='store_false')
    args = parser.parse_args()

    config = from_arguments(args)
    config = check_config(config)

    simulate(config, args.saveDir, args.numParallelJobs, args.load, args.initialize, args.transplantate, args.verbose)
