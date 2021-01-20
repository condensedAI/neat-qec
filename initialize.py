import neat
import numpy as np
import os
import pickle
import json

def initialize_population(p, initialize_file):
    # Load the genome to initialize the population
    if not os.path.exists(initialize_file):
        raise ValueError("Genome file for initialization does not exist.")
    with open(initialize_file, "rb") as f:
        genome_giv = pickle.load(f)

    for key, genome in p.population.items():
        # Clear genome
        genome.connections = genome_giv.connections
        genome.nodes = genome_giv.nodes
