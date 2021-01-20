import neat
import numpy as np
import os
import pickle
import json

def transplantate_population(p, transplantation_file, config_rec, size_rec, neat_config):
    # Load the genome to be transplantated
    if not os.path.exists(transplantation_file):
        raise ValueError("Genome file does not exist.")
    with open(transplantation_file, "rb") as f:
        genome_giv = pickle.load(f)

    # Load the board size of transplanted genome
    transplanted_dir=transplantation_file[:transplantation_file.rfind("/")]
    if not os.path.exists("%s/config.json"%transplanted_dir):
        raise ValueError("Configuration file does not exist (%s)."%("%s/config.json"%transplanted_dir))

    with open("%s/config.json"%transplanted_dir) as f:
        transplantated_config = json.load(f)

    size_giv = transplantated_config["Physics"]["distance"]

    for key, genome in p.population.items():
        # Clear genome
        genome.connections = {}
        genome.nodes = {}
        # Transplantate
        # TODO: Maybe incorporate a bit of mutation, so far the population is composed of the same individual
        transplantate(config_rec, genome, size_rec, genome_giv, size_giv, neat_config)

# Transplantate the genome giver into the genome receiver
def transplantate(config_rec, genome_rec, size_rec, genome_giv, size_giv, neat_config):
    if size_rec <= size_giv or (size_rec-size_giv)%2!=0:
        raise ValueError("The board size of the genomes to copy does not fit the requirements.")

    channels=[0] if neat_config['Physics']['error_model'] == 0 else [0,1]
    memory=neat_config['Training']['memory']


    # Look-up table for the nodes in genome_giv
    lookup_keys = {}

    # Create node genes for the output keys
    for node_key in config_rec.output_keys:
        new_node = genome_rec.create_node(config_rec, node_key)
        new_node.bias = genome_giv.nodes[node_key].bias
        genome_rec.nodes[node_key] = new_node

        lookup_keys[node_key] = node_key

    ## Add hidden nodes from to_copy_genome
    # TODO: treat case where hidden nodes are requested in config_rec
    hidden_nodes = [k for k in genome_giv.nodes if k not in config_rec.output_keys]

    for previous_key in hidden_nodes:
        node_key = config_rec.get_new_node_key(genome_rec.nodes)
        assert node_key not in genome_rec.nodes
        node = genome_rec.create_node(config_rec, node_key)
        genome_rec.nodes[node_key] = node

        # Keep track
        lookup_keys[previous_key] = node_key

    ## Add connections from to_copy_genome

    # First create the mask of nodes from to_copy_genome
    if memory:
        raise ValueError("Transplantation with memory is not yet supported.")

    L=size_rec
    L_copy=size_giv
    to_copy_mask = np.zeros((L, L))

    for i in range(L):
        for j in range(L):
            if i < L_copy and j < L_copy:
                to_copy_mask[i,j] = True

    to_copy_mask = np.roll(to_copy_mask, int((size_rec - size_giv)/2), axis=[0,1])
    print(to_copy_mask)
    to_copy_mask = to_copy_mask.flatten().tolist()

    to_copy_mask *= len(channels)

    # Loop over the input nodes and finish the lookup table
    count=-1
    for i, to_copy in enumerate(to_copy_mask):
        if to_copy:
            lookup_keys[count] = -i-1
            count -= 1

    print(lookup_keys)

    # Create connections
    for input_id, output_id in genome_giv.connections:
        #print(lookup_keys[input_id], lookup_keys[output_id])
        connection = genome_rec.create_connection(config_rec, lookup_keys[input_id], lookup_keys[output_id])
        connection.weight = genome_giv.connections[input_id, output_id].weight
        connection.enabled = genome_giv.connections[input_id, output_id].enabled
        genome_rec.connections[connection.key] = connection

    #print([(key, conn.weight) for key, conn in genome_rec.connections.items()])

    # Add other 0-weight (negligible) connections
    for channel in channels:
        for i in range(len(to_copy_mask)):
            input_id = -i-1
            for node in genome_rec.nodes:
                if np.random.rand() < 0.5 and not (input_id, node) in genome_rec.connections:
                    connection = genome_rec.create_connection(config_rec, input_id, node)
                    connection.weight = 0
                    connection.enabled = True
                    genome_rec.connections[connection.key] = connection

    #print([(key, conn.weight) for key, conn in genome_rec.connections.items()])
