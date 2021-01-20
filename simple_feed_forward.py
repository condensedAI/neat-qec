from neat.graphs import feed_forward_layers
from collections import namedtuple
import numpy as np
import copy
from scipy.special import expit

''' EFFICIENT IMPLEMENTATION
This type of neural network has
    - only one type of activation function per layer
    - aggregation_function is set to summing
which allows for matrix multiplication and speed up of the activation computation
'''

Layer = namedtuple("Layer", ["input_global_keys", "output_global_keys", \
                             "weight_matrix", "bias_vector", \
                             "act_function"])

# Vectorized version of activation function
# /!\ Actually the sigmoid from neat-python is sigmoid(5*x)
# The factor 5 is integrated in the weights and bias
def sigmoid(x):
    return expit(x)

class SimpleFeedForwardNetwork(object):
    def __init__(self, layers, global_output_keys, total_nodes):
        ''' param: layers of the neural network
        '''

        self.layers = layers
        self.global_output_keys = global_output_keys
        self.total_nodes = total_nodes

        # Stored variables to avoid reevaluation in activate
        self.n_layers = len(layers)
        self.n_nodes_first_layer = len(layers[0].input_global_keys) if len(layers)>0 else 0


    def activate(self, input_values):
        if self.n_layers == 0:
            return np.zeros(len(self.global_output_keys))
        if self.n_nodes_first_layer > len(input_values):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.layers[0].input_global_keys), len(input_values)))

        # Create a big vector which contains the value of the nodes
        # Note: it is not faster to have node_values as a class variable
        node_values = np.zeros(self.total_nodes)
        node_values[:len(input_values)] = input_values


        for l in self.layers:
            # Compute the vector of values of the next layer
            values = l.weight_matrix.dot(node_values[l.input_global_keys]) + l.bias_vector

            node_values[l.output_global_keys] = l.act_function(values)


        return node_values[self.global_output_keys]

    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (a FeedForwardNetwork). """

        # Ideas for optimization
        # - profiling the code
        # - memoization
        # - use contiguous array in memory

        # Gather expressed connections.
        connections = [cg.key for cg in genome.connections.values() if cg.enabled]

        # TODO: remove this call to feed_forward_layers
        # Because some work done in this function is done again here
        computational_layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections)

        # Insert the input layer as the first layer
        # It matters that the input layer is inserted as a list and not a set, otherwise ordering'd be messed up
        computational_layers.insert(0, config.genome_config.input_keys)

        # TODO: visited_nodes can be a set instead of a list, might speed-up
        # OrderedSet ?
        visited_nodes = copy.copy(config.genome_config.input_keys)

        # Each entry of the dict stores the global id of a node
        global_ids={}
        global_count=0
        global_OUT_keys=np.zeros(config.genome_config.num_outputs, dtype=int)
        for i, l in enumerate(computational_layers):
            for n in l:
                global_ids[n] = global_count

                # If the node is one the output of the NN, keep memory
                if n in config.genome_config.output_keys:
                    global_OUT_keys[n] = global_count

                global_count+=1

        build_layers=[]

        # Start with the first hidden "layer"
        for l in computational_layers[1:]:
            input_global_keys=[]
            output_global_keys=[]
            weight_matrix = []
            bias_vector = []

            # Go over the nodes in the computational layer
            for no, okey in enumerate(l):
                output_global_keys.append(global_ids[okey])
                bias_vector.append(genome.nodes[okey].bias)

                # Fill in the weight matrix
                for ikey in visited_nodes:

                    if (ikey, okey) in connections:
                        # If we created already this row in the weight matrix
                        if global_ids[ikey] in input_global_keys:
                            row_index = input_global_keys.index(global_ids[ikey])
                            weight_matrix[row_index][no] = genome.connections[(ikey, okey)].weight
                        else:

                            input_global_keys.append(global_ids[ikey])
                            weight_matrix.append(np.zeros(len(l)))
                            weight_matrix[-1][no] = genome.connections[(ikey, okey)].weight


                        connections.remove((ikey, okey))

            visited_nodes += list(l)

            ng = genome.nodes[0]
            if ng.activation == 'sigmoid':
                act_function = sigmoid
                # Conversion
                input_global_keys=np.array(input_global_keys)
                output_global_keys=np.array(output_global_keys)
                # The factor 5 is here to match the definition of sigmoid in neat-python library
                weight_matrix=5*np.matrix(weight_matrix).transpose()
                bias_vector=5*np.array(bias_vector)
            else:
                act_function = config.genome_config.activation_defs.get(ng.activation)
                input_global_keys=np.array(input_global_keys)
                output_global_keys=np.array(output_global_keys)
                weight_matrix=np.matrix(weight_matrix).transpose()
                bias_vector=np.array(bias_vector)

            build_layers.append(Layer(input_global_keys, output_global_keys, weight_matrix, bias_vector, act_function))

        return SimpleFeedForwardNetwork(build_layers, global_OUT_keys, global_count)
