
GameMode = {"TRAINING": 0,
            "EVALUATION": 1}

ErrorMode = {"PROBABILISTIC": 0, # The errors are generated to a binomial distribution
             "DETERMINISTIC": 1}# The number of generated errors is fixed by error_rate

ErrorModel = {"UNCORRELATED": 0,
              "DEPOLARIZING": 1}

TrainingMode = {"NORMAL" : 0, #
                "RESAMPLING": 1} # adapt the training dataset by resampling samples the NN struggle to solve

RewardMode = {"BINARY": 0, # Reward is 1 for solved and 0 otherwise
              "CURRICULUM": 1} # Harder problems solved are more positively rewarded, easier problems failed are more negatively rewarded

# Default configuration
def get_default_config():
    return {
        "Physics": {
            "distance" : 3,
            "error_model": ErrorModel["DEPOLARIZING"]
        },

        "Training" : {
            "n_generations" : 100,
            "network_type": 'ffnn',
            "memory": False,
            'rotation_invariant_decoder': True,
            "error_rates" : [0.01, 0.05, 0.1, 0.15],
            "error_mode": ErrorMode["PROBABILISTIC"],
            "reward_mode": RewardMode["BINARY"],
            "training_mode": TrainingMode["NORMAL"],
            "n_games" : 100,
            "max_steps" : 1000,
            "epsilon": 0.1,
            "substrate_type": 0
        },
        "Population" : {
            "pop_size" : 50,
            "initial_connection": 'full',
            "connect_add_prob" : 0.1,
            "add_node_prob" : 0.1,
            "weight_mutate_rate": 0.5,
            "weight_mutate_power": 0.5,
            "bias_mutate_rate": 0.1,
            "bias_mutate_power": 0.5,
            "compatibility_disjoint_coefficient" : 1,
            "compatibility_weight_coefficient" : 2,
            "compatibility_threshold" : 6,
            "species_elitism": 2,
            "activation_mutate_rate": 0,
            "activation_options": "sigmoid",
            "activation_default": "sigmoid"
        }
}

def from_arguments(args):
    config = get_default_config()

    key_converts={"distance":"distance",
                  "numGenerations":"n_generations",
                  "networkType": "network_type",
                  'memory': 'memory',
                  "rotationInvariantDecoder": "rotation_invariant_decoder",
                  "errorMode" : "error_mode",
                  "errorModel" : "error_model",
                  "errorRates": "error_rates",
                  "trainingMode": "training_mode",
                  "rewardMode": "reward_mode",
                  "numPuzzles": "n_games",
                  "maxSteps": "max_steps",
                  "epsilon": "epsilon",
                  "populationSize": "pop_size",
                  "initialConnection": "initial_connection",
                  "connectAddProb" : "connect_add_prob",
                  "addNodeProb": "add_node_prob",
                  "weightMutateRate" : "weight_mutate_rate",
                  "weightMutatePower" : "weight_mutate_power",
                  "biasMutateRate": "bias_mutate_rate",
                  "biasMutatePower": "bias_mutate_power",
                  "compatibilityDisjointCoefficient" : "compatibility_disjoint_coefficient",
                  "compatibilityWeightCoefficient": "compatibility_weight_coefficient",
                  "compatibilityThreshold": "compatibility_threshold",
                  "speciesElitism": "species_elitism",
                  "activationMutateRate": "activation_mutate_rate",
                  "activationOptions": "activation_options",
                  "activationDefault": "activation_default",
                  "substrateType": "substrate_type"}

    for key, value in vars(args).items():
        if not value is None:
            try:
                new_key = key_converts[key]
                config[key_to_section(new_key)][new_key] = value
            except:
                print("The key %s is not recognised for config."%str(key))
                continue

    return config

def key_to_section(key):
    if key in ["distance", "error_model"]:
        return "Physics"
    if key in ["n_generations", "n_games", "max_steps",
                "epsilon", "error_rates", "error_mode",
                "training_mode", "reward_mode", "network_type",
                "rotation_invariant_decoder", "substrate_type", "memory"]:
        return "Training"
    if key in ["pop_size", "connect_add_prob", "add_node_prob",
        "weight_mutate_rate", "weight_mutate_power", "bias_mutate_rate", "bias_mutate_power", "compatibility_disjoint_coefficient",
        "compatibility_weight_coefficient", "compatibility_threshold", "initial_connection",
        "species_elitism", "activation_mutate_rate", "activation_options", "activation_default"]:
        return "Population"

    raise ValueError("Missing key for %s"%key)

def solve_compatibilities(config):
    default_config = get_default_config()
    for key1 in default_config.keys():
        for key2 in default_config[key1].keys():
            if key1 in config and not key2 in config[key1]:
                print("%s is set to the default value: %s"%(key2, str(default_config[key1][key2])))

            elif key1 in config:
                default_config[key1][key2] = config[key1][key2]

    return default_config


def check_config(config):
    if "initial_connection" in config["Population"] and not isinstance(config["Population"]["initial_connection"], str):
        print(config["Population"]["initial_connection"])
        # This entry can take multiple strings, so we need to concatenate them
        if len(config["Population"]["initial_connection"]) > 1:
            config["Population"]["initial_connection"] = ' '.join(config["Population"]["initial_connection"])
        else:
            config["Population"]["initial_connection"] = config["Population"]["initial_connection"][0]

    if "activation_options" in config["Population"] and not isinstance(config["Population"]["activation_options"], str):
        print(config["Population"]["activation_options"])
        # This entry can take multiple strings, so we need to concatenate them
        if len(config["Population"]["activation_options"]) > 1:
            config["Population"]["activation_options"] = ' '.join(config["Population"]["activation_options"])
        else:
            config["Population"]["activation_options"] = config["Population"]["activation_options"][0]

    return solve_compatibilities(config)

def generate_config_file(savedir, cf, distance=None):
    # Change the config file according to the given parameters
    with open('config-toric-code-template-%s'%(cf["Training"]["network_type"])) as file:
        data = file.read()

        # Input dimension
        input_size= cf["Physics"]["distance"]**2 if distance is None else distance**2

        if cf["Physics"]["error_model"] == ErrorModel["DEPOLARIZING"]:
            input_size *= 2

        if cf["Training"]["memory"]:
            input_size *= 3

        data = data.replace("{num_inputs}", str(input_size))

        # Output dimension
        if cf["Physics"]["error_model"] == ErrorModel["UNCORRELATED"]:
            output_size = 1

        elif cf["Physics"]["error_model"] == ErrorModel["DEPOLARIZING"]:
            output_size = 3

        if cf["Training"]["rotation_invariant_decoder"] == False:
            output_size*=4

        data = data.replace("{num_outputs}", str(output_size))

        # Loop over the parameters of the simulation
        for param_name, param_value in cf["Population"].items():
            # Attributes like n_games or epsilon do not appear in config template
            # So no need to check
            data = data.replace("{"+param_name+"}", str(param_value))

        # Create a config file corresponding to these settings
        if distance is None:
            new_file = open(savedir+"/population-config", "w")
        else:
            new_file = open(savedir+"/population-config-temp-d"+str(distance), "w")
        new_file.write(data)
        new_file.close()
