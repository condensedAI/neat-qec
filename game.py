
from scipy.special import softmax
import numpy as np
import time
import neat

from toric_game_env import ToricGameEnv
from perspectives import Perspectives

from config import GameMode, RewardMode, ErrorModel

class ToricCodeGame():
    def __init__(self, config):
        self.board_size = config["Physics"]["distance"]
        self.error_model = config["Physics"]["error_model"]
        self.max_steps = config["Training"]["max_steps"]
        self.epsilon = config["Training"]["epsilon"]
        self.rotation_invariant_decoder = config["Training"]["rotation_invariant_decoder"]

        if self.error_model == ErrorModel['UNCORRELATED']:
            channels=[0]
        elif self.error_model == ErrorModel['DEPOLARIZING']:
            channels=[0,1]

        self.env = ToricGameEnv(self.board_size, self.error_model, channels, config['Training']['memory'])

        # Very important to change the seed here
        # Otherwise for game evaluation in parallel
        # All game objects will share the same seed leading to biased results
        np.random.seed()

        # The perspective includes the masking of star operators
        self.perspectives = Perspectives(self.board_size,
                            channels, config['Training']['memory'])

    # Return the score of the game
    # In evaluation mode, the fitness is in {0, 1} corresponding to success or failure
    # In training mode, fitness can be defined differently
    def play(self, nn, error_rate, reward_mode, mode, seed=None, without_illegal_actions=False, verbose=False):

        current_state = self.env.generate_errors(error_rate)

        # If there is no syndrome in the initial configuration
        # Either we generate a new one containing syndromes
        # Or if there happens to be a logical error, we return a failure
        if mode == GameMode["TRAINING"]:
            while self.env.done and error_rate>0:
                if self.env.reward == -1:
                    # In both reward modes BINARY or CURRICULUM, since it is not possible to correct these errors, there is no reward nor penalty
                    return {"fitness":0, "error_rate": error_rate, "outcome":"logical_error", "nsteps":0}

                current_state = self.env.generate_errors(error_rate)

        # In evaluation mode, we keep even these empty initial configurations
        elif self.env.done:
            if self.env.reward == -1:
                return {"fitness":0, "error_rate": error_rate, "outcome":"logical_error", "nsteps":0}
            else:
                return {"fitness":1, "error_rate": error_rate, "outcome":"success", "nsteps":0}

        if verbose:
            print("Initial", current_state)
            print(self.env.done, self.env.state.syndrome_pos)
            if verbose > 1: self.env.render()

        for step in range(self.max_steps+1):
            current_state = current_state.flatten()

            if self.rotation_invariant_decoder:
                probs, locations, actions = self._get_actions_with_rotations(nn, current_state)
            else:
                probs, locations, actions = self._get_actions(nn, current_state)

            # To avoid calling rand() when evaluating (for testing purposes)
            if self.epsilon == 0 or mode == GameMode["EVALUATION"]:
                index = np.argmax(probs)
            else:
                # epsilon-greedy search
                if np.random.rand() < self.epsilon:
                    index = np.random.randint(len(actions))
                else:
                    index = np.argmax(probs)

            action=actions[index]
            location=locations[index]

            current_state, reward, done, info = self.env.step(location, action, without_illegal_actions)

            if verbose:
                print(step, current_state, reward, location, action, info["message"])
                if verbose > 1: self.env.render()

            # if no syndromes are present anymore
            if done:
                # Fitness is 1 if there is no logical error
                # 0 if there is a logical error
                return {"fitness":(reward+1)/2, "error_rate": error_rate, "outcome":info["message"], "nsteps":step}

        # When the number of moves went beyond max_steps
        return {"fitness":0, "error_rate": error_rate, "outcome":"max_steps", "nsteps":self.max_steps}

    # In this case, the output layer has 3 node (rotation-invariance and 3 actions)
    def _get_actions(self, nn, current_state):

        # Go over syndromes
        locations, actions, probs=[], [], []
        for plaq in self.env.state.syndrome_pos:
            indices = self.perspectives.shift_from(plaq)
            input = current_state[indices]
            probs += list(nn.activate(input)) # NN outputs 3*4 values

            # Rotation  order corresponding the 90degree anti-clockwise rotation
            rots=[[-1, 0], [0,1], [1,0], [0,-1]]

            for rot_i in range(4):

                # Location of the reference qubit
                ref_qubit=[(rots[rot_i][0]+plaq[0])%(2*self.board_size),
                            (rots[rot_i][1]+plaq[1])%(2*self.board_size)]

                if self.error_model == ErrorModel['UNCORRELATED']:
                    locations += [ref_qubit]
                    actions += [0]
                elif self.error_model == ErrorModel['DEPOLARIZING']:
                    locations += [ref_qubit, ref_qubit, ref_qubit]

                    if plaq in self.env.state.plaquet_pos:
                        actions += [0,1,2]
                    elif plaq in self.env.state.star_pos:
                        actions += [1,0,2]

        return probs, locations, actions

    # In this case, the output layer has 3 node (rotation-invariance and 3 actions)
    def _get_actions_with_rotations(self, nn, current_state):

        # Go over syndromes
        locations, actions, probs=[], [], []
        for plaq in self.env.state.syndrome_pos:
            # Rotation  order corresponding the 90degree anti-clockwise rotation
            rots=[[-1, 0], [0,1], [1,0], [0,-1]]

            for rot_i in range(4):
                indices = self.perspectives.shift_from(plaq, rot_i)
                input = current_state[indices]
                probs += list(nn.activate(input))

                # Location of the reference qubit
                ref_qubit=[(rots[rot_i][0]+plaq[0])%(2*self.board_size),
                            (rots[rot_i][1]+plaq[1])%(2*self.board_size)]

                if self.error_model == ErrorModel['UNCORRELATED']:
                    locations += [ref_qubit]
                    actions += [0]
                elif self.error_model == ErrorModel['DEPOLARIZING']:
                    locations += [ref_qubit, ref_qubit, ref_qubit]

                    if plaq in self.env.state.plaquet_pos:
                        actions += [0,1,2]
                    elif plaq in self.env.state.star_pos:
                        actions += [1,0,2]

        return probs, locations, actions

    def close(self):
        self.env.close()
