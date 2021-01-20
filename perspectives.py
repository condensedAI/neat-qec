import numpy as np
from toric_game_env import Board

class Perspectives():
    def __init__(self, board_size, channels, memory):
        self.size = board_size

        self.op_pos={}
        self.qubit_pos, self.op_pos[0], self.op_pos[1]  = Board.component_positions(self.size)

        input_dim=len(channels)*self.size**2

        if memory:
            input_dim *= 3

        indices = np.arange(input_dim)

        # Loading the slices of the input
        slices={}
        index0, dim=0,0
        for channel in channels:
            dim=self.size**2
            slices['op_'+str(channel)]=indices[index0:index0+dim].reshape(self.size, self.size)
            index0+=dim

            if memory:
                dim=2*self.size**2
                slices['qubit_'+str(channel)]=indices[index0:index0+dim]
                index0+=dim

                # The qubit matrix is not squared
                dummy_squared=-np.ones((2*self.size, 2*self.size), dtype=np.int)
                for x in range(2*self.size):
                    for y in range(2*self.size):
                        if [x,y] in self.qubit_pos:
                            qubit_index = self.qubit_pos.index([x,y])
                            dummy_squared[x,y]=slices['qubit_'+str(channel)][qubit_index]

                slices['qubit_'+str(channel)]=dummy_squared

        #print("slices", slices)

        # For the qubits
        # To make sure the lattice is in the right convention
        # Having first row and first column having star operators
        # We sometimes need to shift it by one row or column
        rolling_axis_after_rotation=[[], [0], [0,1], [1]]

        # Define the board_size**2 ways of shifting the board
        self.perspectives = {i : {} for i in range(4)}
        for channel in [0,1]:
            for rot_i in range(4):
                for i in range(self.size):
                    for j in range(self.size):
                        # Shift the syndrome to central plaquette
                        index = j+self.size*i
                        plaq = self.op_pos[channel][index]

                        transformed_slices={}
                        for key in slices:
                            # Translate
                            factor = 2 if "qubit" in key else 1
                            center = int((self.size-1)/2) # Assume that size is odd
                            slice = np.roll(slices[key], factor*(center - i), axis=0)
                            slice = np.roll(slice, factor*(center - j), axis=1)

                            # Rotate
                            slice = np.rot90(slice, rot_i)

                            # Filter out dummy variable in qubits matrix
                            if "qubit" in key:
                                # Rotating the board causes the qubits to move out
                                # of their conventional location
                                slice = np.roll(slice, 1, rolling_axis_after_rotation[rot_i]).flatten()

                                filtered = np.argwhere(slice!=-1)
                                #print(filtered)
                                slice = slice[filtered]

                            transformed_slices[key] = slice.flatten()

                        # Concatenate the indices
                        self.perspectives[rot_i][tuple(plaq)] =np.array([], dtype=np.int)
                        # To use symmetric roles of plaquette and stars, we normalize the order of the input
                        index0, index1 = channel, (channel+1)%2
                        ordered_keys = ["op_"+str(index0), "qubit_"+str(index0), "op_"+str(index1), "qubit_"+str(index1)]
                        for key in ordered_keys:
                            if key in slices.keys():
                                #print(transformed_slices[key])
                                self.perspectives[rot_i][tuple(plaq)] = np.concatenate((self.perspectives[rot_i][tuple(plaq)], transformed_slices[key].flatten()))

        #print(self.perspectives[0])
    # Return the indices of the new lattice shifted such that the syndrome is placed at the central plaquette
    # Also allows for returning the reflected indices given by rotation_number
    def shift_from(self, plaq, rotation_number=0):
        return self.perspectives[rotation_number][(plaq[0], plaq[1])]
