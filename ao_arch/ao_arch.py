# -*- coding: utf-8 -*-
"""
// aolabs.ai software >ao_core/Arch.py (C) 2023 Animo Omnis Corporation. All Rights Reserved.

Open source; temporarily under the MIT license as per the README.md in parent repository; final license and more info TBD at aolabs.ai.

Thank you for your curiosity!
"""

## // About :: https://docs.aolabs.ai/docs/arch
#
# This class defines the Agent ARCHitecture logic used to configure Agents, comprised of 2 components: 
#    1) how many binary neurons to encode your input-output associations (required),
#    2) How those neurons are connected (optional, more advanced-- best to start with the default of fully-connected network of neurons and them trim down)
#
# There are currently 3 archs you can use as reference designs for your own configuration-- a basic clam, an MNIST classifier, and a NetBox device discovery classifier (find them in the root directory of this repo).
#
# For an interactive visual representation of archs to help design your own, visit: https://miro.com/app/board/uXjVM_kESvI=/?share_link_id=72701488535
#
# We eagerly welcome contributors who relate to these ideas. :)
#
#
###############################################################################
#
# arch = Arch(arch_i, arch_z, arch_c, connector_function, description)
#
# if you added extra C neurons beyond the 4 default, then after creating the arch instance, program the C neuron like below:
#    # arch.datamatrix[4, arch.C[1]]= "define a instinct new_function, like ln 92-98"
#    # also you may wish to change how the connections of the new C neuron: arch.datamatrix[3, arch.C[1]]= connections of new C channel
#
##############################################################################

# Standard library
import random as rn
# 3rd party
import numpy as np
# Arch class
from .functions import nearest_points, rectangle_points


class Arch(object):
    """Arch constructor class."""
    def __init__(self, arch_i, arch_z, arch_c=[], connector_function="full_conn",  connector_parameters=(), description=""):
        super(Arch, self).__init__()
        self.i = arch_i
        self.q = self.i.copy()
        self.z = arch_z
        self.c = [3]+arch_c
        self.connector_function = connector_function
        self.connector_parameters = connector_parameters
        self.description = description

        # Neuron Sets
        self.sets = [self.i, self.q, self.z, self.c]
        self.sets_labels = ["I", "Q", "Z", "C"]
        # I - Input neurons: 0 or 1 depending on fixed ENV decoding
        # Q - State or interneurons: 0 or 1 depending on learned lookup tabled comprised of connected neurons
        # Z - Output neurons: also learning binary neurons like Q, except Z actuates Agent in environment
        # C - Control neurons: 0 or 1 depending on designer defined label or trigger like instincts to activate learning

        # Creating nids in Channels in Sets
        si = 0     # sets, i.e. category of neurons corresponding to major type, i.g. I or Z or C
        neuron_counter = 0
        for s in self.sets:
            
            Set_label = self.sets_labels[si]
            Set = self.sets[si].copy()
            
            self.__setattr__(Set_label, Set)
            self.__setattr__(Set_label+"__flat", [])
            
            ci = 0     # channel, i.e. subset of neuron set corresponding to application data channels, e.g. there are 3 Inputs in the Basic Clam: F, A, B
            for c in self.__dict__[Set_label]:
                self.__dict__[Set_label][ci] = list(range(neuron_counter, neuron_counter+c))
                self.__dict__[Set_label+"__flat"] += self.__dict__[Set_label][ci]
                neuron_counter += c
                ci += 1
            self.__dict__[Set_label+"__flat"] = np.array(self.__dict__[Set_label+"__flat"])
            si += 1
    
        self.n_total = sum(self.i + self.q + self.z + self.c)

        self.IQZC = np.concatenate((self.I__flat, self.Q__flat, self.Z__flat, self.C__flat))
        self.IQZ  = np.concatenate((self.I__flat, self.Q__flat, self.Z__flat))
        self.QZ__flat = np.concatenate((self.Q__flat, self.Z__flat))        # remove flat from ao_core later for consistency
        
        self.C__flat_command = np.array(self.C[0])     # the first C channel always contains the command neurons which are default to each Agent
        self.C__flat_pleasure= np.array([self.C[0][0], self.C[0][1]])
        self.C__flat_pain    = np.array([self.C[0][2]])
        
        # Defining Neuron metadata -- the connections of neurons (i.e. which neurons constitute each others' lookup tables)
        self.datamatrix = np.zeros([5, self.n_total], dtype="O")
        # 5 rows, as follows:
            #0 Type
            #1 Input Connections
            #2 Neighbor Connections
            #3 C Connections
            #4 Dominant Connection
            #    ** note; the dominant connection is critical; it is why Q is made in the shape/size of I, so that each Q has a corresponding I as dominant connection (the dominant connection for Z is its own past state [-1]; since if the NSM did something "good / triggered C(s) pleasure neuron(s)" during iconic training, Q will be dominated by I and Z by its past Z (the training becomes; given C at state s, store I(s) and Z(s-1) since Z(s-1) led to the I(s) which triggered C(s)
        
        self.datamatrix[0, self.I__flat] = "Input"
        self.datamatrix[0, self.Q__flat] = "CGA Q"
        self.datamatrix[0, self.Z__flat] = "CGA Z"
        self.datamatrix[0, self.C__flat] = "Control"

        ## Defining C control propertires
        self.datamatrix[4, self.C[0][0]] = "Default if label"
        self.datamatrix[4, self.C[0][1]] = "C+ pleasure signal"
        self.datamatrix[4, self.C[0][2]] = "C- pain signal"

    ## Connector functions follow
    #     
    # Note: Neurons are hierarchically grouped as follows: 
    # ----- Set: fixed; major set of neurons, I-input, Q-inter, Z-output, or C-control only
    # --------- Channel: groups of neurons that share a similar function (i.e. 10 neurons for food or for a name ID)
    # -------------- Group: coming soon; subset of channels of neurons
    #
    # Neurons can be connected in a number of ways across groups and channels. Below are some connector_functions for your convenience.
    # 
    # You can also manually specifiy the connections for your particular arch directly in the datamatrix, or write your own connector_function to do so.

        if self.connector_function=="full_conn":
            """Fully connect the neurons-- Q to all I and Q; Z to all Q and Z"""
        
        #    for Channel in self.I:   # I has no incoming connections; input is supplied ex machina (by the env)
        
            for Channel in self.Q:
                for n in Channel:
                    self.datamatrix[1, n] = sorted(self.I__flat)
                    self.datamatrix[2, n] = sorted(self.Q__flat)
                    self.datamatrix[3, n] = sorted(self.C__flat)
                    self.datamatrix[4, n] = n - sum(self.q)
                    
            for Channel in self.Z:
                for n in Channel:
                    self.datamatrix[1, n] = sorted(self.Q__flat)
                    self.datamatrix[2, n] = sorted(self.Z__flat)
                    self.datamatrix[3, n] = sorted(self.C__flat)
                    self.datamatrix[4, n] = n
                
            for Channel in self.C:
                for n in Channel:
                    self.datamatrix[3, n] = sorted(self.Q__flat) + sorted(self.Z__flat)
                    
                self.datamatrix_type = 'full_conn'
    
    
        elif self.connector_function == "forward_full_conn":    
            """Fully connect the neurons input-wise-- Q channel to *all* I and itself; Z channel to all Q and itself"""
        
            for Channel in self.Q:
                for n in Channel:
                    self.datamatrix[1, n] = sorted(self.I__flat)
                    self.datamatrix[2, n] = sorted(Channel)
                    self.datamatrix[3, n] = sorted(self.C__flat)
                    self.datamatrix[4, n] = n - sum(self.q)
                    
            for Channel in self.Z:
                for n in Channel:
                    self.datamatrix[1, n] = sorted(self.Q__flat)
                    self.datamatrix[2, n] = sorted(Channel)
                    self.datamatrix[3, n] = sorted(self.C__flat)
                    self.datamatrix[4, n] = n
                
            for Channel in self.C:
                for n in Channel:
                    self.datamatrix[3, n] = sorted(self.Q__flat) + sorted(self.Z__flat)
                    
                self.datamatrix_type = 'forward_full_conn'
    
    
        elif self.connector_function == "forward_forward_conn":    
            """Fully connect the neurons forward only-- Q channel to *corresponding* I and itself; Z channel to all Q and itself"""
        
            ci = 0
            for Channel in self.Q:
                for n in Channel:
                    self.datamatrix[1, n] = sorted(self.I[ci])
                    self.datamatrix[2, n] = sorted(Channel)
                    self.datamatrix[3, n] = sorted(self.C__flat)
                    self.datamatrix[4, n] = n - sum(self.q)
                ci += 1
            
            for Channel in self.Z:
                for n in Channel:
                    self.datamatrix[1, n] = sorted(self.Q__flat)
                    self.datamatrix[2, n] = sorted(Channel)
                    self.datamatrix[3, n] = sorted(self.C__flat)
                    self.datamatrix[4, n] = n
                
            for Channel in self.C:
                for n in Channel:
                    self.datamatrix[3, n] = sorted(self.Q__flat) + sorted(self.Z__flat)
                    
                self.datamatrix_type = 'forward_forward_conn'
    
    
        elif self.connector_function == "rand_conn":
            """Connect neurons randomly (choose how many random connections per neuron set).
        
            Keyword arguments:
            q_in_conn -- int of random I-input connections for Q neurons
            q_ne_conn -- int of random Q-neighbor connections for Q neurons
            z_in_conn -- int of random Q-input connections for Z neurons
            z_ne_conn -- int of random Z-neighbor connections for Z neurons
            """    
            if len(connector_parameters) != 4:
                raise ValueError(f"expected 'connector_parameters' of length 4, got length {len(connector_parameters)}")
            if not all(isinstance(x, int) for x in connector_parameters):
                raise TypeError(f"expected 'connector_parameters' of type 'int'")


            q_in_conn = self.connector_parameters[0]
            q_ne_conn = self.connector_parameters[1]
            z_in_conn = self.connector_parameters[2]
            z_ne_conn = self.connector_parameters[3]
        
            for Channel in self.Q:
                for n in Channel:
                    self.datamatrix[1, n] = sorted(rn.sample(list(self.I__flat), q_in_conn))
                    self.datamatrix[2, n] = sorted(rn.sample(list(self.Q__flat), q_ne_conn))
                    self.datamatrix[3, n] = sorted(self.C__flat)
                    self.datamatrix[4, n] = n - sum(self.q)
                    
            for Channel in self.Z:
                for n in Channel:
                    self.datamatrix[1, n] = sorted(rn.sample(list(self.Q__flat), z_in_conn))
                    self.datamatrix[2, n] = sorted(rn.sample(list(self.Z__flat), z_ne_conn))
                    self.datamatrix[3, n] = sorted(self.C__flat)
                    self.datamatrix[4, n] = n
                
            for Channel in self.C:
                for n in Channel:
                    self.datamatrix[3, n] = sorted(self.Q__flat) + sorted(self.Z__flat)
                    
            self.datamatrix_type = "rand_conn "+str(q_in_conn)+"-"+str(q_ne_conn)+"--"+str(z_in_conn)+"-"+str(z_ne_conn)
            

        elif self.connector_function == "nearest_neighbour_conn":    
            """Connects channels of neurons in a grid to their nearest neighbors (channels)-- choose how far the connections go along both axis (ax) and along diagonals (dg).
            
            Keyword arguments:
            ax -- int of how many neighbors to connect to along x-y axis
            dg -- int of how many neighbors to connect to diagonally
            neurons_x -- int of X-dimension of grid
            neurons_y -- int of Y-dimension of grid
            Z2I_connections -- boolean, for if Z is connected to corresponding I
            z_in_conn -- int of random Q-input connections for Z neurons (optional, provide if Z and Q have different sizes)
            """
            
            if len(connector_parameters) < 5 or len(connector_parameters) > 6:
                raise ValueError(f"expected 'connector_parameters' of length 5 or 6, got length {len(connector_parameters)}")
            if arch_i != arch_z and len(connector_parameters) < 6:
                raise ValueError(f"expected 6th parameter in 'connector_parameters' when arch_i != arch_z")
            # type check params
            if ((not all(isinstance(x, int) for x in connector_parameters[:4])) and
                (not isinstance(connector_parameters[4], bool)) and
                ((len(connector_parameters) < 6) or (not isinstance(connector_parameters[5], int)))):
                raise TypeError(f"expected 'connector_parameters' of type 'int'")


            ax = int(self.connector_parameters[0])
            dg = int(self.connector_parameters[1])
            neurons_x = int(self.connector_parameters[2])
            neurons_y = int(self.connector_parameters[3])
            Z2I_connections = self.connector_parameters[4]  #True or False

            if neurons_x * neurons_y != len(arch_i):
                raise ValueError(f"Expected 'neurons_x*neurons_y' to equal the {len(arch_i)}")

            ch = 0
            row = 0
            col = 0
            for Channel in self.Q:
                neighbour_indices = nearest_points(row, col, ax, dg, size=(neurons_x, neurons_y))
                input_con = sorted(self.I[ch])
                neigh_con = sorted(Channel)
                for index in neighbour_indices:
                    temp_ch = int(( (neurons_x)*index[0] ) + index[1]) 
                    input_con = input_con + sorted(self.I[temp_ch])
                    neigh_con = neigh_con + sorted(self.Q[temp_ch])

                for n in Channel:
                    self.datamatrix[1, n] = sorted(input_con)
                    self.datamatrix[2, n] = sorted(neigh_con)
                    self.datamatrix[3, n] = sorted(self.C__flat)
                    self.datamatrix[4, n] = n - sum(self.q)
                ch += 1
                col += 1 
                if ch%(neurons_x) == 0:
                    row += 1
                    col = 0

            ch = 0
            row = 0
            col = 0
            for Channel in self.Z:
                neighbour_indices = nearest_points(row, col, ax, dg, size=(neurons_x, neurons_y))
                input_con = (sorted(self.I[ch]) if Z2I_connections else [])+ sorted(self.Q[ch])
                neigh_con = sorted(Channel)
                
                if len(self.Z) != len(self.Q):
                    if self.connector_parameters[5]: #if random is true 
                        z_in_conn = self.connector_parameters[5]
                        input_con = sorted(rn.sample(list(self.Q__flat), z_in_conn))

                    else:  #if random is not true, perform full connection
                        input_con = (sorted(self.I__flat) if Z2I_connections else [])+ sorted(self.Q__flat)

                else:    
                    for index in neighbour_indices:
                        temp_ch = int(( (neurons_x)*index[0] ) + index[1])    
                        input_con = input_con + (sorted(self.I[temp_ch]) if Z2I_connections else []) + sorted(self.Q[temp_ch])
                        neigh_con = neigh_con + sorted(self.Z[temp_ch])

                for n in Channel:
                    self.datamatrix[1, n] = sorted(input_con)
                    self.datamatrix[2, n] = sorted(neigh_con)
                    self.datamatrix[3, n] = sorted(self.C__flat)
                    self.datamatrix[4, n] = n - sum(self.q)
                ch += 1
                col += 1 
                if ch%(neurons_x) == 0:
                    row += 1
                    col = 0
                
            for Channel in self.C:
                for n in Channel:
                    self.datamatrix[3, n] = sorted(self.Q__flat) + sorted(self.Z__flat)
                    
                self.datamatrix_type = 'nearest_neighbour_conn'
        
        

        elif self.connector_function == "rectangular_conn":    
            """Connects channels of neurons in a grid to their nearest neighbors (channels)-- choose how far the connections go along both axis (ax) and along diagonals (dg).
            
            Keyword arguments:
            x -- int, represents the number of units along one side of the x-axis (total rectangle width will be 2x).
            y -- int, represents the number of units along one side of the y-axis (total rectangle height will be 2y).
            neurons_x -- int of X-dimension of grid
            neurons_y -- int of Y-dimension of grid
            Z2I_connections -- boolean, for if Z is connected to corresponding I
            z_in_conn -- int of random Q-input connections for Z neurons (optional, provide if Z and Q have different sizes)
            """
            
            if len(connector_parameters) < 5 or len(connector_parameters) > 6:
                raise ValueError(f"expected 'connector_parameters' of length 5 or 6, got length {len(connector_parameters)}")
            if arch_i != arch_z and len(connector_parameters) < 6:
                raise ValueError(f"expected 6th parameter in 'connector_parameters' when arch_i != arch_z")
            if ((not all(isinstance(x, int) for x in connector_parameters[:4])) and
                (not isinstance(connector_parameters[4], bool)) and
                ((len(connector_parameters) < 6) or (not isinstance(connector_parameters[5], int)))):
                raise TypeError(f"invalid type(s) in 'connector_parameters'")

            x = int(self.connector_parameters[0])
            y = int(self.connector_parameters[1])
            neurons_x = int(self.connector_parameters[2])
            neurons_y = int(self.connector_parameters[3])
            Z2I_connections = self.connector_parameters[4]  #True or False

            if neurons_x * neurons_y != len(arch_i):
                raise ValueError(f"Expected 'neurons_x*neurons_y' to equal the {len(arch_i)}")

            ch = 0
            row = 0
            col = 0
            for Channel in self.Q:
                neighbour_indices = rectangle_points(row, col, x, y, size=(neurons_x, neurons_y))
                input_con = sorted(self.I[ch])
                neigh_con = sorted(Channel)
                for index in neighbour_indices:
                    temp_ch = int(( (neurons_x)*index[0] ) + index[1]) 
                    input_con = input_con + sorted(self.I[temp_ch])
                    neigh_con = neigh_con + sorted(self.Q[temp_ch])

                for n in Channel:
                    self.datamatrix[1, n] = sorted(input_con)
                    self.datamatrix[2, n] = sorted(neigh_con)
                    self.datamatrix[3, n] = sorted(self.C__flat)
                    self.datamatrix[4, n] = n - sum(self.q)
                ch += 1
                col += 1 
                if ch%(neurons_x) == 0:
                    row += 1
                    col = 0

            ch = 0
            row = 0
            col = 0
            for Channel in self.Z:
                neighbour_indices = rectangle_points(row, col, x, y, size=(neurons_x, neurons_y))
                input_con = (sorted(self.I[ch]) if Z2I_connections else [])+ sorted(self.Q[ch])
                neigh_con = sorted(Channel)
                
                if len(self.Z) != len(self.Q):
                    if self.connector_parameters[5]: #if random is true 
                        z_in_conn = self.connector_parameters[5]
                        input_con = sorted(rn.sample(list(self.Q__flat), z_in_conn))

                    else:  #if random is not true, perform full connection
                        input_con = (sorted(self.I__flat) if Z2I_connections else [])+ sorted(self.Q__flat)

                else:    
                    for index in neighbour_indices:
                        temp_ch = int(( (neurons_x)*index[0] ) + index[1])    
                        input_con = input_con + (sorted(self.I[temp_ch]) if Z2I_connections else []) + sorted(self.Q[temp_ch])
                        neigh_con = neigh_con + sorted(self.Z[temp_ch])

                for n in Channel:
                    self.datamatrix[1, n] = sorted(input_con)
                    self.datamatrix[2, n] = sorted(neigh_con)
                    self.datamatrix[3, n] = sorted(self.C__flat)
                    self.datamatrix[4, n] = n - sum(self.q)
                ch += 1
                col += 1 
                if ch%(neurons_x) == 0:
                    row += 1
                    col = 0
                
            for Channel in self.C:
                for n in Channel:
                    self.datamatrix[3, n] = sorted(self.Q__flat) + sorted(self.Z__flat)
                    
                self.datamatrix_type = 'rectangular_conn'
        else:
            raise ValueError("invalid connector function")
