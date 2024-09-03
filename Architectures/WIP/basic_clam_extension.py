# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 00:37:29 2022

@author: alebr
"""

## AO Labs Modules
import ao_core as ao

# 3rd Party Modules
import numpy as np


#####

# - Contains 3 seperate archs
# - Needs to be modified to handle the extra C command neurons (Cpos and Cneg)

#%%



b4Clam = ao.Arch("4 INPUT neurons, 2 OUTPUTs - Same as our 11 neuron basic Clam, but with an extra i (and subsequent q) and extra z, so 12 neurons total")

## Topology
    # Number and shape of neurons

# Channels of data
b4Clam.I_channels = 1     
b4Clam.Q_channels = 1     
b4Clam.Z_channels = 1     
b4Clam.C_channels = 1     

# Creating registery of WNNs for each layer
b4Clam.I = np.arange(b4Clam.I_channels, dtype='O')
b4Clam.Q = np.copy(b4Clam.I)     # Q is by default a mirror copy of I
b4Clam.Z = np.arange(b4Clam.Z_channels, dtype='O')
b4Clam.C = np.arange(b4Clam.C_channels, dtype='O')

# Populating the registries with unique identifiers for the WNNs

## number of neurons
b4Clam.i = (4,)     # to match MNIST (28x28 pixels), black and white
b4Clam.q = (4,)     # Q mirrors I
b4Clam.z = (2,)         # 4 are used since 4 binary digits are necesary to encode 0-9 int
b4Clam.c = (2,)         # 1 control neuron to identify training vs. inference states
b4Clam.n_total = sum(b4Clam.i) + sum(b4Clam.q) + sum(b4Clam.z) + sum(b4Clam.c)

# nids of neurons
b4Clam.I[0] = np.arange(4)
b4Clam.Q[0] = np.arange(4, 8)
b4Clam.Z[0] = np.arange(8, 10)
b4Clam.C[0] = np.arange(10, 12)
b4Clam.IQZC = np.arange(b4Clam.n_total)
b4Clam.IQZ = np.arange(b4Clam.n_total - 2)

b4Clam.I__flat = np.concatenate(b4Clam.I, axis=None).ravel() 
b4Clam.Q__flat = np.concatenate(b4Clam.Q, axis=None).ravel() 
b4Clam.Z__flat = np.concatenate(b4Clam.Z, axis=None).ravel() 
b4Clam.C__flat = np.concatenate(b4Clam.C, axis=None).ravel() 
b4Clam.QZ__flat = np.concatenate((b4Clam.Q__flat, b4Clam.Z__flat))


# Defining Neuron metadata
b4Clam.datamatrix = np.zeros([5, b4Clam.n_total], dtype="O")
# 5 rows, as follows:
    #0 Type
    #1 Input Connections; during iconic learning / formation of neuron lookup table (tsets), input connections are taken from the current state
    #2 Neighbor Connections; in constrast to input connections, these take from the past state
    #3 C Connections; neurons connections to C
    #4 Dominant Connection; 1 connection only; a CGA type neuron copies its dominant connection 
    #    ** note; the dominant connection is critical; it is why Q is made in the shape/size of I, so that each Q has a corresponding I as dominant connection (the dominant connection for Z is its own past state [-1]; since if the NSM did something "good / triggered C(s) pleasure neuron(s)" during iconic training, Q will be dominated by I and Z by its past Z (the training becomes; given C at state s, store I(s) and Z(s-1) since Z(s-1) led to the I(s) which triggered C(s)

b4Clam.datamatrix[0, b4Clam.I__flat] = "Input"
b4Clam.datamatrix[0, b4Clam.Q__flat] = "CGA"
b4Clam.datamatrix[0, b4Clam.Z__flat] = "CGA"
b4Clam.datamatrix[0, b4Clam.C__flat] = "Control"

b4Clam.datamatrix[1, b4Clam.Q[0][0]] = b4Clam.I__flat
b4Clam.datamatrix[1, b4Clam.Q[0][1]] = b4Clam.I__flat
b4Clam.datamatrix[1, b4Clam.Q[0][2]] = b4Clam.I__flat
b4Clam.datamatrix[1, b4Clam.Q[0][3]] = b4Clam.I__flat

b4Clam.datamatrix[1, b4Clam.Z[0][0]] = b4Clam.Q__flat
b4Clam.datamatrix[1, b4Clam.Z[0][1]] = b4Clam.Q__flat


b4Clam.datamatrix[2, b4Clam.Q[0][0]] = b4Clam.Q__flat
b4Clam.datamatrix[2, b4Clam.Q[0][1]] = b4Clam.Q__flat
b4Clam.datamatrix[2, b4Clam.Q[0][2]] = b4Clam.Q__flat
b4Clam.datamatrix[2, b4Clam.Q[0][3]] = b4Clam.Q__flat

b4Clam.datamatrix[2, b4Clam.Z[0][0]] = b4Clam.Z__flat
b4Clam.datamatrix[2, b4Clam.Z[0][1]] = b4Clam.Z__flat


b4Clam.datamatrix[3, b4Clam.QZ__flat] = [b4Clam.C__flat]
b4Clam.datamatrix[3, b4Clam.C[0][0]] = b4Clam.QZ__flat
b4Clam.datamatrix[3, b4Clam.C[0][1]] = b4Clam.QZ__flat

b4Clam.datamatrix[4, b4Clam.Q__flat] = b4Clam.I__flat
b4Clam.datamatrix[4, b4Clam.Z__flat] = b4Clam.Z__flat
b4Clam.datamatrix[4, b4Clam.C[0][0]] = "Default if label"


def c0_instinct_rule(INPUT, Agent):
    "b4 clam help!!"

    # if F == 1, C = 1
    
    if INPUT[0] == 1    and    sum(Agent.story[ Agent.state-1,  Agent.arch.Z__flat] == [1, 1]) >= 1:
        
        instinct_response = [1, "F/I[0]=1, food!"]
    
    else:
        instinct_response = [0, ""]
    
    return instinct_response
            
b4Clam.datamatrix[4, b4Clam.C[0][1]] = c0_instinct_rule



#%%


multiClam = ao.Arch("6 INPUT neurons (2 channels), 2 OUTPUTs")


## Topology
    # Number and shape of neurons

# Channels of data
multiClam.I_channels = 2     
multiClam.Q_channels = 2     
multiClam.Z_channels = 1
multiClam.C_channels = 1     

# Creating registery of WNNs for each layer
multiClam.I = np.arange(multiClam.I_channels, dtype='O')
multiClam.Q = np.copy(multiClam.I)     # Q is by default a mirror copy of I
multiClam.Z = np.arange(multiClam.Z_channels, dtype='O')
multiClam.C = np.arange(multiClam.C_channels, dtype='O')

# Populating the registries with unique identifiers for the WNNs

## number of neurons
multiClam.i = (4,2)     # to match MNIST (28x28 pixels), black and white
multiClam.q = (4,2)     # Q mirrors I
multiClam.z = (2,)         # 4 are used since 4 binary digits are necesary to encode 0-9 int
multiClam.c = (2,)         # 1 control neuron to identify training vs. inference states
multiClam.n_total = sum(multiClam.i) + sum(multiClam.q) + sum(multiClam.z) + sum(multiClam.c)

# nids of neurons
multiClam.I[0] = np.arange(4)
multiClam.I[1] = np.arange(4, 6)
multiClam.Q[0] = np.arange(6, 10)
multiClam.Q[1] = np.arange(10, 12)

multiClam.Z[0] = np.arange(12, 14)
multiClam.C[0] = np.arange(14, 16)

multiClam.IQZC = np.arange(multiClam.n_total)
multiClam.IQZ = np.arange(multiClam.n_total - sum(multiClam.c))

multiClam.I__flat = np.concatenate(multiClam.I, axis=None).ravel() 
multiClam.Q__flat = np.concatenate(multiClam.Q, axis=None).ravel() 
multiClam.Z__flat = np.concatenate(multiClam.Z, axis=None).ravel() 
multiClam.C__flat = np.concatenate(multiClam.C, axis=None).ravel() 
multiClam.QZ__flat = np.concatenate((multiClam.Q__flat, multiClam.Z__flat))


# Defining Neuron metadata
multiClam.datamatrix = np.zeros([5, multiClam.n_total], dtype="O")
# 5 rows, as follows:
    #0 Type
    #1 Input Connections; during iconic learning / formation of neuron lookup table (tsets), input connections are taken from the current state
    #2 Neighbor Connections; in constrast to input connections, these take from the past state
    #3 C Connections; neurons connections to C
    #4 Dominant Connection; 1 connection only; a CGA type neuron copies its dominant connection 
    #    ** note; the dominant connection is critical; it is why Q is made in the shape/size of I, so that each Q has a corresponding I as dominant connection (the dominant connection for Z is its own past state [-1]; since if the NSM did something "good / triggered C(s) pleasure neuron(s)" during iconic training, Q will be dominated by I and Z by its past Z (the training becomes; given C at state s, store I(s) and Z(s-1) since Z(s-1) led to the I(s) which triggered C(s)

multiClam.datamatrix[0, multiClam.I__flat] = "Input"
multiClam.datamatrix[0, multiClam.Q__flat] = "CGA"
multiClam.datamatrix[0, multiClam.Z__flat] = "CGA"
multiClam.datamatrix[0, multiClam.C__flat] = "Control"

multiClam.datamatrix[1, multiClam.Q[0][0]] = multiClam.I__flat
multiClam.datamatrix[1, multiClam.Q[0][1]] = multiClam.I__flat
multiClam.datamatrix[1, multiClam.Q[0][2]] = multiClam.I__flat
multiClam.datamatrix[1, multiClam.Q[0][3]] = multiClam.I__flat

multiClam.datamatrix[1, multiClam.Q[1][0]] = multiClam.I__flat
multiClam.datamatrix[1, multiClam.Q[1][1]] = multiClam.I__flat

multiClam.datamatrix[1, multiClam.Z[0][0]] = multiClam.Q__flat
multiClam.datamatrix[1, multiClam.Z[0][1]] = multiClam.Q__flat


multiClam.datamatrix[2, multiClam.Q[0][0]] = multiClam.Q__flat
multiClam.datamatrix[2, multiClam.Q[0][1]] = multiClam.Q__flat
multiClam.datamatrix[2, multiClam.Q[0][2]] = multiClam.Q__flat
multiClam.datamatrix[2, multiClam.Q[0][3]] = multiClam.Q__flat

multiClam.datamatrix[2, multiClam.Q[1][0]] = multiClam.Q__flat
multiClam.datamatrix[2, multiClam.Q[1][1]] = multiClam.Q__flat

multiClam.datamatrix[2, multiClam.Z[0][0]] = multiClam.Z__flat
multiClam.datamatrix[2, multiClam.Z[0][1]] = multiClam.Z__flat


multiClam.datamatrix[3, multiClam.QZ__flat] = [multiClam.C__flat]
multiClam.datamatrix[3, multiClam.C[0][0]] = multiClam.QZ__flat
multiClam.datamatrix[3, multiClam.C[0][1]] = multiClam.QZ__flat

multiClam.datamatrix[4, multiClam.Q__flat] = multiClam.I__flat
multiClam.datamatrix[4, multiClam.Z__flat] = multiClam.Z__flat
multiClam.datamatrix[4, multiClam.C[0][0]] = "Default if label"


def c0_instinct_rule(INPUT, Agent):

    # if F == 1, C = 1
    
    if INPUT[0] == 1    and    sum(Agent.story[ Agent.state-1,  Agent.arch.Z__flat] == [1, 1]) >= 1:
        
        instinct_response = [1, "F/I[0]=1, food!"]
    
    else:
        instinct_response = [0, ""]
    
    return instinct_response
            
multiClam.datamatrix[4, multiClam.C[0][1]] = c0_instinct_rule





#%%



multiFoodClam = ao.Arch("8 neuron input (6 and 3, 2nd channel is for F1 F2 F3)")


## Topology
    # Number and shape of neurons

# Channels of data
multiFoodClam.I_channels = 2     
multiFoodClam.Q_channels = 2     
multiFoodClam.Z_channels = 1     
multiFoodClam.C_channels = 1     

# Creating registery of WNNs for each layer
multiFoodClam.I = np.arange(multiFoodClam.I_channels, dtype='O')
multiFoodClam.Q = np.copy(multiFoodClam.I)     # Q is by default a mirror copy of I
multiFoodClam.Z = np.arange(multiFoodClam.Z_channels, dtype='O')
multiFoodClam.C = np.arange(multiFoodClam.C_channels, dtype='O')

# Populating the registries with unique identifiers for the WNNs

## number of neurons
multiFoodClam.i = (8,3)     # to match MNIST (28x28 pixels), black and white
multiFoodClam.q = (8,3)     # Q mirrors I
multiFoodClam.z = (2,)         # 4 are used since 4 binary digits are necesary to encode 0-9 int
multiFoodClam.c = (2,)         # 1 control neuron to identify training vs. inference states
multiFoodClam.n_total = sum(multiFoodClam.i) + sum(multiFoodClam.q) + sum(multiFoodClam.z) + sum(multiFoodClam.c)

# nids of neurons
multiFoodClam.I[0] = np.arange(8)
multiFoodClam.I[1] = np.arange(8, 11)
multiFoodClam.Q[0] = np.arange(11, 19)
multiFoodClam.Q[1] = np.arange(19, 22)
multiFoodClam.Z[0] = np.arange(22, 24)
multiFoodClam.C[0] = np.arange(24, 26)

multiFoodClam.IQZC = np.arange(multiFoodClam.n_total)
multiFoodClam.IQZ = np.arange(multiFoodClam.n_total - sum(multiFoodClam.c))

multiFoodClam.I__flat = np.concatenate(multiFoodClam.I, axis=None).ravel() 
multiFoodClam.Q__flat = np.concatenate(multiFoodClam.Q, axis=None).ravel() 
multiFoodClam.Z__flat = np.concatenate(multiFoodClam.Z, axis=None).ravel() 
multiFoodClam.C__flat = np.concatenate(multiFoodClam.C, axis=None).ravel() 
multiFoodClam.QZ__flat = np.concatenate((multiFoodClam.Q__flat, multiFoodClam.Z__flat))


# Defining Neuron metadata
multiFoodClam.datamatrix = np.zeros([5, multiFoodClam.n_total], dtype="O")
# 5 rows, as follows:
    #0 Type
    #1 Input Connections; during iconic learning / formation of neuron lookup table (tsets), input connections are taken from the current state
    #2 Neighbor Connections; in constrast to input connections, these take from the past state
    #3 C Connections; neurons connections to C
    #4 Dominant Connection; 1 connection only; a CGA type neuron copies its dominant connection 
    #    ** note; the dominant connection is critical; it is why Q is made in the shape/size of I, so that each Q has a corresponding I as dominant connection (the dominant connection for Z is its own past state [-1]; since if the NSM did something "good / triggered C(s) pleasure neuron(s)" during iconic training, Q will be dominated by I and Z by its past Z (the training becomes; given C at state s, store I(s) and Z(s-1) since Z(s-1) led to the I(s) which triggered C(s)

multiFoodClam.datamatrix[0, multiFoodClam.I__flat] = "Input"
multiFoodClam.datamatrix[0, multiFoodClam.Q__flat] = "CGA"
multiFoodClam.datamatrix[0, multiFoodClam.Z__flat] = "CGA"
multiFoodClam.datamatrix[0, multiFoodClam.C__flat] = "Control"

multiFoodClam.datamatrix[1, multiFoodClam.Q[0][0]] = multiFoodClam.I__flat
multiFoodClam.datamatrix[1, multiFoodClam.Q[0][1]] = multiFoodClam.I__flat
multiFoodClam.datamatrix[1, multiFoodClam.Q[0][2]] = multiFoodClam.I__flat
multiFoodClam.datamatrix[1, multiFoodClam.Q[0][3]] = multiFoodClam.I__flat
multiFoodClam.datamatrix[1, multiFoodClam.Q[0][4]] = multiFoodClam.I__flat
multiFoodClam.datamatrix[1, multiFoodClam.Q[0][5]] = multiFoodClam.I__flat
multiFoodClam.datamatrix[1, multiFoodClam.Q[0][6]] = multiFoodClam.I__flat
multiFoodClam.datamatrix[1, multiFoodClam.Q[0][7]] = multiFoodClam.I__flat

multiFoodClam.datamatrix[1, multiFoodClam.Q[1][0]] = multiFoodClam.I[1]
multiFoodClam.datamatrix[1, multiFoodClam.Q[1][1]] = multiFoodClam.I[1]
multiFoodClam.datamatrix[1, multiFoodClam.Q[1][2]] = multiFoodClam.I[1]

multiFoodClam.datamatrix[1, multiFoodClam.Z[0][0]] = multiFoodClam.Q__flat
multiFoodClam.datamatrix[1, multiFoodClam.Z[0][1]] = multiFoodClam.Q__flat


multiFoodClam.datamatrix[2, multiFoodClam.Q[0][0]] = multiFoodClam.Q__flat
multiFoodClam.datamatrix[2, multiFoodClam.Q[0][1]] = multiFoodClam.Q__flat
multiFoodClam.datamatrix[2, multiFoodClam.Q[0][2]] = multiFoodClam.Q__flat
multiFoodClam.datamatrix[2, multiFoodClam.Q[0][3]] = multiFoodClam.Q__flat
multiFoodClam.datamatrix[2, multiFoodClam.Q[0][4]] = multiFoodClam.Q__flat
multiFoodClam.datamatrix[2, multiFoodClam.Q[0][5]] = multiFoodClam.Q__flat
multiFoodClam.datamatrix[2, multiFoodClam.Q[0][6]] = multiFoodClam.Q__flat
multiFoodClam.datamatrix[2, multiFoodClam.Q[0][7]] = multiFoodClam.Q__flat

multiFoodClam.datamatrix[2, multiFoodClam.Q[1][0]] = multiFoodClam.Q[1]
multiFoodClam.datamatrix[2, multiFoodClam.Q[1][1]] = multiFoodClam.Q[1]
multiFoodClam.datamatrix[2, multiFoodClam.Q[1][2]] = multiFoodClam.Q[1]

multiFoodClam.datamatrix[2, multiFoodClam.Z[0][0]] = multiFoodClam.Z__flat
multiFoodClam.datamatrix[2, multiFoodClam.Z[0][1]] = multiFoodClam.Z__flat


multiFoodClam.datamatrix[3, multiFoodClam.QZ__flat] = [multiFoodClam.C__flat]
multiFoodClam.datamatrix[3, multiFoodClam.C[0][0]] = multiFoodClam.QZ__flat
multiFoodClam.datamatrix[3, multiFoodClam.C[0][1]] = multiFoodClam.QZ__flat

multiFoodClam.datamatrix[4, multiFoodClam.Q__flat] = multiFoodClam.I__flat
multiFoodClam.datamatrix[4, multiFoodClam.Z__flat] = multiFoodClam.Z__flat
multiFoodClam.datamatrix[4, multiFoodClam.C[0][0]] = "Default if label"


def c0_instinct_rule(INPUT, Agent):

    # if F == 1, C = 1
    
    if sum(INPUT[ Agent.arch.I[1] ]) >= 1    and    sum(Agent.story[ Agent.state-1,  Agent.arch.Z__flat]) >= 1:
        
        instinct_response = [1, "F/I[0]=1, food!"]
    
    else:
        instinct_response = [0, ""]
    
    return instinct_response
            
multiFoodClam.datamatrix[4, multiFoodClam.C[0][1]] = c0_instinct_rule

