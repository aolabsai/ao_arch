# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 00:37:29 2022

@author: alebr
"""





## AO Labs Modules
import ao_core as ao

# 3rd Party Modules
import numpy as np



# no longer compatiable with core as it uses only 2 C neurons (label and instinct)

#%%


bClam = ao.Arch("Basic calm architecture, as a general case of MNIST v0.1.0,\
 with 2 c neurons (the default-if-LABEL and 1 pleasure neuron)!")


## Topology
    # Number and shape of neurons

# Channels of data
bClam.I_channels = 1     
bClam.Q_channels = 1     
bClam.Z_channels = 1     
bClam.C_channels = 1     

# Creating registery of WNNs for each layer
bClam.I = np.arange(bClam.I_channels, dtype='O')
bClam.Q = np.copy(bClam.I)     # Q is by default a mirror copy of I
bClam.Z = np.arange(bClam.Z_channels, dtype='O')
bClam.C = np.arange(bClam.C_channels, dtype='O')

# Populating the registries with unique identifiers for the WNNs

## number of neurons
bClam.i = (3,)     # to match MNIST (28x28 pixels), black and white
bClam.q = (3,)     # Q mirrors I
bClam.z = (1,)         # 4 are used since 4 binary digits are necesary to encode 0-9 int
bClam.c = (2,)         # 1 control neuron to identify training vs. inference states
bClam.n_total = sum(bClam.i) + sum(bClam.q) + sum(bClam.z) + sum(bClam.c)

# nids of neurons
bClam.I[0] = np.arange(3)
bClam.Q[0] = np.arange(3, 6)
bClam.Z[0] = np.arange(6, 7)
bClam.C[0] = np.arange(7, 9)
bClam.IQZC = np.arange(bClam.n_total)
bClam.IQZ = np.arange(bClam.n_total - 2)

bClam.I__flat = np.concatenate(bClam.I, axis=None).ravel() 
bClam.Q__flat = np.concatenate(bClam.Q, axis=None).ravel() 
bClam.Z__flat = np.concatenate(bClam.Z, axis=None).ravel() 
bClam.C__flat = np.concatenate(bClam.C, axis=None).ravel() 
bClam.QZ__flat = np.concatenate((bClam.Q__flat, bClam.Z__flat))


# Defining Neuron metadata
bClam.datamatrix = np.zeros([5, bClam.n_total], dtype="O")
# 5 rows, as follows:
    #0 Type
    #1 Input Connections; during iconic learning / formation of neuron lookup table (tsets), input connections are taken from the current state
    #2 Neighbor Connections; in constrast to input connections, these take from the past state
    #3 C Connections; neurons connections to C
    #4 Dominant Connection; 1 connection only; a CGA type neuron copies its dominant connection 
    #    ** note; the dominant connection is critical; it is why Q is made in the shape/size of I, so that each Q has a corresponding I as dominant connection (the dominant connection for Z is its own past state [-1]; since if the NSM did something "good / triggered C(s) pleasure neuron(s)" during iconic training, Q will be dominated by I and Z by its past Z (the training becomes; given C at state s, store I(s) and Z(s-1) since Z(s-1) led to the I(s) which triggered C(s)

bClam.datamatrix[0, bClam.I__flat] = "Input"
bClam.datamatrix[0, bClam.Q__flat] = "CGA"
bClam.datamatrix[0, bClam.Z__flat] = "CGA Z"
bClam.datamatrix[0, bClam.C__flat] = "Control"

bClam.datamatrix[1, bClam.Q[0][0]] = bClam.I__flat
bClam.datamatrix[1, bClam.Q[0][1]] = bClam.I__flat
bClam.datamatrix[1, bClam.Q[0][2]] = bClam.I__flat

bClam.datamatrix[1, bClam.Z[0][0]] = bClam.Q__flat


bClam.datamatrix[2, bClam.Q[0][0]] = bClam.Q__flat
bClam.datamatrix[2, bClam.Q[0][1]] = bClam.Q__flat
bClam.datamatrix[2, bClam.Q[0][2]] = bClam.Q__flat

bClam.datamatrix[2, bClam.Z[0]] = [bClam.Z__flat]


bClam.datamatrix[3, bClam.QZ__flat] = [bClam.C__flat]
bClam.datamatrix[3, bClam.C[0][0]] = bClam.QZ__flat
bClam.datamatrix[3, bClam.C[0][1]] = bClam.QZ__flat

bClam.datamatrix[4, bClam.Q__flat] = bClam.I__flat
bClam.datamatrix[4, bClam.Z__flat] = bClam.Z__flat
bClam.datamatrix[4, bClam.C[0][0]] = "Default if label"


def c0_instinct_rule(INPUT, Agent):

    # if F == 1, C = 1
    
    if INPUT[0] == 1    and    Agent.story[ Agent.state-1,  Agent.arch.Z__flat] == 1 :
        
        instinct_response = [1, "F/I[0]=1, food!"]
    
    else:
        instinct_response = [0, ""]
    
    return instinct_response
            
bClam.datamatrix[4, bClam.C[0][1]] = c0_instinct_rule