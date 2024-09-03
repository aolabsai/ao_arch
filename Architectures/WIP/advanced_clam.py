# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 00:37:29 2022

@author: alebr
"""





## AO Labs Modules
import ao_core as ao

# 3rd Party Modules
import numpy as np


#%%

aClam_note = "WIP Advanced clam architecture; a reference design closer to a real clam (highly multi-modal, multi-sensory, etc.)"
aClam = ao.Arch(aClam_note)




## Topology
    # Number and shape of neurons

# Channels of data
aClam.I_channels = 5     # there is only 1 input type in MNIST if taken as B&W
aClam.Q_channels = 5     # by default, Q is a mirror copy of I
aClam.Z_channels = 1     # classes of outputs; again only 1 type for MNIST
aClam.C_channels = 2     # classes of outputs; again only 1 type for MNIST

# Creating registery of WNNs for each layer
aClam.I = np.arange(aClam.I_channels, dtype='O')
aClam.Q = np.copy(aClam.I)     # Q is by default a mirror copy of I
aClam.Z = np.arange(aClam.Z_channels, dtype='O')
aClam.C = np.arange(aClam.C_channels, dtype='O')     # classes of outputs; again only 1 type for MNIST

# Populating the registrys with unique identifiers for the WNNs
aClam.i = (5*5,   10,   5,   3,   3)     # to match MNIST (28x28 pixels), black and white
aClam.q = (5*5,   10,   5,   3,   3)     # Q mirrors I
aClam.z = (4,)         # 4 are used since 4 binary digits are necesary to encode 0-9 int
aClam.c = (1,1)         # 1 control neuron to identify training vs. inference states
aClam.n_total = sum(aClam.i) + sum(aClam.q) + sum(aClam.z) + sum(aClam.c)

aClam.I[0] = list(range(aClam.i[0]))
aClam.I[1] = list(range(aClam.i[0], aClam.i[0]+aClam.i[1]))
aClam.I[2] = list(range(aClam.i[0]+aClam.i[1], aClam.i[0]+aClam.i[1]+aClam.i[2]))
aClam.I[3] = list(range(aClam.i[0]+aClam.i[1]+aClam.i[2], aClam.i[0]+aClam.i[1]+aClam.i[2]+aClam.i[3]))
aClam.I[4] = list(range(aClam.i[0]+aClam.i[1]+aClam.i[2]+aClam.i[3], aClam.i[0]+aClam.i[1]+aClam.i[2]+aClam.i[3]+aClam.i[4]))
aClam.ie = sum(aClam.i)

aClam.Q[0] = list(range(aClam.ie, aClam.ie+aClam.q[0]))
aClam.Q[1] = list(range(aClam.ie+aClam.q[0], aClam.ie+aClam.q[0]+aClam.q[1]))
aClam.Q[2] = list(range(aClam.ie+aClam.q[0]+aClam.q[1], aClam.ie+aClam.q[0]+aClam.q[1]+aClam.q[2]))
aClam.Q[3] = list(range(aClam.ie+aClam.q[0]+aClam.q[1]+aClam.q[2], aClam.ie+aClam.q[0]+aClam.q[1]+aClam.q[2]+aClam.q[3]))
aClam.Q[4] = list(range(aClam.ie+aClam.q[0]+aClam.q[1]+aClam.q[2]+aClam.q[3], aClam.ie+aClam.q[0]+aClam.q[1]+aClam.q[2]+aClam.q[3]+aClam.q[4]))
aClam.qe = sum(aClam.i) + sum(aClam.q)

aClam.Z[0] = list(range(aClam.qe, aClam.qe+aClam.z[0]))
aClam.ze = sum(aClam.i) + sum(aClam.q) + sum(aClam.z)

aClam.C[0] = list(range(aClam.ze, aClam.ze+aClam.c[0]))
aClam.C[1] = list(range(aClam.ze+aClam.c[0], aClam.ze+aClam.c[0]+aClam.c[1]))

aClam.I__flat = np.concatenate(aClam.I, axis=None).ravel()  # is .ravel even needed?
aClam.Q__flat = np.concatenate(aClam.Q, axis=None).ravel()
aClam.Z__flat = np.concatenate(aClam.Z, axis=None).ravel()
aClam.C__flat = np.concatenate(aClam.C, axis=None).ravel()
aClam.QZ__flat = np.concatenate((aClam.Q__flat, aClam.Z__flat))

# Defining Neuron metadata
aClam.metaIQZC = np.zeros([5, aClam.n_total], dtype="O")
# 5 rows, as follows:
    #0 Type
    #1 Input Connections
    #2 Neighbor Connections
    #3 C Connections
    #4 Dominant Connection

# Only setting types for now (other neuron meta data will be populated by 
    # the function this_module.rand_conn below)
aClam.metaIQZC[0, aClam.I__flat] = "Input"
aClam.metaIQZC[0, aClam.Q__flat] = "CGA"
aClam.metaIQZC[0, aClam.Z__flat] = "CGA"
aClam.metaIQZC[0, aClam.C__flat] = "Control"




aClam.metaIQZC[1, aClam.Q[0]] = aClam.I[0]
aClam.metaIQZC[1, aClam.Q[1]] = aClam.I[1]
aClam.metaIQZC[1, aClam.Q[2]] = aClam.I[2]
aClam.metaIQZC[1, aClam.Q[3]] = aClam.I[3]
aClam.metaIQZC[1, aClam.Q[4]] = aClam.I[4]

aClam.metaIQZC[2, aClam.Q[0]] = aClam.I[0] + aClam.Q__flat
aClam.metaIQZC[2, aClam.Q[1]] = aClam.I[1] + aClam.Q__flat
aClam.metaIQZC[2, aClam.Q[2]] = aClam.I[2] + aClam.Q__flat
aClam.metaIQZC[2, aClam.Q[3]] = aClam.I[3] + aClam.Q__flat
aClam.metaIQZC[2, aClam.Q[4]] = aClam.I[4] + aClam.Q__flat





aClam.metaIQZC[0, aClam.Z[0]] = "CGA"
aClam.metaIQZC[0, aClam.C[0]] = "Control"




def rand_conn(i_conn, q_conn):
            
    for q in aClam.Q__flat:
        aClam.metaIQZC[1, q] = sorted(np.random.randint(0, aClam.i, i_conn))
        aClam.metaIQZC[2, q] = sorted(np.random.randint(aClam.i, aClam.i + aClam.q, q_conn))
        aClam.metaIQZC[3, q] = sorted(aClam.C__flat)
        aClam.metaIQZC[4, q] = q - aClam.q
        
    z0 = 0
    for z in aClam.Z__flat:
        aClam.metaIQZC[1, z] = sorted(aClam.Q__flat)
        aClam.metaIQZC[2, z] = sorted(aClam.Z__flat)
        aClam.metaIQZC[3, z] = sorted(aClam.C__flat)
        aClam.metaIQZC[4, z] = 0    # should fix
        z0 += 1
    
    for c in aClam.C__flat:
        aClam.metaIQZC[3, c] = sorted(aClam.Q__flat) + sorted(aClam.Z__flat)
        


aClam.rand_conn = rand_conn





