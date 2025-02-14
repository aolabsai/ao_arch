# -*- coding: utf-8 -*-
"""
// aolabs.ai software >ao_core/Arch.py (C) 2023 Animo Omnis Corporation. All Rights Reserved.

Thank you for your curiosity!
"""


## // Basic Clam -- Reference Design #0
# 
# Our simplest Agent, our 'hello, world.'
#
# For interactive visual representation of this Arch:
#    https://miro.com/app/board/uXjVM_kESvI=/?share_link_id=72701488535
#
# Customize and upload this Arch to our API to create Agents: https://docs.aolabs.ai/reference/kennelcreate
#


description = "Basic Clam"
arch_i = [1, 1, 1]     # 3 neurons, 1 in each of 3 channels, corresponding to Food, Chemical-A, Chemical-B (present=1/not=0)
arch_z = [1]           # corresponding to Open=1/Close=0
arch_c = [1]           # adding 1 control neuron which we'll define with the instinct control function below
connector_function = "full_conn"
connector_parameters=()

arch_qa = [10]
qa_conn = "full"

# To maintain compatibility with our API, do not change the variable name "Arch" or the constructor class "ar.Arch" in the line below
Arch = ar.Arch(arch_i, arch_z, arch_c, connector_function, connector_parameters, arch_qa, qa_conn, description)

# Adding Instinct Control Neuron
def c0_instinct_rule(INPUT, Agent):
    if INPUT[0] == 1    and    Agent.story[ Agent.state-1,  Agent.arch.Z__flat[0]] == 1 :        # self.Z__flat[0] needs to be adjusted as per the agent, which output the designer wants the agent to repeat while learning postively or negatively
        instinct_response = [1, "c0 instinct triggered"]
    else:
        instinct_response = [0, "c0 pass"]    
    return instinct_response            
# Saving the function to the Arch so the Agent can access it
Arch.datamatrix[4, Arch.C[1][0]] = c0_instinct_rule


# Adding Aux Action (as a counter)
def qa0_firing_rule(Agent):
    if not hasattr(Agent, 'qa0_counter'):
        Agent.__setattr__("qa0_counter", 0)

    if Agent.qa0_counter < 11:
        group_response = np.zeros(10)
        group_response[0 : Agent.qa0_counter] = 1
        print("Qa0 counter:: "+group_response)
        Agent.qa0_counter += 1
    else:
        Agent.qa0_counter = 1
        group_response = np.zeros(10)
        print("Qa0 counter reset at:: " + str(Agent.state))                

    group_meta = np.ones(10, dtype="O")
    group_meta[:] = "qa0"
    return group_response, group_meta
# Saving the function to the Arch so the Agent can access it
Arch.datamatrix_aux[2] = qa0_firing_rule