# -*- coding: utf-8 -*-
"""
// aolabs.ai software >ao_core/Arch.py (C) 2023 Animo Omnis Corporation. All Rights Reserved.

Thank you for your curiosity!
"""


## // Basic ARC -- Reference Design #3
# 
# The simplist agent we could concieve of for the ARC-AGI benchmark (see archprize.org).
#
# For interactive visual representation of this Arch:
#    https://miro.com/app/board/uXjVM_kESvI=/?share_link_id=72701488535
#
# Customize and upload this Arch to our API to create Agents: https://docs.aolabs.ai/reference/kennelcreate
#


description = "Basic ARC - an agent for the ARC-AGI benchmark"
arch_i = [1, 1, 1,
          1, 1, 1,
          1, 1, 1]
arch_z = [1, 1, 1,
          1, 1, 1,
          1, 1, 1]
arch_c = []           # adding 1 control neuron which we'll define with the instinct control function below
connector_function = "nearest_neighbour_conn"
connector_parameters = [1, 1, 3, 3, False]

# To maintain compatibility with our API, do not change the variable name "Arch" or the constructor class "ar.Arch" in the line below
Arch = ar.Arch(arch_i, arch_z, arch_c, connector_function, connector_parameters, description)