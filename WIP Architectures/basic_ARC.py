# -*- coding: utf-8 -*-
"""
// aolabs.ai software >ao_core/Arch.py (C) 2023 Animo Omnis Corporation. All Rights Reserved.

Thank you for your curiosity!
"""


## // ARC Agent -- Work-In-Progress
# 
# A 1D application of an Agent trained on small samples of MNIST: https://en.wikipedia.org/wiki/MNIST_database
# 
# We've gotten Agents with between 20-60%+ accuracy having trained on only 120 training examples (instead of the full 60,000),
# the variance depends on how their neurons are connected (fully connecting an Agent with neurons of this mangnitude was prohbitive for our small machines 
# and unoptimized code, hence the "rand_conn" connector function)
#
# For interactive visual representation of this Arch:
#    https://miro.com/app/board/uXjVM_kESvI=/?share_link_id=72701488535
#
# Cutomize and upload this Arch to our API to create Agents: https://docs.aolabs.ai/reference/kennelcreate
#


description = "3600-neuron Agent for ARC benchmarking"      #    MNIST is in grayscale, which we downscaled to B&W for the simple 28x28 neuron count -- 788 = 28x28 + 4
arch_i = [30*30 * 4]               # note that the 784 I neurons are in 1 input channel; MNIST is like a single channel clam, so it's limitations are obvious from the prespective of our approach, more on this here: 
arch_z = [30*30 * 4]                   # 4 neurons in 1 channel as 4 binary digits encodes up to integer 16, and only 10 (0-9) are needed for MNIST
arch_c = []
connector_function = "rand_conn"
connector_parameters = [1200, 1200, 1200, 1200]
# so all Q neurons are connected randomly to 1200 I and 1200 neighbor Q
# and all Z neurons are connected randomly to 784 Q and 4 (or all) neighbor Z

# To maintain compatability with our API, do not change the variable name "Arch" or the constructor class "ao.Arch" in the line below (the API is pre-loaded with a version of the Arch class in this repo's main branch, hence "ao.Arch")
Arch = ao.Arch(arch_i, arch_z, arch_c, connector_function, connector_parameters, description)

# varying these architecture details has been explored in this WIP aolabs research paper: https://docs.google.com/document/d/1p3FvYYPsD9XunJg2Dfaw0wLvnxIUspPsMvBi8acp0EI/edit