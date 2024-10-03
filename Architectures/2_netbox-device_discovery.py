# -*- coding: utf-8 -*-
"""
// aolabs.ai software >ao_core/Arch.py (C) 2023 Animo Omnis Corporation. All Rights Reserved.

Thank you for your curiosity!
"""


## // Netbox- Device Discovery -- Reference Design #2
# 
# Our first practically useful Agent, a simple 40 neuron Agent (a 10-factor scale-up of our 4-neuron Basic Clam)
# trained on instances of Netbox to predict Roles of new devices given local device configuration of live devices.
# In simpler words, an Agent to answer the question, "what Role is this new device {Router, Patch Panel, Access Port} like to be given its info and the current devices we have?" 
#
# For interactive visual representation of this Arch:
#    https://miro.com/app/board/uXjVM_kESvI=/?share_link_id=72701488535
#
# Customize and upload this Arch to our API to create Agents: https://docs.aolabs.ai/reference/kennelcreate
#


description = "Netbox device type relational autocomplete (10 binary digits per field to encode ids)"
arch_i = [10, 10, 10]               # a scaled up Basic Clam, with 3 input channels with have 10 neurons each, corresponding to device Mfg, Type, and Site (from IDss to 10-digit binary)
arch_z = [10]                       # 10 neurons in 1 channel to encode device Role IDs
arch_c = []
connector_function = "forward_full_conn"
# device mfg, type, and site are stored as strings (names) with associated unique IDs
# using 10 binary neurons to encode integer IDs means we can encode up to 2^10 = 1048 unique binary values.

# To maintain compatibility with our API, do not change the variable name "Arch" or the constructor class "ar.Arch" in the line below
Arch = ar.Arch(arch_i, arch_z, arch_c, connector_function, description)

