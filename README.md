# Welcome to  AO Labs

Thank you for being here!

To install the latest version of ao_arch locally, run `pip install git+https://github.com/aolabsai/ao_arch` in your terminal.

We're building the next layer in the AI stack, AI continuously trained to maintain itself as a referent in the learned local context, for awareness as code. We believe this is what's missing from our conception of AI today, the lack of which leads to hallucinations and the blackbox problem.

At these earliest of stages our AI is at a very basic level of self-association, comparable to simplest of animals. It's not ChadGPT or anywhere near human levels, but it has qualities that LLMs and other status quo AI lack, and as a community we have a clear bottom-up path to building more cognitively complicated agents following the evolution and phylogenetics of intelligence.

To get started, you can fork a reference design (the numbered files in the *archs repo* root directory) to make a new Arch for your application and use it with our Agents-as-a-Service API to create and use persistent Agents, [here’s a guide](https://docs.aolabs.ai/docs/arch).

This *archs repo* contains our first agent configurations and their applications, all open-source:
- [Arch constructor class](https://github.com/aolabsai/archs/blob/main/Arch.py)
- Arch reference designs
	- [0- Basic Clam](0_basic_clam.py) -  our "Hello, World," a Clam-level AGI and simplest reference design     [[try demo](https://aolabs.streamlit.app/)] [[view code](https://github.com/aolabsai/archs/tree/main/Applications/HelloWorld-BasicClam)]
- [Applications](https://github.com/aolabsai/archs/tree/main/Applications)
	- [1 - Basic MNIST](1_basic_MNIST.py) - a 1D application of an Agent trained on small samples of MNIST: https://en.wikipedia.org/wiki/MNIST_database     [demo app coming soon]
	- [2 - NetBox Device-Discovery](2_netbox-device_discovery.py) - Netbox- Device Discovery-- an Agent as a context-aware autocomplete for relational data, applied to Netbox for network automation     [[try demo](https://aolabs-netbox.streamlit.app/)] [[view code](https://github.com/aolabsai/archs/tree/main/Applications/Netbox/Device_Discovery)]
- [WIP Applications](https://github.com/aolabsai/archs/tree/main/WIP%20Architectures)
- [API OAS 3 Definition](https://github.com/aolabsai/archs/blob/main/core_api.yaml)-- details of the Agents-as-a-Service API running on AWS provided as a playground and Agent hosting/deployment service

### More resources:
- View a [visual vepresentation of archs and applications](https://miro.com/app/board/uXjVM_kESvI=/?share_link_id=686677372269) on Miro
- Our docs, API reference, and research background on [docs.aolabs.ai](https://docs.aolabs.ai)
- Say hi and ask questions on our [discord](https://discord.gg/FTQgAgzZT7)

---
## RE Licensing: Open Source & Our API

Everything in the **archs** repo is open-source-- the **Arch** class, particular Agents' archs, and associated applications. We hope this inspires more use-cases as we're fresh from the lab.

We plan to open-source the rest of our code progressively over time, that is, the **Agent** class that constructs an Agent given an Arch, and this is an evolving discussion as we're careful to ensure we make the right decisions now to realize the long-term potential we've seen with this code. We had started this journey as researchers after all, so there are years of context to be shared.

For now, our Agents-as-a-Service API is our **Core**'s **Agent** and **Arch** classes running on AWS, as a playground to spin up and test Agents without having to install code, so that you also don't have to worry about hosting multiple unique Agents for your users. However, our Agents are designed to be run and trained directly at the edge.
- If you are interested in running Agents locally and need access to Core now, please reach out, we can accommodate you and we need more use-cases to learn from. 
- If you are interested in joining us as a Core Contributor, please email ali@aolabs.ai, [book a meeting](https://calendly.com/aee/meeting), or say hi on our [Discord server](https://discord.gg/FTQgAgzZT7).

Thank you very much for being here. We welcome contributors of all levels, on the application layer or with our **Core** work.
