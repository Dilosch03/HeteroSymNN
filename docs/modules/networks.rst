.. _network-classes:

Network Classes
===============

HeteroSymNN provides a tiered set of network builders designed to bridge the gap between standard deep learning and highly experimental, mathematical research.

This section is structured not just as a dictionary of classes, but as a map of **Architecture Families**. Each family represents a different structural topology, and the classes within them represent your desired level of mathematical control.

Architecture Families
---------------------

Different experimental paradigms require fundamentally different graph structures. The framework categorizes network builders by their structural topology (the "packet" they belong to):

* **Feed-Forward / Dense Family:** The foundational architecture. Data flows in a single direction from input to output. Perfect for standard classification, regression, and baseline physics approximations.
* **Recurrent Family (RNNs) [Planned]:** Temporal topologies with internal state retention, designed for time-series and sequential logic.
* **Evolutionary Family (EvoNets) [Planned]:** Highly dynamic, mutated graph topologies designed specifically for NEAT algorithms and continuous structural growth.

The Feed-Forward Family
-----------------------

Within the Feed-Forward family, the classes are organized by their **level of mathematical abstraction (granularity)**:

* **Level 1: MLP (Multi-Layer Perceptron)** - *The Baseline.*
  When you need to establish a rapid standard baseline. It enforces a homogeneous topology where all hidden layers share the same math (e.g., standard ReLU networks).

* **Level 2: Dense (The "Cocktail" Network)** - *The Macroscopic Step.*
  When your research requires macroscopic heterogeneity. The ``Dense`` class allows you to effortlessly assign different mathematical strings to different layers.

* **Level 3: HeteroDense (Node-Level Customization)** - *The Microscopic Frontier.*
  When you need per node control. It allows you to define a unique mathematical equation and dynamic constants for *every single neuron* in the network.

.. toctree::
    :maxdepth: 1
    :caption: Feed-Forward Architectures:

    nn_clases/mlp
    nn_clases/dense
    nn_clases/heterodense

Core Architecture
-----------------

All of the builders above inherit from a unified base architecture. **Why read this?** You should consult the Base classes if:

* **You are mixing and matching layers:** To manually build highly custom network topologies by snapping together layers from different families (e.g., combining standard Dense layers with future Recurrent or EvoNet layers). Note that while you can mix layers as you please, there are structural restrictions depending on the specific topologies being connected.
* **You are extending the framework:** To build an entirely new "Architecture Family" from scratch.
* **You are debugging at the hardware level:** To understand how the JIT compiler safely handles memory allocation and hardware routing under extreme mathematical complexity.
* **You want to understand the underlying logic:** To explore the core initialization process, internal data routing, and the fundamental methods that power the network's architecture. *(Note: The JIT compiler and the intricate internal mechanics of the individual layers are documented in their own dedicated modules).*

.. toctree::
    :maxdepth: 1
    :caption: Core Architecture:
    
    nn_clases/base