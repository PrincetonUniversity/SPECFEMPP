
.. _Chapter12:

Chapter 12: Runtime Configuration
=================================

In the last chapter, we implemented the ``main`` function that drives the simulation. However, the entire simulation is driven by compile-time struct ``simulation_params``. While this works for this tutorial example, this is not ideal for several reasons:

1. It requires recompilation of the code for every change in the simulation parameters,
2. Simulation path essentially becomes hardcoded in the code
3. It is not possible to run multiple simulations with different parameters in parallel

To overcome these limitations, SPECFEM++ provides a runtime configuration system, which allows the user to specify the simulation parameters in a YAML configuration file. The configuration file then determines the simulation path at runtime.

For more information on the runtime configuration system, refer to the :ref:`runtime configuration <runtime_configuration>`.
