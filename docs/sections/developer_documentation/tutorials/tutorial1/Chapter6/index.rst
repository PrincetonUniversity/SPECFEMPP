
.. _Chapter6:

Chapter 6: SPECFEM++ iteration policies
========================================

In :ref:`Chapter 5 <Chapter5>`, we discussed data-types in SPECFEM++ that essentially depend on the heirarchy of the mesh. In this chapter, we will introduce the iteration policies within SPECFEM++ that enable us to iterate over the mesh heirarchy in a consistent manner.

There are three iteration policies in SPECFEM++:



.. toctree::
   :maxdepth: 1
   :glob:

   Iterate over a range of quadrature point where every distinct quadrature point within the assembly is visited exactly once. <range_policy>
   Iterate over a chunks of elements where every quadrature point within each chunk is visited exactly once. <chunk_element_policy>
   Iterate over a range of edges where every quadrature point with each edge is visited exactly once. <edge_policy>
