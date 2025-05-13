
.. _Chapter6:

Chapter 6: SPECFEM++ iteration policies
========================================

In :ref:`Chapter 5 <Chapter5>`, we discussed data-types in SPECFEM++ that essentially depend on the heirarchy of the mesh. In this chapter, we will introduce the iteration policies within SPECFEM++ that enable us to iterate over the mesh heirarchy in a consistent manner.

There are three iteration policies in SPECFEM++:

1. :ref:`Range Policy <RangePolicy>` : This policy is used to iterate over a range of quadrature point where every distinct quadrature point within the assembly is visited exactly once.
2. :ref:`Chunk Element Policy <ChunkElementPolicy>` : This policy is used to iterate over a chunks of elements where every quadrature point within the chunk is visited exactly once.
3. :ref:`Edge Policy <policy_edge_index>`
