
# Test Description

This test verifies the correctness of the index mapping computation using an adjacency map on a homogeneous elastic domain with no interfaces. With the latest changes, the mesher writes the adjacency map to the mesh file. We use this adjacency map to compute the index mapping. We currently do not have a test that verifies the correctness of the index mapping itself. However, this test should serve as a good proxy since if the adjacency map is incorrect, the index mapping will also be incorrect, leading to incorrect traces.

# Test Reproducibility

Generate the traces using ``SPECFEM2D``

- Par file : ``provenance/SPECFEM2D_Par_file``

Generate the database using ``SPECFEM++``

- Par file : ``provenance/SPECFEMPP_Par_file``
