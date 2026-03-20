#include "divide_mass_matrix.tpp"
#include "specfem/element.hpp"
#include "specfem/macros/compute_instantiation_macros.hpp"
#include "specfem/simulation.hpp"
#include "specfem/tags.hpp"

using specfem::element::dimension_tag;
using specfem::element::medium_tag;
using specfem::simulation::field_type;

#define INST_DIVIDE_MASS_MATRIX(NGLL, DIM, WF, MED)                            \
  INSTANTIATE_COMPUTE_FUNCTION(divide_mass_matrix, NGLL, DIM, WF, MED)
SPECFEM_COMPUTE_COMBINATIONS(INST_DIVIDE_MASS_MATRIX)
#undef INST_DIVIDE_MASS_MATRIX
