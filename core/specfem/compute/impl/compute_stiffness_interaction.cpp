#include "compute_stiffness_interaction.tpp"
#include "specfem/element.hpp"
#include "specfem/macros/compute_instantiation_macros.hpp"
#include "specfem/simulation.hpp"
#include "specfem/tags.hpp"

using specfem::element::dimension_tag;
using specfem::element::medium_tag;
using specfem::simulation::field_type;

#define INST_COMPUTE_STIFFNESS_INTERACTION(NGLL, DIM, WF, MED)                 \
  INSTANTIATE_COMPUTE_FUNCTION_INT_CONST_INT(compute_stiffness_interaction,    \
                                             NGLL, DIM, WF, MED)
SPECFEM_COMPUTE_COMBINATIONS(INST_COMPUTE_STIFFNESS_INTERACTION)
#undef INST_COMPUTE_STIFFNESS_INTERACTION
