#include "compute_source_interaction.tpp"
#include "specfem/element.hpp"
#include "specfem/macros/compute_instantiation_macros.hpp"
#include "specfem/simulation.hpp"
#include "specfem/tags.hpp"

using specfem::element::dimension_tag;
using specfem::element::medium_tag;
using specfem::simulation::field_type;

#define INST_COMPUTE_SOURCE_INTERACTION(NGLL, DIM, WF, MED)                    \
  INSTANTIATE_COMPUTE_FUNCTION_VOID_NONCONST_INT(compute_source_interaction,   \
                                                 NGLL, DIM, WF, MED)
SPECFEM_COMPUTE_COMBINATIONS(INST_COMPUTE_SOURCE_INTERACTION)
#undef INST_COMPUTE_SOURCE_INTERACTION
