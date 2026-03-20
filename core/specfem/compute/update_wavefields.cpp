#include "update_wavefields.tpp"
#include "specfem/element.hpp"
#include "specfem/macros/compute_instantiation_macros.hpp"
#include "specfem/simulation.hpp"
#include "specfem/tags.hpp"

using specfem::element::dimension_tag;
using specfem::element::medium_tag;
using specfem::simulation::field_type;

#define INST_UPDATE_WAVEFIELDS(NGLL, DIM, WF, MED)                             \
  INSTANTIATE_COMPUTE_FUNCTION_INT_NONCONST_INT(update_wavefields, NGLL, DIM,  \
                                                WF, MED)
SPECFEM_COMPUTE_COMBINATIONS(INST_UPDATE_WAVEFIELDS)
#undef INST_UPDATE_WAVEFIELDS
