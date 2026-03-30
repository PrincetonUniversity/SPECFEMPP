#include "compute_source_interaction.tpp"
#include "specfem/element.hpp"
#include "specfem/macros/compute_instantiation_macros.hpp"
#include "specfem/simulation.hpp"
#include "specfem/tags.hpp"

using specfem::element::dimension_tag;
using specfem::element::medium_tag;
using specfem::simulation::field_type;

#define SIGNATURE(NGLL, DIM, WF, MED)                                          \
  template void specfem::compute::impl::compute_source_interaction<            \
      NGLL, specfem::tags::Tags<DIM, WF, MED> >(                               \
      specfem::assembly::assembly<DIM> &, const int);
SPECFEM_COMPUTE_COMBINATIONS(SIGNATURE)
#undef SIGNATURE
