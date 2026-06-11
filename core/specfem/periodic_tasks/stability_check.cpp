#include "stability_check.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/constants.hpp"
#include "specfem/element/attributes.hpp"
#include "specfem/mpi.hpp"
#include "specfem/program/abort.hpp"
#include "specfem/tag_dispatch.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <sstream>

template <specfem::element::dimension_tag DimensionTag>
void specfem::periodic_tasks::stability_check<DimensionTag>::run(
    specfem::assembly::assembly<DimensionTag> &assembly, const int istep) {
  const auto &forward_field = assembly.fields.forward;
  type_real local_max_displacement = 0;

  auto check_medium = [&]<typename TagsType>() {
    // Skip combinations belonging to a different dimension
    if constexpr (TagsType::dimension_tag != DimensionTag)
      return;

    constexpr int num_components =
        specfem::element::attributes<DimensionTag,
                                     TagsType::medium_tag>::components;
    const auto &medium_field =
        forward_field.template get_field<TagsType::medium_tag>();
    if (medium_field.nglob == 0)
      return;
    const auto displacement = medium_field.get_field();
    type_real medium_max = 0;
    Kokkos::parallel_reduce(
        "specfem::periodic_tasks::stability_check",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,
                                                           medium_field.nglob),
        KOKKOS_LAMBDA(int iglob, type_real &lmax) {
          for (int icomp = 0; icomp < num_components; ++icomp)
            lmax = Kokkos::max(lmax, Kokkos::fabs(displacement(iglob, icomp)));
        },
        Kokkos::Max<type_real>(medium_max));
    Kokkos::fence();
    local_max_displacement = std::max(local_max_displacement, medium_max);
  };

  // Single for_each covers all dimensions; the constexpr guard skips
  // combinations that don't belong to DimensionTag, preventing invalid
  // get_field<> instantiations (e.g. get_field<elastic> on sim_field<dim2>).
  specfem::tag_dispatch::for_each(DIMENSION_SET(dim2, dim3) *
                                      MEDIUM_SET(elastic_psv, elastic_sh,
                                                 acoustic, poroelastic,
                                                 elastic_psv_t, elastic),
                                  check_medium);

  // Reduce across MPI ranks
  const auto comm = specfem::MPI::communicator();
  SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &local_max_displacement, 1,
                                     SPECFEM_MPI_TYPE_REAL, MPI_MAX, comm));

  if (!std::isfinite(local_max_displacement) ||
      local_max_displacement > specfem::constants::STABILITY_THRESHOLD) {
    std::ostringstream error_message;
    error_message << "Solution is diverging at time step " << istep << "!\n"
                  << "  max |displacement| = " << local_max_displacement << "\n"
                  << "  Stability threshold = "
                  << specfem::constants::STABILITY_THRESHOLD << "\n"
                  << "  Verify that dt satisfies the CFL condition and that\n"
                  << "  source parameters are physically reasonable.";
    specfem_abort(error_message.str(), 30);
  }
}

// Explicit template instantiations
template class specfem::periodic_tasks::stability_check<
    specfem::element::dimension_tag::dim2>;
template class specfem::periodic_tasks::stability_check<
    specfem::element::dimension_tag::dim3>;
