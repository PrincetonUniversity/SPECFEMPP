#ifndef _COMPUTE_SOURCES_2_HPP
#define _COMPUTE_SOURCES_2_HPP

#include "compute/compute_mesh.hpp"
#include "compute/compute_partial_derivatives.hpp"
#include "compute/properties/properties.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/wavefield.hpp"
#include "impl/source_medium.hpp"
#include "kokkos_abstractions.h"
#include "source/source.hpp"

namespace specfem {
namespace compute {
struct sources {

  sources() = default;

  sources(
      const std::vector<std::shared_ptr<specfem::sources::source> > &sources,
      const specfem::compute::mesh &mesh,
      const specfem::compute::partial_derivatives &partial_derivatives,
      const specfem::compute::properties &properties, const type_real t0,
      const type_real dt, const int nsteps);

  template <specfem::element::medium_tag Medium>
  inline specfem::compute::impl::sources::source_medium<
      specfem::dimension::type::dim2, Medium>
  get_source_medium() const {
    if constexpr (Medium == specfem::element::medium_tag::acoustic) {
      return acoustic_sources;
    } else if constexpr (Medium == specfem::element::medium_tag::elastic) {
      return elastic_sources;
    } else {
      static_assert("Invalid medium type");
    }
  }

  int nsources;
  specfem::kokkos::HostView1d<int> source_domain_index_mapping;
  specfem::kokkos::HostView1d<specfem::element::medium_tag>
      source_medium_mapping;
  specfem::kokkos::HostView1d<specfem::wavefield::type>
      source_wavefield_mapping;
  specfem::compute::impl::sources::source_medium<
      specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic>
      acoustic_sources;
  specfem::compute::impl::sources::source_medium<
      specfem::dimension::type::dim2, specfem::element::medium_tag::elastic>
      elastic_sources;
};
} // namespace compute
} // namespace specfem

#endif /* _COMPUTE_SOURCES_2_HPP */
