#pragma once

#include "compute/interface.hpp"
#include "domain/impl/sources/acoustic/interface.hpp"
#include "domain/impl/sources/elastic/interface.hpp"
#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace domain {
namespace impl {
namespace kernels {

template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, typename qp_type>
class source_kernel {
public:
  constexpr static int num_dimensions =
      specfem::element::attributes<DimensionType, MediumTag>::dimension();
  constexpr static int components =
      specfem::element::attributes<DimensionType, MediumTag>::components();
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto property_tag = PropertyTag;
  constexpr static auto dimension = DimensionType;

  using quadrature_point_type = qp_type;
  constexpr static bool using_simd = false;

  source_kernel() = default;
  source_kernel(
      const specfem::compute::assembly &assembly,
      const specfem::kokkos::HostView1d<int> h_source_domain_index_mapping,
      const quadrature_point_type quadrature_points);

  void compute_source_interaction(const int timestep) const;

private:
  int nsources;
  specfem::compute::points points;
  specfem::compute::quadrature quadrature;
  specfem::kokkos::DeviceView1d<int> source_domain_index_mapping;
  specfem::kokkos::HostMirror1d<int> h_source_domain_index_mapping;
  specfem::compute::properties properties;
  specfem::compute::simulation_field<WavefieldType> field;
  specfem::compute::source_medium<DimensionType, MediumTag> sources;
  quadrature_point_type quadrature_points;
  specfem::domain::impl::sources::source<dimension, medium_tag, property_tag,
                                         quadrature_point_type, using_simd>
      source;
};
} // namespace kernels
} // namespace impl
} // namespace domain
} // namespace specfem
