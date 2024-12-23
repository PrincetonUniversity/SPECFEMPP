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
          specfem::element::property_tag PropertyTag, int NGLL>
class source_kernel {
public:
  constexpr static int num_dimensions =
      specfem::element::attributes<DimensionType, MediumTag>::dimension();
  constexpr static int components =
      specfem::element::attributes<DimensionType, MediumTag>::components();
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto property_tag = PropertyTag;
  constexpr static auto dimension = DimensionType;
  constexpr static auto wavefield_tag = WavefieldType;

  constexpr static bool using_simd = false;

  source_kernel() = default;
  source_kernel(const specfem::compute::assembly &assembly);

  void compute_source_interaction(const int timestep);

private:
  using IndexViewType = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;
  IndexViewType elements;
  specfem::compute::sources sources;
  specfem::compute::simulation_field<WavefieldType> field;
};
} // namespace kernels
} // namespace impl
} // namespace domain
} // namespace specfem
