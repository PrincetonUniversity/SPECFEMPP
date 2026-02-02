#pragma once
#include "enumerations/interface.hpp"
#include "impl/bounding_box.hpp"

namespace specfem::assembly {

template <specfem::dimension::type DimensionTag> struct assembly;

template <> struct Info<specfem::dimension::type::dim3> {
  constexpr static auto dimension_tag =
      specfem::dimension::type::dim3; ///< Dimension tag

  Info() = default;

  Info(specfem::assembly::assembly<dimension_tag> &assembly);

  // Computed mesh properties
  info::impl::BoundingBox<dimension_tag> domain_bounds;
  info::impl::Bounds element_size;
  info::impl::Bounds gll_distance;
  info::impl::Bounds jacobian_determinant;
  info::impl::Bounds vp;
  info::impl::Bounds vs;
  info::impl::Bounds v;
  info::impl::Bounds rho;

  // Suggested time step based on CFL condition
  type_real suggested_time_step;
  type_real largest_minimum_period;

  std::string string() const;
};

} // namespace specfem::assembly
