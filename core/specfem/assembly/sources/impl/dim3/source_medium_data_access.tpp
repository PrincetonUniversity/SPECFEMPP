#pragma once

#include "../source_medium.hpp"
#include <Kokkos_Core.hpp>

// 3D load_on_device
template <specfem::element::dimension_tag DimensionTag, specfem::element::medium_tag MediumTag>
template <typename IndexType, typename PointSourceType,
          specfem::element::dimension_tag U,
          typename std::enable_if<U == specfem::element::dimension_tag::dim3>::type*>
KOKKOS_INLINE_FUNCTION void
specfem::assembly::sources_impl::source_medium<DimensionTag, MediumTag>::load_on_device(
    const int timestep, const IndexType &index, PointSourceType &point_source) const {
  const auto isource = index.imap;
  for (int component = 0; component < components; component++) {
    point_source.stf(component) = source_time_function(timestep, isource, component);
    point_source.lagrange_interpolant(component) = source_array(isource, component, index.iz, index.iy, index.ix);
  }
}

// 3D store_on_device
template <specfem::element::dimension_tag DimensionTag, specfem::element::medium_tag MediumTag>
template <typename IndexType, typename PointSourceType,
          specfem::element::dimension_tag U,
          typename std::enable_if<U == specfem::element::dimension_tag::dim3>::type*>
KOKKOS_INLINE_FUNCTION void
specfem::assembly::sources_impl::source_medium<DimensionTag, MediumTag>::store_on_device(
    const int timestep, const IndexType index, const PointSourceType &point_source) const {
  const auto isource = index.imap;
  for (int component = 0; component < components; component++) {
    source_time_function(timestep, isource, component) = point_source.stf(component);
    source_array(isource, component, index.iz, index.iy, index.ix) = point_source.lagrange_interpolant(component);
  }
}

// 3D load_on_host
template <specfem::element::dimension_tag DimensionTag, specfem::element::medium_tag MediumTag>
template <typename IndexType, typename PointSourceType,
          specfem::element::dimension_tag U,
          typename std::enable_if<U == specfem::element::dimension_tag::dim3>::type*>
void
specfem::assembly::sources_impl::source_medium<DimensionTag, MediumTag>::load_on_host(
    const int timestep, const IndexType index, PointSourceType &point_source) const {
  const auto isource = index.imap;
  for (int component = 0; component < components; component++) {
    point_source.stf(component) = h_source_time_function(timestep, isource, component);
    point_source.lagrange_interpolant(component) = h_source_array(isource, component, index.iz, index.iy, index.ix);
  }
}

// 3D store_on_host
template <specfem::element::dimension_tag DimensionTag, specfem::element::medium_tag MediumTag>
template <typename IndexType, typename PointSourceType,
          specfem::element::dimension_tag U,
          typename std::enable_if<U == specfem::element::dimension_tag::dim3>::type*>
void
specfem::assembly::sources_impl::source_medium<DimensionTag, MediumTag>::store_on_host(
    const int timestep, const IndexType index, const PointSourceType &point_source) const {
  const auto isource = index.imap;
  for (int component = 0; component < components; component++) {
    h_source_time_function(timestep, isource, component) = point_source.stf(component);
    h_source_array(isource, component, index.iz, index.iy, index.ix) = point_source.lagrange_interpolant(component);
  }
}
