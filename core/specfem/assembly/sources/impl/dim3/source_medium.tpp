#pragma once

#include "algorithms/locate_point.hpp"
#include "specfem/point.hpp"
#include "../source_medium.hpp"
#include "specfem/assembly/compute_source_array.hpp"
#include <Kokkos_Core.hpp>

// 3D Constructor
/*
// TODO(Lucas : CPP20 update)
template <specfem::dimension::type DimensionTag, specfem::element::medium_tag MediumTag>
template <typename... Args>
requires (DimensionTag == specfem::dimension::type::dim3)
specfem::assembly::sources_impl::source_medium<DimensionTag, MediumTag>::source_medium(Args&&... args)
*/
template <specfem::dimension::type DimensionTag, specfem::element::medium_tag MediumTag>
template <specfem::dimension::type U, typename std::enable_if<U == specfem::dimension::type::dim3>::type*>
specfem::assembly::sources_impl::source_medium<DimensionTag, MediumTag>::source_medium(
    const std::vector<std::shared_ptr<specfem::sources::source<dimension_tag> > > &sources,
    const specfem::assembly::mesh<dimension_tag> &mesh,
    const specfem::assembly::jacobian_matrix<dimension_tag> &jacobian_matrix,
    const specfem::assembly::element_types<dimension_tag> &element_types,
    const type_real t0, const type_real dt, const int nsteps)
    : source_index_mapping("specfem::sources::source_index_mapping", sources.size()),
      h_source_index_mapping(Kokkos::create_mirror_view(source_index_mapping)),
      source_time_function("specfem::sources::source_time_function", nsteps, sources.size(), components),
      h_source_time_function(Kokkos::create_mirror_view(source_time_function)),
      source_array("specfem::sources::source_array", sources.size(), components,
                   mesh.element_grid.ngllz,
                   mesh.element_grid.nglly,
                   mesh.element_grid.ngllx),
      h_source_array(Kokkos::create_mirror_view(source_array)) {

  for (int isource = 0; isource < sources.size(); isource++) {
    auto sv_source_array = Kokkos::subview(this->h_source_array, isource, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

    specfem::assembly::compute_source_array(sources[isource], mesh, jacobian_matrix, sv_source_array);

    auto sv_stf_array = Kokkos::subview(this->h_source_time_function, Kokkos::ALL, isource, Kokkos::ALL);
    sources[isource]->compute_source_time_function(t0, dt, nsteps, sv_stf_array);

    const auto coord = sources[isource]->get_global_coordinates();
    auto lcoord = specfem::algorithms::locate_point(coord, mesh);
    this->h_source_index_mapping(isource) = lcoord.ispec;
  }

  Kokkos::deep_copy(source_array, h_source_array);
  Kokkos::deep_copy(source_time_function, h_source_time_function);
  Kokkos::deep_copy(source_index_mapping, h_source_index_mapping);
}

// 3D load_on_device
/*
// TODO(Lucas : CPP20 update)
template <specfem::dimension::type DimensionTag, specfem::element::medium_tag MediumTag>
template <typename IndexType, typename PointSourceType>
requires (DimensionTag == specfem::dimension::type::dim3)
KOKKOS_INLINE_FUNCTION void
specfem::assembly::sources_impl::source_medium<DimensionTag, MediumTag>::load_on_device(
    const int timestep, const IndexType &index, PointSourceType &point_source) const
*/
template <specfem::dimension::type DimensionTag, specfem::element::medium_tag MediumTag>
template <typename IndexType, typename PointSourceType,
          specfem::dimension::type U,
          typename std::enable_if<U == specfem::dimension::type::dim3>::type*>
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
/*
// TODO(Lucas : CPP20 update)
template <specfem::dimension::type DimensionTag, specfem::element::medium_tag MediumTag>
template <typename IndexType, typename PointSourceType>
requires (DimensionTag == specfem::dimension::type::dim3)
KOKKOS_INLINE_FUNCTION void
specfem::assembly::sources_impl::source_medium<DimensionTag, MediumTag>::store_on_device(
    const int timestep, const IndexType index, const PointSourceType &point_source) const
*/
template <specfem::dimension::type DimensionTag, specfem::element::medium_tag MediumTag>
template <typename IndexType, typename PointSourceType,
          specfem::dimension::type U,
          typename std::enable_if<U == specfem::dimension::type::dim3>::type*>
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
/*
// TODO(Lucas : CPP20 update)
template <specfem::dimension::type DimensionTag, specfem::element::medium_tag MediumTag>
template <typename IndexType, typename PointSourceType>
requires (DimensionTag == specfem::dimension::type::dim3)
void
specfem::assembly::sources_impl::source_medium<DimensionTag, MediumTag>::load_on_host(
    const int timestep, const IndexType index, PointSourceType &point_source) const
*/
template <specfem::dimension::type DimensionTag, specfem::element::medium_tag MediumTag>
template <typename IndexType, typename PointSourceType,
          specfem::dimension::type U,
          typename std::enable_if<U == specfem::dimension::type::dim3>::type*>
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
/*
// TODO(Lucas : CPP20 update)
template <specfem::dimension::type DimensionTag, specfem::element::medium_tag MediumTag>
template <typename IndexType, typename PointSourceType>
requires (DimensionTag == specfem::dimension::type::dim3)
void
specfem::assembly::sources_impl::source_medium<DimensionTag, MediumTag>::store_on_host(
    const int timestep, const IndexType index, const PointSourceType &point_source) const
*/
template <specfem::dimension::type DimensionTag, specfem::element::medium_tag MediumTag>
template <typename IndexType, typename PointSourceType,
          specfem::dimension::type U,
          typename std::enable_if<U == specfem::dimension::type::dim3>::type*>
void
specfem::assembly::sources_impl::source_medium<DimensionTag, MediumTag>::store_on_host(
    const int timestep, const IndexType index, const PointSourceType &point_source) const {
  const auto isource = index.imap;
  for (int component = 0; component < components; component++) {
    h_source_time_function(timestep, isource, component) = point_source.stf(component);
    h_source_array(isource, component, index.iz, index.iy, index.ix) = point_source.lagrange_interpolant(component);
  }
}
