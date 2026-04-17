#pragma once

#include "specfem/element.hpp"
#include "specfem/assembly/sources/impl/source_medium.hpp"
#include "specfem/assembly/compute_source_array.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/point.hpp"
#include "specfem/source.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <type_traits>
#include <vector>


template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
template <specfem::element::dimension_tag U,
          typename std::enable_if<
              U == specfem::element::dimension_tag::dim2>::type *>
specfem::assembly::sources_impl::source_medium<DimensionTag, MediumTag>::source_medium(
    const std::vector<
        std::shared_ptr<specfem::sources::source<DimensionTag> > > &sources,
    const specfem::assembly::mesh<DimensionTag> &mesh,
    const specfem::assembly::jacobian_matrix<DimensionTag> &jacobian_matrix,
    const specfem::assembly::element_types<DimensionTag> &element_types,
    const type_real t0, const type_real dt, const int nsteps)
    : source_index_mapping("specfem::sources::source_index_mapping",
                           sources.size()),
      h_source_index_mapping(
          Kokkos::create_mirror_view(source_index_mapping)),
      source_time_function("specfem::sources::source_time_function", nsteps,
                           sources.size(), components),
      h_source_time_function(
          Kokkos::create_mirror_view(source_time_function)),
      source_array("specfem::sources::source_array", sources.size(),
                   components, mesh.element_grid.ngllz,
                   mesh.element_grid.ngllx),
      h_source_array(Kokkos::create_mirror_view(source_array)) {

  for (int isource = 0; isource < (int)sources.size(); isource++) {
    auto sv_source_array = Kokkos::subview(
        this->h_source_array, isource, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    specfem::assembly::compute_source_array(sources[isource], mesh,
                                            jacobian_matrix, sv_source_array);
    auto sv_stf_array = Kokkos::subview(this->h_source_time_function,
                                        Kokkos::ALL, isource, Kokkos::ALL);
    sources[isource]->compute_source_time_function(t0, dt, nsteps,
                                                   sv_stf_array);
    this->h_source_index_mapping(isource) =
        sources[isource]->get_local_coordinates().ispec;
  }

  Kokkos::deep_copy(source_array, h_source_array);
  Kokkos::deep_copy(source_time_function, h_source_time_function);
  Kokkos::deep_copy(source_index_mapping, h_source_index_mapping);
}

template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
template <typename IndexType, typename PointSourceType,
          specfem::element::dimension_tag U,
          typename std::enable_if<
              U == specfem::element::dimension_tag::dim2>::type *>
KOKKOS_INLINE_FUNCTION void
specfem::assembly::sources_impl::source_medium<DimensionTag, MediumTag>::load_on_device(
    const int timestep, const IndexType &index,
    PointSourceType &point_source) const {
  const auto isource = index.imap;
  for (int component = 0; component < components; component++) {
    point_source.stf(component) =
        source_time_function(timestep, isource, component);
    point_source.lagrange_interpolant(component) =
        source_array(isource, component, index.iz, index.ix);
  }
}

template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
template <typename IndexType, typename PointSourceType,
          specfem::element::dimension_tag U,
          typename std::enable_if<
              U == specfem::element::dimension_tag::dim2>::type *>
KOKKOS_INLINE_FUNCTION void
specfem::assembly::sources_impl::source_medium<DimensionTag, MediumTag>::store_on_device(
    const int timestep, const IndexType index,
    const PointSourceType &point_source) const {
  const auto isource = index.imap;
  for (int component = 0; component < components; component++) {
    source_time_function(timestep, isource, component) =
        point_source.stf(component);
    source_array(isource, component, index.iz, index.ix) =
        point_source.lagrange_interpolant(component);
  }
}

template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
template <typename IndexType, typename PointSourceType,
          specfem::element::dimension_tag U,
          typename std::enable_if<
              U == specfem::element::dimension_tag::dim2>::type *>
void specfem::assembly::sources_impl::source_medium<DimensionTag, MediumTag>::load_on_host(
    const int timestep, const IndexType index,
    PointSourceType &point_source) const {
  const auto isource = index.imap;
  for (int component = 0; component < components; component++) {
    point_source.stf(component) =
        h_source_time_function(timestep, isource, component);
    point_source.lagrange_interpolant(component) =
        h_source_array(isource, component, index.iz, index.ix);
  }
}

template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
template <typename IndexType, typename PointSourceType,
          specfem::element::dimension_tag U,
          typename std::enable_if<
              U == specfem::element::dimension_tag::dim2>::type *>
void specfem::assembly::sources_impl::source_medium<DimensionTag, MediumTag>::store_on_host(
    const int timestep, const IndexType index,
    const PointSourceType &point_source) const {
  const auto isource = index.imap;
  for (int component = 0; component < components; component++) {
    h_source_time_function(timestep, isource, component) =
        point_source.stf(component);
    h_source_array(isource, component, index.iz, index.ix) =
        point_source.lagrange_interpolant(component);
  }
}
