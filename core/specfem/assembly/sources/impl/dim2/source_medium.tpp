#pragma once

#include "specfem/algorithms.hpp"
#include "specfem/point.hpp"
#include "../source_medium.hpp"
#include "specfem/assembly/compute_source_array.hpp"
#include <Kokkos_Core.hpp>

template<typename ViewType>
void print_view_info(const ViewType& view, const std::string& name) {
    std::cout << name << " info:" << std::endl;
    std::cout << "  Layout: " << typeid(typename ViewType::array_layout).name() << std::endl;
    std::cout << "  Memory space: " << typeid(typename ViewType::memory_space).name() << std::endl;
    std::cout << "  Execution space: " << typeid(typename ViewType::execution_space).name() << std::endl;
    std::cout << "  Dimensions: ";
    for (int i = 0; i < ViewType::rank; ++i) {
        std::cout << view.extent(i) << " ";
    }
    std::cout << std::endl;
}

// 2D Constructor
/*
// TODO(Lucas : CPP20 update)
template <specfem::dimension::type DimensionTag, specfem::element::medium_tag MediumTag>
template <typename... Args>
requires (DimensionTag == specfem::dimension::type::dim2)
specfem::assembly::sources_impl::source_medium<DimensionTag, MediumTag>::source_medium(Args&&... args)
*/
template <specfem::dimension::type DimensionTag, specfem::element::medium_tag MediumTag>
template <specfem::dimension::type U, typename std::enable_if<U == specfem::dimension::type::dim2>::type*>
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
      source_array("specfem::sources::source_array", sources.size(), components, mesh.element_grid.ngllz, mesh.element_grid.ngllx),
      h_source_array(Kokkos::create_mirror_view(source_array)) {

  for (int isource = 0; isource < sources.size(); isource++) {

    // Get source array for a single source
    auto sv_source_array = Kokkos::subview(this->h_source_array, isource, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

    // Compute source contribution on the element
    specfem::assembly::compute_source_array(sources[isource], mesh, jacobian_matrix, sv_source_array);

    // Get source time function array for this source
    auto sv_stf_array = Kokkos::subview(this->h_source_time_function, Kokkos::ALL, isource, Kokkos::ALL);

    // Compute source time function array for this source
    sources[isource]->compute_source_time_function(t0, dt, nsteps, sv_stf_array);

    // Get global coordinates for this source
    const auto coord = sources[isource]->get_global_coordinates();

    // Get local coordinates for this source
    auto lcoord = sources[isource]->get_local_coordinates();

    // Assign local spectral element index to the mapping
    this->h_source_index_mapping(isource) = lcoord.ispec;
  }

  Kokkos::deep_copy(source_array, h_source_array);
  Kokkos::deep_copy(source_time_function, h_source_time_function);
  Kokkos::deep_copy(source_index_mapping, h_source_index_mapping);
}

// 2D load_on_device
/*
// TODO(Lucas : CPP20 update)
template <specfem::dimension::type DimensionTag, specfem::element::medium_tag MediumTag>
template <typename IndexType, typename PointSourceType>
requires (DimensionTag == specfem::dimension::type::dim2)
KOKKOS_INLINE_FUNCTION void
specfem::assembly::sources_impl::source_medium<DimensionTag, MediumTag>::load_on_device(
    const int timestep, const IndexType &index, PointSourceType &point_source) const
*/
template <specfem::dimension::type DimensionTag, specfem::element::medium_tag MediumTag>
template <typename IndexType, typename PointSourceType,
          specfem::dimension::type U,
          typename std::enable_if<U == specfem::dimension::type::dim2>::type*>
KOKKOS_INLINE_FUNCTION void
specfem::assembly::sources_impl::source_medium<DimensionTag, MediumTag>::load_on_device(
    const int timestep, const IndexType &index, PointSourceType &point_source) const {
  const auto isource = index.imap;
  for (int component = 0; component < components; component++) {
    point_source.stf(component) = source_time_function(timestep, isource, component);
    point_source.lagrange_interpolant(component) = source_array(isource, component, index.iz, index.ix);
  }
}

// 2D store_on_device
/*
// TODO(Lucas : CPP20 update)
template <specfem::dimension::type DimensionTag, specfem::element::medium_tag MediumTag>
template <typename IndexType, typename PointSourceType>
requires (DimensionTag == specfem::dimension::type::dim2)
KOKKOS_INLINE_FUNCTION void
specfem::assembly::sources_impl::source_medium<DimensionTag, MediumTag>::store_on_device(
    const int timestep, const IndexType index, const PointSourceType &point_source) const
*/
template <specfem::dimension::type DimensionTag, specfem::element::medium_tag MediumTag>
template <typename IndexType, typename PointSourceType,
          specfem::dimension::type U,
          typename std::enable_if<U == specfem::dimension::type::dim2>::type*>
KOKKOS_INLINE_FUNCTION void
specfem::assembly::sources_impl::source_medium<DimensionTag, MediumTag>::store_on_device(
    const int timestep, const IndexType index, const PointSourceType &point_source) const {
  const auto isource = index.imap;
  for (int component = 0; component < components; component++) {
    source_time_function(timestep, isource, component) = point_source.stf(component);
    source_array(isource, component, index.iz, index.ix) = point_source.lagrange_interpolant(component);
  }
}

// 2D load_on_host
/*
// TODO(Lucas : CPP20 update)
template <specfem::dimension::type DimensionTag, specfem::element::medium_tag MediumTag>
template <typename IndexType, typename PointSourceType>
requires (DimensionTag == specfem::dimension::type::dim2)
void
specfem::assembly::sources_impl::source_medium<DimensionTag, MediumTag>::load_on_host(
    const int timestep, const IndexType index, PointSourceType &point_source) const
*/
template <specfem::dimension::type DimensionTag, specfem::element::medium_tag MediumTag>
template <typename IndexType, typename PointSourceType,
          specfem::dimension::type U,
          typename std::enable_if<U == specfem::dimension::type::dim2>::type*>
void
specfem::assembly::sources_impl::source_medium<DimensionTag, MediumTag>::load_on_host(
    const int timestep, const IndexType index, PointSourceType &point_source) const {
  const auto isource = index.imap;
  for (int component = 0; component < components; component++) {
    point_source.stf(component) = h_source_time_function(timestep, isource, component);
    point_source.lagrange_interpolant(component) = h_source_array(isource, component, index.iz, index.ix);
  }
}

// 2D store_on_host
/*
// TODO(Lucas : CPP20 update)
template <specfem::dimension::type DimensionTag, specfem::element::medium_tag MediumTag>
template <typename IndexType, typename PointSourceType>
requires (DimensionTag == specfem::dimension::type::dim2)
void
specfem::assembly::sources_impl::source_medium<DimensionTag, MediumTag>::store_on_host(
    const int timestep, const IndexType index, const PointSourceType &point_source) const
*/
template <specfem::dimension::type DimensionTag, specfem::element::medium_tag MediumTag>
template <typename IndexType, typename PointSourceType,
          specfem::dimension::type U,
          typename std::enable_if<U == specfem::dimension::type::dim2>::type*>
void
specfem::assembly::sources_impl::source_medium<DimensionTag, MediumTag>::store_on_host(
    const int timestep, const IndexType index, const PointSourceType &point_source) const {
  const auto isource = index.imap;
  for (int component = 0; component < components; component++) {
    h_source_time_function(timestep, isource, component) = point_source.stf(component);
    h_source_array(isource, component, index.iz, index.ix) = point_source.lagrange_interpolant(component);
  }
}
