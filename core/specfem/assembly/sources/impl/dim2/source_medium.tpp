#pragma once

#include "algorithms/locate_point.hpp"
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

// 2D-specific template implementations

// 2D Constructor
/*
// Future C++20 version:
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
      source_array("specfem::sources::source_array", sources.size(), components, mesh.ngllz, mesh.ngllx),
      h_source_array(Kokkos::create_mirror_view(source_array)) {

  for (int isource = 0; isource < sources.size(); isource++) {

    std::cout << "Getting subview of h_source_array: " <<  specfem::element::to_string(MediumTag) << std::endl;
    // print layout of h_source_array
    print_view_info(this->h_source_array, "h_source_array");


    auto sv_source_array = Kokkos::subview(this->h_source_array, isource, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);


    print_view_info(sv_source_array, "sv_source_array");
    std::cout << "Got subview of h_source_array:     " <<  specfem::element::to_string(MediumTag) << std::endl;


    specfem::assembly::compute_source_array(sources[isource], mesh, jacobian_matrix, sv_source_array);

    std::cout << "computed source array:             " <<  specfem::element::to_string(MediumTag) << std::endl;
    std::cout << "Getting sv of h_stf:               " <<  specfem::element::to_string(MediumTag) << std::endl;
    auto sv_stf_array = Kokkos::subview(this->h_source_time_function, Kokkos::ALL, isource, Kokkos::ALL);
    std::cout << "Got subview of h_stf:              " <<  specfem::element::to_string(MediumTag) << std::endl;
    sources[isource]->compute_source_time_function(t0, dt, nsteps, sv_stf_array);
    std::cout << "Computed source time function:     " <<  specfem::element::to_string(MediumTag) << std::endl;

    const auto coord = sources[isource]->get_global_coordinates();
    auto lcoord = specfem::algorithms::locate_point(coord, mesh);
    this->h_source_index_mapping(isource) = lcoord.ispec;
  }

  Kokkos::deep_copy(source_array, h_source_array);
  Kokkos::deep_copy(source_time_function, h_source_time_function);
  Kokkos::deep_copy(source_index_mapping, h_source_index_mapping);
}

// 2D load_on_device
/*
// CPP20_CHANGE
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
// CPP20_CHANGE
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
// CPP20_CHANGE
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
// CPP20_CHANGE
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
