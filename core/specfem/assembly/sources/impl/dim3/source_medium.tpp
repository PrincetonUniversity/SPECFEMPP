#pragma once

#include "specfem/point.hpp"
#include "../source_medium.hpp"
#include "specfem/assembly/compute_source_array.hpp"
#include <Kokkos_Core.hpp>

// 3D Constructor
/*
// TODO(Lucas : CPP20 update)
template <specfem::element::dimension_tag DimensionTag, specfem::element::medium_tag MediumTag>
template <typename... Args>
requires (DimensionTag == specfem::element::dimension_tag::dim3)
specfem::assembly::sources_impl::source_medium<DimensionTag, MediumTag>::source_medium(Args&&... args)
*/
template <specfem::element::dimension_tag DimensionTag, specfem::element::medium_tag MediumTag>
template <specfem::element::dimension_tag U, typename std::enable_if<U == specfem::element::dimension_tag::dim3>::type*>
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

    const auto lcoord = sources[isource]->get_local_coordinates();
    this->h_source_index_mapping(isource) = lcoord.ispec;
  }

  Kokkos::deep_copy(source_array, h_source_array);
  Kokkos::deep_copy(source_time_function, h_source_time_function);
  Kokkos::deep_copy(source_index_mapping, h_source_index_mapping);
}
