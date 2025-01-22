#pragma once

#include "algorithms/locate_point.hpp"
#include "point/coordinates.hpp"
#include "source_medium.hpp"
#include <Kokkos_Core.hpp>

template <specfem::dimension::type Dimension,
          specfem::element::medium_tag Medium>
specfem::compute::impl::source_medium<Dimension, Medium>::source_medium(
    const std::vector<std::shared_ptr<specfem::sources::source> > &sources,
    const specfem::compute::mesh &mesh,
    const specfem::compute::partial_derivatives &partial_derivatives,
    const specfem::compute::element_types &element_types, const type_real t0,
    const type_real dt, const int nsteps)
    : source_index_mapping("specfem::sources::source_index_mapping",
                           sources.size()),
      h_source_index_mapping(Kokkos::create_mirror_view(source_index_mapping)),
      source_time_function("specfem::sources::source_time_function", nsteps,
                           sources.size(), components),
      h_source_time_function(Kokkos::create_mirror_view(source_time_function)),
      source_array("specfem::sources::source_array", sources.size(), components,
                   mesh.quadratures.gll.N, mesh.quadratures.gll.N),
      h_source_array(Kokkos::create_mirror_view(source_array)) {

  for (int isource = 0; isource < sources.size(); isource++) {
    auto sv_source_array = Kokkos::subview(
        this->h_source_array, isource, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    sources[isource]->compute_source_array(mesh, partial_derivatives,
                                           element_types, sv_source_array);
    auto sv_stf_array = Kokkos::subview(this->h_source_time_function,
                                        Kokkos::ALL, isource, Kokkos::ALL);
    sources[isource]->compute_source_time_function(t0, dt, nsteps,
                                                   sv_stf_array);
    specfem::point::global_coordinates<specfem::dimension::type::dim2> coord(
        sources[isource]->get_x(), sources[isource]->get_z());

    auto lcoord = specfem::algorithms::locate_point(coord, mesh);
    this->h_source_index_mapping(isource) = lcoord.ispec;
  }

  Kokkos::deep_copy(source_array, h_source_array);
  Kokkos::deep_copy(source_time_function, h_source_time_function);
  Kokkos::deep_copy(source_index_mapping, h_source_index_mapping);

  return;
}
