#include "algorithms/interface.hpp"
#include "compute/sources/source_medium.hpp"
#include "compute/sources/source_medium.tpp"
#include "compute/sources/sources.hpp"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "source/interface.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <vector>

// Forward declarations

template class specfem::compute::impl::source_medium<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic>;

template class specfem::compute::impl::source_medium<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic>;

specfem::compute::sources::sources(
    const std::vector<std::shared_ptr<specfem::sources::source> > &sources,
    const specfem::compute::mesh &mesh,
    const specfem::compute::partial_derivatives &partial_derivatives,
    const specfem::compute::properties &properties, const type_real t0,
    const type_real dt, const int nsteps)
    : nspec(mesh.nspec),
      source_domain_index_mapping(
          "specfem::sources::source_domain_index_mapping", nspec),
      h_source_domain_index_mapping(
          Kokkos::create_mirror_view(source_domain_index_mapping)),
      medium_types("specfem::sources::medium_types", nspec),
      h_medium_types(Kokkos::create_mirror_view(medium_type)),
      wavefield_types("specfem::sources::wavefield_types", nspec),
      h_wavefield_types(Kokkos::create_mirror_view(wavefield_types)) {

  const specfem::element::medium_tag acoustic =
      specfem::element::medium_tag::acoustic;
  const specfem::element::medium_tag elastic =
      specfem::element::medium_tag::elastic;

  std::vector<std::shared_ptr<specfem::sources::source> > acoustic_sources;
  std::vector<std::shared_ptr<specfem::sources::source> > elastic_sources;

  for (int ispec = 0; ispec < nspec; ispec++) {
    source_domain_index_mapping(ispec) = -1;
  }

  for (int isource = 0; isource < sources.size(); isource++) {
    const auto &source = sources[isource];

    // Get local coordinate for the source
    const type_real x = source->get_x();
    const type_real z = source->get_z();
    const specfem::point::global_coordinates<specfem::dimension::type::dim2>
        coord(x, z);
    const auto lcoord = specfem::algorithms::locate_point(coord, mesh);
    //-------------------------------------

    if (properties.h_element_types(lcoord.ispec) == acoustic) {
      acoustic_sources.push_back(source);
      h_source_domain_index_mapping(lcoord.ispec) = acoustic_sources.size() - 1;
      h_medium_types(lcoord.ispec) = acoustic;
      h_wavefield_types(lcoord.ispec) = source->get_wavefield_type();
    } else if (properties.h_element_types(lcoord.ispec) == elastic) {
      elastic_sources.push_back(source);
      h_source_domain_index_mapping(lcoord.ispec) = elastic_sources.size() - 1;
      h_medium_types(lcoord.ispec) = elastic;
      h_wavefield_types(lcoord.ispec) = source->get_wavefield_type();
    } else {
      throw std::runtime_error("Unknown medium type");
    }
  }

  Kokkos::deep_copy(source_domain_index_mapping, h_source_domain_index_mapping);
  Kokkos::deep_copy(medium_types, h_medium_types);
  Kokkos::deep_copy(wavefield_types, h_wavefield_types);

  this->acoustic_sources = specfem::compute::impl::source_medium<
      specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic>(
      acoustic_sources, mesh, partial_derivatives, properties, t0, dt, nsteps);

  this->elastic_sources = specfem::compute::impl::source_medium<
      specfem::dimension::type::dim2, specfem::element::medium_tag::elastic>(
      elastic_sources, mesh, partial_derivatives, properties, t0, dt, nsteps);
}

Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>
specfem::compute::sources::get_sources_on_host(
    const specfem::element::medium_tag medium,
    const specfem::wavefield::simulation_field wavefield) const {

  int nsources = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if ((medium_types(ispec) == medium) &&
        (wavefield_types(ispec) == wavefield)) {
      nsources++;
    }
  }

  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> indices(
      "specfem::compute::sources::get_sources_on_host", nsources);

  int isource = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if ((medium_types(ispec) == medium) &&
        (wavefield_types(ispec) == wavefield)) {
      indices(isource) = ispec;
      isource++;
    }
  }

  return indices;
}

Kokkos::View<int *, Kokkos::DefaultExecutionSpace>
specfem::compute::sources::get_sources_on_device(
    const specfem::element::medium_tag medium,
    const specfem::wavefield::simulation_field wavefield) const {

  auto h_indices = get_sources_on_host(medium, wavefield);

  Kokkos::View<int *, Kokkos::DefaultExecutionSpace> indices(
      "specfem::compute::sources::get_sources_on_device", h_indices.extent(0));

  Kokkos::deep_copy(indices, h_indices);

  return indices;
}
