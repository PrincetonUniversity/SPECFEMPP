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

template class specfem::compute::source_medium<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic>;

template class specfem::compute::source_medium<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic>;

specfem::compute::sources::sources(
    const std::vector<std::shared_ptr<specfem::sources::source> > &sources,
    const specfem::compute::mesh &mesh,
    const specfem::compute::partial_derivatives &partial_derivatives,
    const specfem::compute::properties &properties, const type_real t0,
    const type_real dt, const int nsteps)
    : nsources(sources.size()),
      source_domain_index_mapping(
          "specfem::sources::source_domain_index_mapping", sources.size()),
      source_medium_mapping("specfem::sources::source_medium_mapping",
                            sources.size()),
      source_wavefield_mapping("specfem::sources::source_wavefield_mapping",
                               sources.size()) {

  const specfem::element::medium_tag acoustic =
      specfem::element::medium_tag::acoustic;
  const specfem::element::medium_tag elastic =
      specfem::element::medium_tag::elastic;

  std::vector<std::shared_ptr<specfem::sources::source> > acoustic_sources;
  std::vector<std::shared_ptr<specfem::sources::source> > elastic_sources;

  for (int isource = 0; isource < sources.size(); isource++) {
    const auto &source = sources[isource];

    // Get local coordinate for the source
    const type_real x = source->get_x();
    const type_real z = source->get_z();
    const specfem::point::global_coordinates<specfem::dimension::type::dim2>
        coord(x, z);
    const auto lcoord = specfem::algorithms::locate_point(coord, mesh);
    //-------------------------------------

    if (properties.h_medium_tags(lcoord.ispec) == acoustic) {
      acoustic_sources.push_back(source);
      source_domain_index_mapping(isource) = acoustic_sources.size() - 1;
      source_medium_mapping(isource) = acoustic;
      source_wavefield_mapping(isource) = source->get_wavefield_type();
    } else if (properties.h_medium_tags(lcoord.ispec) == elastic) {
      elastic_sources.push_back(source);
      source_domain_index_mapping(isource) = elastic_sources.size() - 1;
      source_medium_mapping(isource) = elastic;
      source_wavefield_mapping(isource) = source->get_wavefield_type();
    } else {
      throw std::runtime_error("Unknown medium type");
    }
  }

  this->acoustic_sources =
      specfem::compute::source_medium<specfem::dimension::type::dim2,
                                      specfem::element::medium_tag::acoustic>(
          acoustic_sources, mesh, partial_derivatives, properties, t0, dt,
          nsteps);

  this->elastic_sources =
      specfem::compute::source_medium<specfem::dimension::type::dim2,
                                      specfem::element::medium_tag::elastic>(
          elastic_sources, mesh, partial_derivatives, properties, t0, dt,
          nsteps);
}
