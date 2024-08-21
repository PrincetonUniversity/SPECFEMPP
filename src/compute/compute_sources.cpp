#include "algorithms/interface.hpp"
#include "compute/sources/impl/source_medium.hpp"
#include "compute/sources/impl/source_medium.tpp"
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

template class specfem::compute::impl::sources::source_medium<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic>;

template class specfem::compute::impl::sources::source_medium<
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

    if (properties.h_element_types(lcoord.ispec) == acoustic) {
      acoustic_sources.push_back(source);
      source_domain_index_mapping(isource) = acoustic_sources.size() - 1;
      source_medium_mapping(isource) = acoustic;
      source_wavefield_mapping(isource) = source->get_wavefield_type();
    } else if (properties.h_element_types(lcoord.ispec) == elastic) {
      elastic_sources.push_back(source);
      source_domain_index_mapping(isource) = elastic_sources.size() - 1;
      source_medium_mapping(isource) = elastic;
      source_wavefield_mapping(isource) = source->get_wavefield_type();
    } else {
      throw std::runtime_error("Unknown medium type");
    }
  }

  this->acoustic_sources = specfem::compute::impl::sources::source_medium<
      specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic>(
      acoustic_sources, mesh, partial_derivatives, properties, t0, dt, nsteps);

  this->elastic_sources = specfem::compute::impl::sources::source_medium<
      specfem::dimension::type::dim2, specfem::element::medium_tag::elastic>(
      elastic_sources, mesh, partial_derivatives, properties, t0, dt, nsteps);
}

// specfem::compute::sources::sources(
//     const std::vector<std::shared_ptr<specfem::sources::source> > &sources,
//     const specfem::quadrature::quadrature *quadx,
//     const specfem::quadrature::quadrature *quadz, const type_real xmax,
//     const type_real xmin, const type_real zmax, const type_real zmin,
//     specfem::MPI::MPI *mpi) {

//   // Get  sources which lie in processor
//   std::vector<std::shared_ptr<specfem::sources::source> > my_sources;
//   for (auto &source : sources) {
//     if (source->get_islice() == mpi->get_rank()) {
//       my_sources.push_back(source);
//     }
//   }

//   // allocate source array view
//   this->source_array = specfem::kokkos::DeviceView4d<type_real>(
//       "specfem::compute::sources::source_array", my_sources.size(),
//       quadz->get_N(), quadx->get_N(), ndim);

//   this->h_source_array = Kokkos::create_mirror_view(this->source_array);

//   this->stf_array =
//       specfem::kokkos::DeviceView1d<specfem::forcing_function::stf_storage>(
//           "specfem::compute::sources::stf_array", my_sources.size());

//   this->h_stf_array = Kokkos::create_mirror_view(this->stf_array);

//   this->ispec_array = specfem::kokkos::DeviceView1d<int>(
//       "specfem::compute::sources::ispec_array", my_sources.size());

//   this->h_ispec_array = Kokkos::create_mirror_view(ispec_array);

//   // store source array for sources in my islice
//   for (int isource = 0; isource < my_sources.size(); isource++) {

//     my_sources[isource]->check_locations(xmax, xmin, zmax, zmin, mpi);

//     auto sv_source_array = Kokkos::subview(
//         this->h_source_array, isource, Kokkos::ALL, Kokkos::ALL,
//         Kokkos::ALL);
//     my_sources[isource]->compute_source_array(quadx, quadz, sv_source_array);

//     this->h_stf_array(isource).T = my_sources[isource]->get_stf();
//     this->h_ispec_array(isource) = my_sources[isource]->get_ispec();
//   }

//   this->sync_views();

//   return;
// };

// void specfem::compute::sources::sync_views() {
//   Kokkos::deep_copy(source_array, h_source_array);
//   Kokkos::deep_copy(stf_array, h_stf_array);
//   Kokkos::deep_copy(ispec_array, h_ispec_array);

//   return;
// }
