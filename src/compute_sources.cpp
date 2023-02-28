#include "../include/compute.h"
#include "../include/kokkos_abstractions.h"
#include "../include/quadrature.h"
#include "../include/source.h"
#include "../include/specfem_mpi.h"
#include <Kokkos_Core.hpp>
#include <vector>

specfem::compute::sources::sources(
    std::vector<specfem::sources::source *> sources,
    const specfem::quadrature::quadrature &quadx,
    const specfem::quadrature::quadrature &quadz, const type_real xmax,
    const type_real xmin, const type_real zmax, const type_real zmin,
    specfem::MPI::MPI *mpi) {

  // Get  sources which lie in processor
  std::vector<specfem::sources::source *> my_sources;
  for (auto &source : sources) {
    if (source->get_islice() == mpi->get_rank()) {
      my_sources.push_back(source);
    }
  }

  // allocate source array view
  this->source_array = specfem::kokkos::DeviceView4d<type_real>(
      "specfem::compute::sources::source_array", my_sources.size(),
      quadz.get_N(), quadx.get_N(), ndim);

  this->h_source_array = Kokkos::create_mirror_view(this->source_array);

  this->stf_array =
      specfem::kokkos::DeviceView1d<specfem::forcing_function::stf_storage>(
          "specfem::compute::sources::stf_array", my_sources.size());

  this->h_stf_array = Kokkos::create_mirror_view(this->stf_array);

  this->ispec_array = specfem::kokkos::DeviceView1d<int>(
      "specfem::compute::sources::ispec_array", my_sources.size());

  this->h_ispec_array = Kokkos::create_mirror_view(ispec_array);

  // store source array for sources in my islice
  for (int isource = 0; isource < my_sources.size(); isource++) {

    my_sources[isource]->check_locations(xmax, xmin, zmax, zmin, mpi);

    auto sv_source_array = Kokkos::subview(
        this->h_source_array, isource, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    my_sources[isource]->compute_source_array(quadx, quadz, sv_source_array);

    this->h_stf_array(isource).T = my_sources[isource]->get_stf();
    this->h_ispec_array(isource) = my_sources[isource]->get_ispec();
  }

  this->sync_views();

  return;
};

void specfem::compute::sources::sync_views() {
  Kokkos::deep_copy(source_array, h_source_array);
  Kokkos::deep_copy(stf_array, h_stf_array);
  Kokkos::deep_copy(ispec_array, h_ispec_array);

  return;
}
