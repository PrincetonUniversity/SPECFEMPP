#include "../include/compute.h"
#include "../include/kokkos_abstractions.h"
#include "../include/quadrature.h"
#include "../include/source.h"
#include "../include/specfem_mpi.h"
#include <Kokkos_Core.hpp>
#include <vector>

specfem::compute::sources::sources(
    std::vector<specfem::sources::source *> sources,
    specfem::quadrature::quadrature &quadx,
    specfem::quadrature::quadrature &quadz, specfem::TimeScheme::TimeScheme *it,
    specfem::MPI::MPI *mpi) {

  // Get  sources which lie in processor
  std::vector<specfem::sources::source *> my_sources;
  for (auto &source : sources) {
    if (source->get_islice() == mpi->get_rank()) {
      my_sources.push_back(source);
    }
  }

  // allocate source array view
  this->source_array = specfem::HostView4d<type_real>(
      "specfem::compute::sources::source_array", my_sources.size(),
      quadz.get_N(), quadx.get_N(), ndim);

  this->stf_array =
      specfem::HostView2d<type_real>("specfem::compute::sources::stf_array",
                                     it->get_max_time(), my_sources.size());

  // store source array for sources in my islice
  for (int isource = 0; isource < my_sources.size(); isource++) {
    auto sv_source_array = Kokkos::subview(
        this->source_array, isource, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    my_sources[isource]->compute_source_array(quadx, quadz, sv_source_array);
    auto sv_stf_array = Kokkos::subview(this->stf_array, Kokkos::ALL, isource);
    my_sources[isource]->compute_stf(sv_stf_array, it);
  }

  return;
};
