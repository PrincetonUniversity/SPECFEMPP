#include "../include/compute.h"
#include "../include/kokkos_abstractions.h"
#include "../include/quadrature.h"
#include "../include/source.h"
#include "../include/specfem_mpi.h"
#include <Kokkos-Core.cpp>
#include <vector>

specfem::compute::sources::sources(
    std::vector<specfem::sources::source *> sources,
    quadrature::quadrature &quadx, quadrature::quadrature &quadz, specfem::MPI
    : MPI *mpi) {

  // Get  sources which lie in processor
  std::vector<specfem::sources::source *> my_sources;
  for (auto &source : sources) {
    if (source->get_islice() == mpi->get_rank()) {
      my_sources.append(source);
    }
  }

  // allocate source array view
  this->source_array =
      specfem::HostView4d("specfem::compute::sources::source_array", my_sources,
                          quadz.get_N(), quadx.get_N(), ndim);

  // store source array for sources in my islice
  for (int isource = 0; isource < my_sources.size(); isource++) {
    auto sv_source_array = Kokkos::subview(
        this->source_array, isource, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    my_sources[isources]->compute_source_array(quadx, quadz, sv_source_array);
  }

  return;
};
