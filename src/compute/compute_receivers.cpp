#include "compute/interface.hpp"
#include "globals.h"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "receiver/interface.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <vector>

specfem::compute::receivers::receivers(
    const std::vector<specfem::receivers::receiver *> &receivers,
    const std::vector<specfem::enums::seismogram::type> &stypes,
    const specfem::quadrature::quadrature *quadx,
    const specfem::quadrature::quadrature *quadz, const type_real xmax,
    const type_real xmin, const type_real zmax, const type_real zmin,
    const int max_sig_step, specfem::MPI::MPI *mpi) {

  // Get  sources which lie in processor
  std::vector<specfem::receivers::receiver *> my_receivers;
  for (auto &receiver : receivers) {
    if (receiver->get_islice() == mpi->get_rank()) {
      my_receivers.push_back(receiver);
    }
  }

  // allocate source array view
  this->receiver_array = specfem::kokkos::DeviceView4d<type_real>(
      "specfem::compute::receiver::receiver_array", my_receivers.size(),
      quadz->get_N(), quadx->get_N(), ndim);

  this->h_receiver_array = Kokkos::create_mirror_view(this->receiver_array);

  this->ispec_array = specfem::kokkos::DeviceView1d<int>(
      "specfem::compute::receivers::ispec_array", my_receivers.size());

  this->h_ispec_array = Kokkos::create_mirror_view(ispec_array);

  this->cos_recs = specfem::kokkos::DeviceView1d<type_real>(
      "specfem::compute::receivers::cos_recs", my_receivers.size());

  this->h_cos_recs = Kokkos::create_mirror_view(cos_recs);

  this->sin_recs = specfem::kokkos::DeviceView1d<type_real>(
      "specfem::compute::receivers::sin_recs", my_receivers.size());

  this->h_sin_recs = Kokkos::create_mirror_view(sin_recs);

  this->field = specfem::kokkos::DeviceView5d<type_real>(
      "specfem::compute::receivers::field", stypes.size(), my_receivers.size(),
      2, quadz->get_N(), quadx->get_N());

  this->seismogram = specfem::kokkos::DeviceView4d<type_real>(
      "specfem::compute::receivers::seismogram", max_sig_step, stypes.size(),
      my_receivers.size(), 2);

  this->h_seismogram = Kokkos::create_mirror_view(this->seismogram);

  // store source array for sources in my islice
  for (int irec = 0; irec < my_receivers.size(); irec++) {

    my_receivers[irec]->check_locations(xmax, xmin, zmax, zmin, mpi);

    auto sv_receiver_array = Kokkos::subview(
        this->h_receiver_array, irec, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    my_receivers[irec]->compute_receiver_array(quadx, quadz, sv_receiver_array);

    this->h_ispec_array(irec) = my_receivers[irec]->get_ispec();
    this->h_cos_recs(irec) = my_receivers[irec]->get_cosine();
    this->h_sin_recs(irec) = my_receivers[irec]->get_sine();
  }

  this->seismogram_types =
      specfem::kokkos::DeviceView1d<specfem::enums::seismogram::type>(
          "specfem::compute::receivers::seismogram_types", stypes.size());

  this->h_seismogram_types = Kokkos::create_mirror_view(this->seismogram_types);

  for (int i = 0; i < stypes.size(); i++) {
    this->h_seismogram_types(i) = stypes[i];
  }

  this->sync_views();

  return;
};

void specfem::compute::receivers::sync_views() {
  Kokkos::deep_copy(receiver_array, h_receiver_array);
  Kokkos::deep_copy(ispec_array, h_ispec_array);
  Kokkos::deep_copy(cos_recs, h_cos_recs);
  Kokkos::deep_copy(sin_recs, h_sin_recs);
  Kokkos::deep_copy(seismogram_types, h_seismogram_types);

  return;
}

void specfem::compute::receivers::sync_seismograms() {
  Kokkos::deep_copy(h_seismogram, seismogram);

  return;
}
