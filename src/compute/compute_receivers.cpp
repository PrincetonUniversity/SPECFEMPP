#include "algorithms/locate_point.hpp"
#include "compute/interface.hpp"
#include "globals.h"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "receiver/interface.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <vector>

specfem::compute::receivers::receivers(const int nreceivers,
                                       const int max_sig_step, const int N,
                                       const int n_seis_types)
    : receiver_array("specfem::compute::receiver::receiver_array", nreceivers,
                     ndim, N, N),
      h_receiver_array(Kokkos::create_mirror_view(receiver_array)),
      ispec_array("specfem::compute::receivers::ispec_array", nreceivers),
      h_ispec_array(Kokkos::create_mirror_view(ispec_array)),
      cos_recs("specfem::compute::receivers::cos_recs", nreceivers),
      h_cos_recs(Kokkos::create_mirror_view(cos_recs)),
      sin_recs("specfem::compute::receivers::sin_recs", nreceivers),
      h_sin_recs(Kokkos::create_mirror_view(sin_recs)),
      seismogram("specfem::compute::receivers::seismogram", max_sig_step,
                 n_seis_types, nreceivers, ndim),
      h_seismogram(Kokkos::create_mirror_view(seismogram)),
      seismogram_types("specfem::compute::receivers::seismogram_types",
                       n_seis_types),
      h_seismogram_types(Kokkos::create_mirror_view(seismogram_types)),
      receiver_field("specfem::compute::receivers::receiver_field",
                     max_sig_step, nreceivers, n_seis_types, ndim, N, N),
      h_receiver_field(Kokkos::create_mirror_view(receiver_field)) {}

specfem::compute::receivers::receivers(
    const int max_sig_step,
    const std::vector<std::shared_ptr<specfem::receivers::receiver> >
        &receivers,
    const std::vector<specfem::enums::seismogram::type> &stypes,
    const specfem::compute::mesh &mesh) {

  const int nreceivers = receivers.size();
  const int N = mesh.quadratures.gll.N;
  const int n_seis_types = stypes.size();

  *this =
      specfem::compute::receivers(nreceivers, max_sig_step, N, n_seis_types);

  for (int irec = 0; irec < nreceivers; irec++) {
    auto sv_receiver_array = Kokkos::subview(
        this->h_receiver_array, irec, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    receivers[irec]->compute_receiver_array(mesh, sv_receiver_array);

    specfem::point::gcoord2 coord = { receivers[irec]->get_x(),
                                      receivers[irec]->get_z() };
    auto lcoord = specfem::algorithms::locate_point(coord, mesh);

    this->h_ispec_array(irec) = lcoord.ispec;
    const auto angle = receivers[irec]->get_angle();

    this->h_cos_recs(irec) =
        std::cos(Kokkos::numbers::pi_v<type_real> / 180 * angle);
    this->h_sin_recs(irec) =
        std::sin(Kokkos::numbers::pi_v<type_real> / 180 * angle);
  }

  for (int i = 0; i < n_seis_types; i++) {
    this->h_seismogram_types(i) = stypes[i];
  }

  Kokkos::deep_copy(receiver_array, h_receiver_array);
  Kokkos::deep_copy(ispec_array, h_ispec_array);
  Kokkos::deep_copy(cos_recs, h_cos_recs);
  Kokkos::deep_copy(sin_recs, h_sin_recs);
  Kokkos::deep_copy(seismogram_types, h_seismogram_types);

  return;
};

void specfem::compute::receivers::sync_seismograms() {
  Kokkos::deep_copy(h_seismogram, seismogram);

  return;
}
