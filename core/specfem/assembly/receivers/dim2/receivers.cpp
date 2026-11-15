#include "specfem/receivers.hpp"

#include "specfem/algorithms.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/assembly/receivers.hpp"
#include "specfem/element.hpp"
#include "specfem/logger.hpp"
#include "specfem/mpi.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <array>
#include <map>
#include <string>
#include <vector>

specfem::assembly::receivers<specfem::element::dimension_tag::dim2>::receivers(
    const int nspec, const int ngllz, const int ngllx, const int max_sig_step,
    const type_real dt, const type_real t0, const int nsteps_between_samples,
    const std::vector<std::shared_ptr<
        specfem::receivers::receiver<specfem::element::dimension_tag::dim2>>>
        &receivers,
    const std::vector<specfem::enums::wavefield> &stypes,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh,
    const specfem::mesh::tags<specfem::element::dimension_tag::dim2> &tags,
    const specfem::assembly::element_types<
        specfem::element::dimension_tag::dim2> &element_types)
    : nspec(nspec),
      lagrange_interpolant("specfem::assembly::receivers::lagrange_interpolant",
                           receivers.size(), mesh.element_grid.ngllz,
                           mesh.element_grid.ngllx),
      h_lagrange_interpolant(Kokkos::create_mirror_view(lagrange_interpolant)),
      elements("specfem::assembly::receivers::elements", receivers.size()),
      h_elements(Kokkos::create_mirror_view(elements)),
      specfem::assembly::receivers_impl::StationIterator(receivers.size(),
                                                         stypes),
      specfem::assembly::receivers_impl::SeismogramIterator<
          specfem::element::dimension_tag::dim2>(
          receivers.size(),
          specfem::assembly::receivers_impl::SeismogramTypes(stypes),
          max_sig_step, dt, t0, nsteps_between_samples) {

  // MPI-aware receiver location with inside-preference: prefer elements where
  // xi/gamma ∈ [-1,1]; fall back to minimum distance if no rank has an inside
  // element. Uses locate_point for unified MPI communication.
  const int nreceivers = static_cast<int>(receivers.size());
  const int myrank = specfem::MPI::get_rank();

  std::vector<
      specfem::point::global_coordinates<specfem::element::dimension_tag::dim2>>
      gcoords;
  gcoords.reserve(nreceivers);
  for (int ireceiver = 0; ireceiver < nreceivers; ++ireceiver)
    gcoords.push_back(receivers[ireceiver]->get_global_coordinates());

  const auto located = specfem::algorithms::locate_point(gcoords, mesh);
  const auto &local_coords = located.local;
  const auto &partition_index_selected = located.partition_index;

  for (int ireceiver = 0; ireceiver < nreceivers; ++ireceiver) {
    const auto receiver = receivers[ireceiver];
    std::string station_name = receiver->get_station_name();
    std::string network_name = receiver->get_network_name();

    stations_.push_back({ network_name, station_name, 0, seismogram_types_ });
    station_network_map[station_name][network_name] = ireceiver;

    // Position-only information: every rank stores every receiver's matrix.
    const type_real angle_rad =
        Kokkos::numbers::pi_v<type_real> / 180 * receiver->get_angle();
    const type_real cos_a = std::cos(angle_rad);
    const type_real sin_a = std::sin(angle_rad);
    // 2D rotation matrix: R = [[cos, -sin], [sin, cos]]
    const std::array<std::array<type_real, 2>, 2> rotation_matrix = {
      { { { cos_a, -sin_a } }, { { sin_a, cos_a } } }
    };
    this->set_rotation_matrix(ireceiver, rotation_matrix);

    if (partition_index_selected[ireceiver] != myrank) {
      h_elements(ireceiver) = -1;
      continue;
    }

    const auto &lcoord = local_coords[ireceiver];
    h_elements(ireceiver) = lcoord.ispec;

    // Warn when the recovered local coordinates land outside the reference
    // element beyond a small tolerance (coordinate resolution is not exact).
    if (specfem::algorithms::outside(lcoord, type_real(1.001))) {
      specfem::Logger::warning(
          "Receiver " + network_name + "." + station_name +
              " located outside its element: " + lcoord.print(),
          /*root_only=*/false);
    }

    const auto xi = mesh.h_xi;
    const auto gamma = mesh.h_xi;

    auto [hxi_receiver, hpxi_receiver] =
        specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
            lcoord.xi, mesh.element_grid.ngllx, xi);

    auto [hgamma_receiver, hpgamma_receiver] =
        specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
            lcoord.gamma, mesh.element_grid.ngllx, gamma);

    for (int iz = 0; iz < mesh.element_grid.ngllz; ++iz) {
      for (int ix = 0; ix < mesh.element_grid.ngllx; ++ix) {
        type_real hlagrange = hxi_receiver(ix) * hgamma_receiver(iz);

        h_lagrange_interpolant(ireceiver, iz, ix, 0) = hlagrange;
        h_lagrange_interpolant(ireceiver, iz, ix, 1) = hlagrange;
      }
    }
  }

  // Build per-material receiver index stores using tag_dispatch::Storage.
  this->build_index_stores(h_elements, element_types);

  for (int i = 0; i < nreceivers; ++i)
    stations_[i].partition_index = partition_index_selected[i];

  for (int ireceiver = 0; ireceiver < nreceivers; ++ireceiver) {
    receivers[ireceiver]->set_partition_index(
        partition_index_selected[ireceiver]);
    receivers[ireceiver]->set_location_error(located.error[ireceiver]);
  }

  Kokkos::deep_copy(lagrange_interpolant, h_lagrange_interpolant);
  Kokkos::deep_copy(elements, h_elements);
  this->sync_rotation_matrices();

  return;
}
