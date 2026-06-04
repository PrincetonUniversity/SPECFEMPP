#include "specfem/receivers.hpp"

#include "specfem/algorithms.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/assembly/receivers.hpp"
#include "specfem/element.hpp"
#include "specfem/mpi.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <vector>

specfem::assembly::receivers<specfem::element::dimension_tag::dim2>::receivers(
    const int nspec, const int ngllz, const int ngllx, const int max_sig_step,
    const type_real dt, const type_real t0, const int nsteps_between_samples,
    const std::vector<std::shared_ptr<
        specfem::receivers::receiver<specfem::element::dimension_tag::dim2> > >
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
          receivers.size(), stypes.size(), max_sig_step, dt, t0,
          nsteps_between_samples) {

  // Validate and populate seismogram type mapping
  for (int isies = 0; isies < stypes.size(); ++isies) {
    auto seis_type = stypes[isies];

    if (seis_type != specfem::enums::wavefield::displacement &&
        seis_type != specfem::enums::wavefield::velocity &&
        seis_type != specfem::enums::wavefield::acceleration &&
        seis_type != specfem::enums::wavefield::pressure &&
        seis_type != specfem::enums::wavefield::rotation &&
        seis_type != specfem::enums::wavefield::intrinsic_rotation &&
        seis_type != specfem::enums::wavefield::curl) {
      std::ostringstream message;
      message << "Error reading specfem receiver configuration.(" << __FILE__
              << ":" << __LINE__ << ")\n";
      message << "Unknown seismogram type: "
              << specfem::enums::to_string(seis_type) << "\n";
      message
          << "Valid seismogram types are: displacement, velocity, "
          << "acceleration, pressure, rotation, intrinsic_rotation, curl.\n";
      message << "Please check your configuration file.\n";
      throw std::runtime_error(message.str());
    }

    seismogram_type_map[seis_type] = isies;
  }

  // MPI-aware receiver location with inside-preference: prefer elements where
  // xi/gamma ∈ [-1,1]; fall back to minimum distance if no rank has an inside
  // element. Uses locate_point for unified MPI communication.
  const int nreceivers = static_cast<int>(receivers.size());
  const int myrank = specfem::MPI::get_rank();

  std::vector<specfem::point::global_coordinates<
      specfem::element::dimension_tag::dim2> >
      gcoords;
  gcoords.reserve(nreceivers);
  for (int ireceiver = 0; ireceiver < nreceivers; ++ireceiver)
    gcoords.push_back(receivers[ireceiver]->get_global_coordinates());

  auto [local_coords, partition_index_selected] =
      specfem::algorithms::locate_point(gcoords, mesh);

  for (int ireceiver = 0; ireceiver < nreceivers; ++ireceiver) {
    const auto receiver = receivers[ireceiver];
    std::string station_name = receiver->get_station_name();
    std::string network_name = receiver->get_network_name();

    stations_.push_back({ network_name, station_name, 0, seismogram_types_ });
    station_network_map[station_name][network_name] = ireceiver;

    if (partition_index_selected[ireceiver] != myrank) {
      h_elements(ireceiver) = -1;
      continue;
    }

    const auto &lcoord = local_coords[ireceiver];
    h_elements(ireceiver) = lcoord.ispec;

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

        const type_real angle_rad =
            Kokkos::numbers::pi_v<type_real> / 180 * receiver->get_angle();
        const type_real cos_a = std::cos(angle_rad);
        const type_real sin_a = std::sin(angle_rad);
        // 2D rotation matrix: R = [[cos, -sin], [sin, cos]]
        h_rotation_matrices(ireceiver, 0, 0) = cos_a;
        h_rotation_matrices(ireceiver, 0, 1) = -sin_a;
        h_rotation_matrices(ireceiver, 1, 0) = sin_a;
        h_rotation_matrices(ireceiver, 1, 1) = cos_a;
      }
    }
  }

  // Build per-material receiver index stores using tag_dispatch::Storage.
  this->build_index_stores(h_elements, element_types);

  for (int i = 0; i < nreceivers; ++i)
    stations_[i].partition_index = partition_index_selected[i];

  for (int ireceiver = 0; ireceiver < nreceivers; ++ireceiver)
    receivers[ireceiver]->set_partition_index(
        partition_index_selected[ireceiver]);

  Kokkos::deep_copy(lagrange_interpolant, h_lagrange_interpolant);
  Kokkos::deep_copy(elements, h_elements);

  return;
}
