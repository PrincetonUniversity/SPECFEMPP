#include "specfem/receivers.hpp"

#include "specfem/algorithms.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/assembly/receivers.hpp"
#include "specfem/assembly/resolve_coordinates.hpp"
#include "specfem/element.hpp"
#include "specfem/mpi.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <vector>

specfem::assembly::receivers<specfem::element::dimension_tag::dim3>::receivers(
    const int max_sig_step, const type_real dt, const type_real t0,
    const int nsteps_between_samples,
    const std::vector<std::shared_ptr<
        specfem::receivers::receiver<specfem::element::dimension_tag::dim3>>>
        &receivers,
    const std::vector<specfem::enums::wavefield> &stypes,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const specfem::mesh::tags<specfem::element::dimension_tag::dim3> &tags,
    const specfem::assembly::element_types<
        specfem::element::dimension_tag::dim3> &element_types)
    : lagrange_interpolant("specfem::assembly::receivers::lagrange_interpolant",
                           receivers.size(), mesh.element_grid.ngllz,
                           mesh.element_grid.nglly, mesh.element_grid.ngllx, 3),
      h_lagrange_interpolant(Kokkos::create_mirror_view(lagrange_interpolant)),
      elements("specfem::assembly::receivers::elements", receivers.size()),
      h_elements(Kokkos::create_mirror_view(elements)),
      specfem::assembly::receivers_impl::StationIterator(receivers.size(),
                                                         stypes),
      specfem::assembly::receivers_impl::SeismogramIterator<
          specfem::element::dimension_tag::dim3>(
          receivers.size(), stypes.size(), max_sig_step, dt, t0,
          nsteps_between_samples) {

  // Discretization from the mesh
  this->nspec = mesh.nspec;
  const auto &element_grid = mesh.element_grid;

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
  // xi/eta/gamma ∈ [-1,1]; fall back to minimum distance if no rank has an
  // inside element. Uses locate_point for unified MPI communication.
  const int nreceivers = static_cast<int>(receivers.size());
  const int myrank = specfem::MPI::get_rank();

  // Resolve any generic coordinates to global coordinates using mesh context.
  for (int ireceiver = 0; ireceiver < nreceivers; ++ireceiver) {
    if (const auto *coords = receivers[ireceiver]->get_coordinates()) {
      auto gc = specfem::assembly::resolve_coordinates(*coords, mesh);
      receivers[ireceiver]->set_global_coordinates(gc);
    }
  }

  std::vector<
      specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>>
      gcoords;
  gcoords.reserve(nreceivers);
  for (int ireceiver = 0; ireceiver < nreceivers; ++ireceiver)
    gcoords.push_back(receivers[ireceiver]->get_global_coordinates());

  auto [local_coords, islice_selected] =
      specfem::algorithms::locate_point(gcoords, mesh);

  for (int ireceiver = 0; ireceiver < nreceivers; ++ireceiver) {
    const auto receiver = receivers[ireceiver];
    std::string station_name = receiver->get_station_name();
    std::string network_name = receiver->get_network_name();

    station_names_.push_back(station_name);
    network_names_.push_back(network_name);
    station_network_map[station_name][network_name] = ireceiver;

    if (islice_selected[ireceiver] != myrank) {
      h_elements(ireceiver) = -1;
      continue;
    }

    const auto &lcoord = local_coords[ireceiver];
    h_elements(ireceiver) = lcoord.ispec;

    const auto xi = mesh.h_xi;
    const auto eta = mesh.h_xi;   // Use same as dim2 pattern
    const auto gamma = mesh.h_xi; // Use same as dim2 pattern

    auto [hxi_receiver, hpxi_receiver] =
        specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
            lcoord.xi, element_grid.ngllx, xi);

    auto [heta_receiver, hpeta_receiver] =
        specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
            lcoord.eta, element_grid.nglly, eta);

    auto [hgamma_receiver, hpgamma_receiver] =
        specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
            lcoord.gamma, element_grid.ngllz, gamma);

    for (int iz = 0; iz < element_grid.ngllz; ++iz) {
      for (int iy = 0; iy < element_grid.nglly; ++iy) {
        for (int ix = 0; ix < element_grid.ngllx; ++ix) {
          type_real hlagrange =
              hxi_receiver(ix) * heta_receiver(iy) * hgamma_receiver(iz);

          h_lagrange_interpolant(ireceiver, iz, iy, ix, 0) = hlagrange;
          h_lagrange_interpolant(ireceiver, iz, iy, ix, 1) = hlagrange;
          h_lagrange_interpolant(ireceiver, iz, iy, ix, 2) = hlagrange;
        }
      }
    }

    // Initialize rotation matrix to identity for this receiver
    // In the future, this could be set based on surface normal or user
    // specification
    std::array<std::array<type_real, 3>, 3> identity_matrix = {
      { { { 1.0, 0.0, 0.0 } }, { { 0.0, 1.0, 0.0 } }, { { 0.0, 0.0, 1.0 } } }
    };
    this->set_rotation_matrix(ireceiver, identity_matrix);
  }

  // Build per-material receiver index stores using tag_dispatch::Storage.
  this->build_index_stores(h_elements, element_types);

  receiver_islice_.assign(islice_selected.begin(), islice_selected.end());

  for (int ireceiver = 0; ireceiver < nreceivers; ++ireceiver)
    receivers[ireceiver]->set_islice(islice_selected[ireceiver]);

  Kokkos::deep_copy(lagrange_interpolant, h_lagrange_interpolant);
  Kokkos::deep_copy(elements, h_elements);

  return;
}
