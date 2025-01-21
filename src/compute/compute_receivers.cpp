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

specfem::compute::receivers::receivers(
    const int nspec, const int ngllz, const int ngllx, const int max_sig_step,
    const type_real dt, const type_real t0, const int nsteps_between_samples,
    const std::vector<std::shared_ptr<specfem::receivers::receiver> >
        &receivers,
    const std::vector<specfem::enums::seismogram::type> &stypes,
    const specfem::compute::mesh &mesh,
    const specfem::mesh::tags<specfem::dimension::type::dim2> &tags,
    const specfem::compute::properties &properties)
    : lagrange_interpolant("specfem::compute::receivers::lagrange_interpolant",
                           receivers.size(), mesh.ngllz, mesh.ngllx),
      h_lagrange_interpolant(Kokkos::create_mirror_view(lagrange_interpolant)),
      elements("specfem::compute::receivers::elements", receivers.size()),
      h_elements(Kokkos::create_mirror_view(elements)),
      receiver_domain_index_mapping(
          "specfem::compute::receivers::receiver_domain_index_mapping", nspec),
      h_receiver_domain_index_mapping(
          Kokkos::create_mirror_view(receiver_domain_index_mapping)),
      impl::element_types(static_cast<impl::element_types>(properties)),
      impl::StationIterator(receivers.size(), stypes.size()),
      impl::SeismogramIterator(receivers.size(), stypes.size(), max_sig_step,
                               dt, t0, nsteps_between_samples) {

  for (int ispec = 0; ispec < nspec; ++ispec) {
    h_receiver_domain_index_mapping(ispec) = -1;
  }

  for (int isies = 0; isies < stypes.size(); ++isies) {
    auto seis_type = stypes[isies];
    switch (seis_type) {
    case specfem::enums::seismogram::type::displacement:
      seismogram_types[isies] = specfem::wavefield::type::displacement;
      seismogram_type_map[specfem::wavefield::type::displacement] = isies;
      break;
    case specfem::enums::seismogram::type::velocity:
      seismogram_types[isies] = specfem::wavefield::type::velocity;
      seismogram_type_map[specfem::wavefield::type::velocity] = isies;
      break;
    case specfem::enums::seismogram::type::acceleration:
      seismogram_types[isies] = specfem::wavefield::type::acceleration;
      seismogram_type_map[specfem::wavefield::type::acceleration] = isies;
      break;
    case specfem::enums::seismogram::type::pressure:
      seismogram_types[isies] = specfem::wavefield::type::pressure;
      seismogram_type_map[specfem::wavefield::type::pressure] = isies;
      break;
    default:
      throw std::runtime_error("Invalid seismogram type");
    }
  }

  for (int ireceiver = 0; ireceiver < receivers.size(); ++ireceiver) {
    const auto receiver = receivers[ireceiver];
    std::string station_name = receiver->get_station_name();
    std::string network_name = receiver->get_network_name();

    station_names[ireceiver] = station_name;
    network_names[ireceiver] = network_name;
    station_network_map[station_name][network_name] = ireceiver;
    const auto gcoord =
        specfem::point::global_coordinates<specfem::dimension::type::dim2>{
          receiver->get_x(), receiver->get_z()
        };
    const auto lcoord = specfem::algorithms::locate_point(gcoord, mesh);

    if (h_receiver_domain_index_mapping(lcoord.ispec) != -1) {
      throw std::runtime_error(
          "Multiple receivers are detected in the same element");
    }

    h_receiver_domain_index_mapping(lcoord.ispec) = ireceiver;
    h_elements(ireceiver) = lcoord.ispec;

    const auto xi = mesh.quadratures.gll.h_xi;
    const auto gamma = mesh.quadratures.gll.h_xi;

    auto [hxi_receiver, hpxi_receiver] =
        specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
            lcoord.xi, mesh.quadratures.gll.N, xi);

    auto [hgamma_receiver, hpgamma_receiver] =
        specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
            lcoord.gamma, mesh.quadratures.gll.N, gamma);

    for (int iz = 0; iz < mesh.ngllz; ++iz) {
      for (int ix = 0; ix < mesh.ngllx; ++ix) {
        type_real hlagrange = hxi_receiver(ix) * hgamma_receiver(iz);

        h_lagrange_interpolant(ireceiver, iz, ix, 0) = hlagrange;
        h_lagrange_interpolant(ireceiver, iz, ix, 1) = hlagrange;

        h_sine_receiver_angle(ireceiver) = std::sin(
            Kokkos::numbers::pi_v<type_real> / 180 * receiver->get_angle());

        h_cosine_receiver_angle(ireceiver) = std::cos(
            Kokkos::numbers::pi_v<type_real> / 180 * receiver->get_angle());
      }
    }
  }

  Kokkos::deep_copy(lagrange_interpolant, h_lagrange_interpolant);
  Kokkos::deep_copy(elements, h_elements);
  Kokkos::deep_copy(receiver_domain_index_mapping,
                    h_receiver_domain_index_mapping);

  return;
}

Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>
specfem::compute::receivers::get_elements_on_host(
    const specfem::element::medium_tag medium_tag,
    const specfem::element::property_tag property_tag) const {
  int nreceivers = h_elements.extent(0);

  int ntags = h_medium_tags.extent(0);

  int count = 0;
  for (int ireceiver = 0; ireceiver < nreceivers; ++ireceiver) {
    int ispec = h_elements(ireceiver);

    if (h_medium_tags(ispec) == medium_tag &&
        h_property_tags(ispec) == property_tag) {
      count++;
    }
  }

  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> elements(
      "specfem::compute::receivers::elements", count);

  count = 0;
  for (int ireceiver = 0; ireceiver < nreceivers; ++ireceiver) {
    int ispec = h_elements(ireceiver);

    if (h_medium_tags(ispec) == medium_tag &&
        h_property_tags(ispec) == property_tag) {
      elements(count) = ispec;
      count++;
    }
  }

  return elements;
}

Kokkos::View<int *, Kokkos::DefaultExecutionSpace>
specfem::compute::receivers::get_elements_on_device(
    const specfem::element::medium_tag medium_tag,
    const specfem::element::property_tag property_tag) const {

  const auto h_elements = get_elements_on_host(medium_tag, property_tag);

  Kokkos::View<int *, Kokkos::DefaultExecutionSpace> elements(
      "specfem::compute::receivers::elements", h_elements.extent(0));

  Kokkos::deep_copy(elements, h_elements);

  return elements;
}
