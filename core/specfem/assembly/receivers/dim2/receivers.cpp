#include "specfem/receivers.hpp"
#include "globals.h"
#include "kokkos_abstractions.h"
#include "specfem/algorithms.hpp"
#include "specfem/assembly.hpp"
#include "specfem/quadrature.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <vector>

specfem::assembly::receivers<specfem::dimension::type::dim2>::receivers(
    const int nspec, const int ngllz, const int ngllx, const int max_sig_step,
    const type_real dt, const type_real t0, const int nsteps_between_samples,
    const std::vector<std::shared_ptr<
        specfem::receivers::receiver<specfem::dimension::type::dim2> > >
        &receivers,
    const std::vector<specfem::wavefield::type> &stypes,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::mesh::tags<specfem::dimension::type::dim2> &tags,
    const specfem::assembly::element_types<specfem::dimension::type::dim2>
        &element_types)
    : nspec(nspec),
      lagrange_interpolant("specfem::assembly::receivers::lagrange_interpolant",
                           receivers.size(), mesh.element_grid.ngllz,
                           mesh.element_grid.ngllx),
      h_lagrange_interpolant(Kokkos::create_mirror_view(lagrange_interpolant)),
      elements("specfem::assembly::receivers::elements", receivers.size()),
      h_elements(Kokkos::create_mirror_view(elements)),
      element_types(element_types),
      specfem::assembly::receivers_impl::StationIterator(receivers.size(),
                                                         stypes),
      specfem::assembly::receivers_impl::SeismogramIterator<
          specfem::dimension::type::dim2>(receivers.size(), stypes.size(),
                                          max_sig_step, dt, t0,
                                          nsteps_between_samples) {

  // Validate and populate seismogram type mapping
  for (int isies = 0; isies < stypes.size(); ++isies) {
    auto seis_type = stypes[isies];

    if (seis_type != specfem::wavefield::type::displacement &&
        seis_type != specfem::wavefield::type::velocity &&
        seis_type != specfem::wavefield::type::acceleration &&
        seis_type != specfem::wavefield::type::pressure &&
        seis_type != specfem::wavefield::type::rotation &&
        seis_type != specfem::wavefield::type::intrinsic_rotation &&
        seis_type != specfem::wavefield::type::curl) {
      std::ostringstream message;
      message << "Error reading specfem receiver configuration.(" << __FILE__
              << ":" << __LINE__ << ")\n";
      message << "Unknown seismogram type: "
              << specfem::wavefield::to_string(seis_type) << "\n";
      message
          << "Valid seismogram types are: displacement, velocity, "
          << "acceleration, pressure, rotation, intrinsic_rotation, curl.\n";
      message << "Please check your configuration file.\n";
      throw std::runtime_error(message.str());
    }

    seismogram_type_map[seis_type] = isies;
  }

  for (int ireceiver = 0; ireceiver < receivers.size(); ++ireceiver) {
    const auto receiver = receivers[ireceiver];
    std::string station_name = receiver->get_station_name();
    std::string network_name = receiver->get_network_name();

    station_names_.push_back(station_name);
    network_names_.push_back(network_name);
    station_network_map[station_name][network_name] = ireceiver;
    const auto gcoord =
        specfem::point::global_coordinates<specfem::dimension::type::dim2>{
          receiver->get_x(), receiver->get_z()
        };
    const auto lcoord = specfem::algorithms::locate_point(gcoord, mesh);

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

        h_sine_receiver_angle(ireceiver) = std::sin(
            Kokkos::numbers::pi_v<type_real> / 180 * receiver->get_angle());

        h_cosine_receiver_angle(ireceiver) = std::cos(
            Kokkos::numbers::pi_v<type_real> / 180 * receiver->get_angle());
      }
    }
  }

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2),
       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                  ELASTIC_PSV_T),
       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT)),
      CAPTURE(elements, h_elements, receiver_indices, h_receiver_indices) {
        int count = 0;
        int index = 0;

        for (int ireceiver = 0; ireceiver < h_elements.extent(0); ++ireceiver) {
          int ispec = h_elements(ireceiver);
          if (element_types.get_medium_tag(ispec) == _medium_tag_ &&
              element_types.get_property_tag(ispec) == _property_tag_) {
            count++;
          }
        }

        _elements_ =
            IndexViewType("specfem::assembly::receivers::elements", count);
        _h_elements_ = Kokkos::create_mirror_view(_elements_);
        _receiver_indices_ =
            IndexViewType("specfem::assembly::receivers::elements", count);
        _h_receiver_indices_ = Kokkos::create_mirror_view(_receiver_indices_);

        for (int ireceiver = 0; ireceiver < h_elements.extent(0); ++ireceiver) {
          int ispec = h_elements(ireceiver);
          if (element_types.get_medium_tag(ispec) == _medium_tag_ &&
              element_types.get_property_tag(ispec) == _property_tag_) {
            _h_elements_(index) = ispec;
            _h_receiver_indices_(index) = ireceiver;
            index++;
          }
        }

        Kokkos::deep_copy(_elements_, _h_elements_);
        Kokkos::deep_copy(_receiver_indices_, _h_receiver_indices_);
      })

  Kokkos::deep_copy(lagrange_interpolant, h_lagrange_interpolant);
  Kokkos::deep_copy(elements, h_elements);

  return;
}

std::tuple<Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>,
           Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> >
specfem::assembly::receivers<specfem::dimension::type::dim2>::
    get_indices_on_host(
        const specfem::element::medium_tag medium_tag,
        const specfem::element::property_tag property_tag) const {

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2),
       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                  ELASTIC_PSV_T),
       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT)),
      CAPTURE(h_elements, h_receiver_indices) {
        if (medium_tag == _medium_tag_ && property_tag == _property_tag_) {
          return std::make_tuple(_h_elements_, _h_receiver_indices_);
        }
      })

  Kokkos::abort("Invalid medium or property tag. Please check the input "
                "parameters and try again.");
  return std::make_tuple(
      Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>(),
      Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>());
}

std::tuple<Kokkos::View<int *, Kokkos::DefaultExecutionSpace>,
           Kokkos::View<int *, Kokkos::DefaultExecutionSpace> >
specfem::assembly::receivers<specfem::dimension::type::dim2>::
    get_indices_on_device(
        const specfem::element::medium_tag medium_tag,
        const specfem::element::property_tag property_tag) const {

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2),
       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                  ELASTIC_PSV_T),
       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT)),
      CAPTURE(elements, receiver_indices) {
        if (medium_tag == _medium_tag_ && property_tag == _property_tag_) {
          return std::make_tuple(_elements_, _receiver_indices_);
        }
      })

  Kokkos::abort("Invalid medium or property tag. Please check the input "
                "parameters and try again.");
  return std::make_tuple(Kokkos::View<int *, Kokkos::DefaultExecutionSpace>(),
                         Kokkos::View<int *, Kokkos::DefaultExecutionSpace>());
}
