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
    const specfem::compute::element_types &element_types)
    : nspec(nspec),
      lagrange_interpolant("specfem::compute::receivers::lagrange_interpolant",
                           receivers.size(), mesh.ngllz, mesh.ngllx),
      h_lagrange_interpolant(Kokkos::create_mirror_view(lagrange_interpolant)),
      elements("specfem::compute::receivers::elements", receivers.size()),
      h_elements(Kokkos::create_mirror_view(elements)),
      element_types(element_types), impl::StationIterator(receivers.size(),
                                                          stypes.size()),
      impl::SeismogramIterator(receivers.size(), stypes.size(), max_sig_step,
                               dt, t0, nsteps_between_samples) {

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

#define COUNT_RECEIVERS_PER_MATERIAL_SYSTEM(DIMENTION_TAG, MEDIUM_TAG,         \
                                            PROPERTY_TAG)                      \
  int CREATE_VARIABLE_NAME(count, GET_NAME(DIMENTION_TAG),                     \
                           GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG)) = 0;  \
  for (int ireceiver = 0; ireceiver < h_elements.extent(0); ++ireceiver) {     \
    int ispec = h_elements(ireceiver);                                         \
    if (element_types.get_medium_tag(ispec) == GET_TAG(MEDIUM_TAG) &&          \
        element_types.get_property_tag(ispec) == GET_TAG(PROPERTY_TAG)) {      \
      CREATE_VARIABLE_NAME(count, GET_NAME(DIMENTION_TAG),                     \
                           GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG))       \
      ++;                                                                      \
    }                                                                          \
  }

  CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
      COUNT_RECEIVERS_PER_MATERIAL_SYSTEM,
      WHERE(DIMENSION_TAG_DIM2)
          WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ACOUSTIC)
              WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC))

#undef COUNT_RECEIVERS_PER_MATERIAL_SYSTEM

#define ALLOCATE_RECEIVERS_PER_MATERIAL_SYSTEM(DIMENTION_TAG, MEDIUM_TAG,      \
                                               PROPERTY_TAG)                   \
  CREATE_VARIABLE_NAME(elements, GET_NAME(DIMENTION_TAG),                      \
                       GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG)) =         \
      IndexViewType("specfem::compute::receivers::elements",                   \
                    CREATE_VARIABLE_NAME(count, GET_NAME(DIMENTION_TAG),       \
                                         GET_NAME(MEDIUM_TAG),                 \
                                         GET_NAME(PROPERTY_TAG)));             \
  CREATE_VARIABLE_NAME(h_elements, GET_NAME(DIMENTION_TAG),                    \
                       GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG)) =         \
      Kokkos::create_mirror_view(                                              \
          CREATE_VARIABLE_NAME(elements, GET_NAME(DIMENTION_TAG),              \
                               GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG))); \
  CREATE_VARIABLE_NAME(receiver_indices, GET_NAME(DIMENTION_TAG),              \
                       GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG)) =         \
      IndexViewType("specfem::compute::receivers::elements",                   \
                    CREATE_VARIABLE_NAME(count, GET_NAME(DIMENTION_TAG),       \
                                         GET_NAME(MEDIUM_TAG),                 \
                                         GET_NAME(PROPERTY_TAG)));             \
  CREATE_VARIABLE_NAME(h_receiver_indices, GET_NAME(DIMENTION_TAG),            \
                       GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG)) =         \
      Kokkos::create_mirror_view(                                              \
          CREATE_VARIABLE_NAME(receiver_indices, GET_NAME(DIMENTION_TAG),      \
                               GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG)));

  CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
      ALLOCATE_RECEIVERS_PER_MATERIAL_SYSTEM,
      WHERE(DIMENSION_TAG_DIM2)
          WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ACOUSTIC)
              WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC))

#undef ALLOCATE_RECEIVERS_PER_MATERIAL_SYSTEM

#define ASSIGN_RECEIVERS_PER_MATERIAL_SYSTEM(DIMENTION_TAG, MEDIUM_TAG,        \
                                             PROPERTY_TAG)                     \
  int CREATE_VARIABLE_NAME(index, GET_NAME(DIMENTION_TAG),                     \
                           GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG)) = 0;  \
  for (int ireceiver = 0; ireceiver < h_elements.extent(0); ++ireceiver) {     \
    int ispec = h_elements(ireceiver);                                         \
    if (element_types.get_medium_tag(ispec) == GET_TAG(MEDIUM_TAG) &&          \
        element_types.get_property_tag(ispec) == GET_TAG(PROPERTY_TAG)) {      \
      CREATE_VARIABLE_NAME(h_elements, GET_NAME(DIMENTION_TAG),                \
                           GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG))       \
      (CREATE_VARIABLE_NAME(index, GET_NAME(DIMENTION_TAG),                    \
                            GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG))) =   \
          ispec;                                                               \
      CREATE_VARIABLE_NAME(h_receiver_indices, GET_NAME(DIMENTION_TAG),        \
                           GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG))       \
      (CREATE_VARIABLE_NAME(index, GET_NAME(DIMENTION_TAG),                    \
                            GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG))) =   \
          ireceiver;                                                           \
      CREATE_VARIABLE_NAME(index, GET_NAME(DIMENTION_TAG),                     \
                           GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG))       \
      ++;                                                                      \
    }                                                                          \
  }                                                                            \
  Kokkos::deep_copy(                                                           \
      CREATE_VARIABLE_NAME(elements, GET_NAME(DIMENTION_TAG),                  \
                           GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG)),      \
      CREATE_VARIABLE_NAME(h_elements, GET_NAME(DIMENTION_TAG),                \
                           GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG)));     \
  Kokkos::deep_copy(                                                           \
      CREATE_VARIABLE_NAME(receiver_indices, GET_NAME(DIMENTION_TAG),          \
                           GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG)),      \
      CREATE_VARIABLE_NAME(h_receiver_indices, GET_NAME(DIMENTION_TAG),        \
                           GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG)));

  CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
      ASSIGN_RECEIVERS_PER_MATERIAL_SYSTEM,
      WHERE(DIMENSION_TAG_DIM2)
          WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ACOUSTIC)
              WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC))

#undef ASSIGN_RECEIVERS_PER_MATERIAL_SYSTEM

  Kokkos::deep_copy(lagrange_interpolant, h_lagrange_interpolant);
  Kokkos::deep_copy(elements, h_elements);

  return;
}

std::tuple<Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>,
           Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> >
specfem::compute::receivers::get_indices_on_host(
    const specfem::element::medium_tag medium_tag,
    const specfem::element::property_tag property_tag) const {

#define RETURN_VALUE(DIMENTION_TAG, MEDIUM_TAG, PROPERTY_TAG)                  \
  if (medium_tag == GET_TAG(MEDIUM_TAG) &&                                     \
      property_tag == GET_TAG(PROPERTY_TAG)) {                                 \
    return std::make_tuple(                                                    \
        CREATE_VARIABLE_NAME(h_elements, GET_NAME(DIMENTION_TAG),              \
                             GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG)),    \
        CREATE_VARIABLE_NAME(h_receiver_indices, GET_NAME(DIMENTION_TAG),      \
                             GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG)));   \
  }

  CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
      RETURN_VALUE, WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC_SV,
                                                    MEDIUM_TAG_ACOUSTIC)
                        WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC))

#undef RETURN_VALUE
}

std::tuple<Kokkos::View<int *, Kokkos::DefaultExecutionSpace>,
           Kokkos::View<int *, Kokkos::DefaultExecutionSpace> >
specfem::compute::receivers::get_indices_on_device(
    const specfem::element::medium_tag medium_tag,
    const specfem::element::property_tag property_tag) const {

#define RETURN_VALUE(DIMENTION_TAG, MEDIUM_TAG, PROPERTY_TAG)                  \
  if (medium_tag == GET_TAG(MEDIUM_TAG) &&                                     \
      property_tag == GET_TAG(PROPERTY_TAG)) {                                 \
    return std::make_tuple(                                                    \
        CREATE_VARIABLE_NAME(elements, GET_NAME(DIMENTION_TAG),                \
                             GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG)),    \
        CREATE_VARIABLE_NAME(receiver_indices, GET_NAME(DIMENTION_TAG),        \
                             GET_NAME(MEDIUM_TAG), GET_NAME(PROPERTY_TAG)));   \
  }

  CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
      RETURN_VALUE, WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC_SV,
                                                    MEDIUM_TAG_ACOUSTIC)
                        WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC))

#undef RETURN_VALUE
}
