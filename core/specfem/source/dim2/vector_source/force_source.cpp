#include "enumerations/interface.hpp"
#include "globals.h"
#include "source_time_function/interface.hpp"
#include "specfem/source.hpp"
#include "specfem_setup.hpp"
#include "utilities/interface.hpp"
#include "yaml-cpp/yaml.h"
#include <cmath>

std::vector<specfem::element::medium_tag>
specfem::sources::force<specfem::dimension::type::dim2>::get_supported_media()
    const {
  return {
    specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::elastic_psv,
    specfem::element::medium_tag::elastic_psv_t,
    specfem::element::medium_tag::elastic_sh,
    specfem::element::medium_tag::poroelastic,
  };
}

specfem::kokkos::HostView1d<type_real>
specfem::sources::force<specfem::dimension::type::dim2>::get_force_vector()
    const {

  // Get the medium tag that the source is located in
  specfem::element::medium_tag medium_tag = this->get_medium_tag();

  // Declare the force vector
  specfem::kokkos::HostView1d<type_real> force_vector;

  // Convert angle to radians
  type_real angle_in_rad = this->angle * Kokkos::numbers::pi_v<type_real> /
                           static_cast<type_real>(180.0);

  // Acoustic
  if (medium_tag == specfem::element::medium_tag::acoustic) {
    force_vector = specfem::kokkos::HostView1d<type_real>("force_vector", 1);
    force_vector(0) = 1.0;
  }
  // Elastic SH
  else if (medium_tag == specfem::element::medium_tag::elastic_sh) {
    force_vector = specfem::kokkos::HostView1d<type_real>("force_vector", 1);
    force_vector(0) = 1.0;
  }
  // Elastic P-SV
  else if (medium_tag == specfem::element::medium_tag::elastic_psv) {
    force_vector = specfem::kokkos::HostView1d<type_real>("force_vector", 2);
    // angle measured clockwise vertical direction
    force_vector(0) = std::sin(angle_in_rad);
    force_vector(1) = std::cos(angle_in_rad);
  }
  // Poroelastic
  else if (medium_tag == specfem::element::medium_tag::poroelastic) {
    force_vector = specfem::kokkos::HostView1d<type_real>("force_vector", 4);
    force_vector(0) = std::sin(angle_in_rad);
    force_vector(1) = std::cos(angle_in_rad);
    force_vector(2) = std::sin(angle_in_rad);
    force_vector(3) = std::cos(angle_in_rad);
  }
  // Elastic P-SV-T
  else if (medium_tag == specfem::element::medium_tag::elastic_psv_t) {
    force_vector = specfem::kokkos::HostView1d<type_real>("force_vector", 3);
    force_vector(0) = std::sin(angle_in_rad);
    force_vector(1) = std::cos(angle_in_rad);
    force_vector(2) = static_cast<type_real>(0.0);
  } else {
    KOKKOS_ABORT_WITH_LOCATION("Force source array computation not "
                               "implemented for requested element type.");
  }

  return force_vector;
}

std::string
specfem::sources::force<specfem::dimension::type::dim2>::print() const {

  const auto gcoord = this->get_global_coordinates();

  std::ostringstream message;
  message << "- Force Source: \n"
          << "    Source Location: \n"
          << "      x = " << type_real(gcoord.x) << "\n"
          << "      z = " << type_real(gcoord.z) << "\n"
          << "    Source Time Function: \n"
          << this->forcing_function->print() << "\n";

  return message.str();
}

bool specfem::sources::force<specfem::dimension::type::dim2>::operator==(
    const specfem::sources::source<specfem::dimension::type::dim2> &other)
    const {

  // Try casting the other source to a force source
  const auto *other_source = dynamic_cast<
      const specfem::sources::force<specfem::dimension::type::dim2> *>(&other);

  // Check if cast was successful
  if (other_source == nullptr) {
    std::cout << "Other source is not a force object" << std::endl;
    return false;
  }
  const auto gcoord = this->get_global_coordinates();
  const auto other_gcoord = other_source->get_global_coordinates();

  return specfem::utilities::is_close(gcoord.x, other_gcoord.x) &&
         specfem::utilities::is_close(gcoord.z, other_gcoord.z) &&
         specfem::utilities::is_close(this->angle, other_source->angle) &&
         *(this->forcing_function) == *(other_source->forcing_function);
}
bool specfem::sources::force<specfem::dimension::type::dim2>::operator!=(
    const specfem::sources::source<specfem::dimension::type::dim2> &other)
    const {
  return !(*this == other);
}
