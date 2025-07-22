#include "cosserat_force_source.hpp"
#include "enumerations/interface.hpp"
#include "globals.h"
#include "source_time_function/interface.hpp"
#include "specfem_setup.hpp"
#include "utilities/interface.hpp"
#include <cmath>

// Static member definitions
const std::string
    specfem::sources::cosserat_force<specfem::dimension::type::dim2>::name =
        "2D cosserat force source";

std::vector<specfem::element::medium_tag> specfem::sources::cosserat_force<
    specfem::dimension::type::dim2>::get_supported_media() const {
  return { specfem::element::medium_tag::elastic_psv_t };
}

specfem::kokkos::HostView1d<type_real> specfem::sources::cosserat_force<
    specfem::dimension::type::dim2>::get_force_vector() const {

  // Get the medium tag that the source is located in
  specfem::element::medium_tag medium_tag = this->get_medium_tag();

  // Declare the force vector
  specfem::kokkos::HostView1d<type_real> force_vector;

  // Convert angle to radians
  type_real angle_in_rad = this->angle * Kokkos::numbers::pi_v<type_real> /
                           static_cast<type_real>(180.0);

  // Only supporting elastic_psv_t medium for Cosserat force sources
  if (medium_tag == specfem::element::medium_tag::elastic_psv_t) {
    force_vector = specfem::kokkos::HostView1d<type_real>("force_vector", 3);
    force_vector(0) = this->f * std::sin(angle_in_rad);
    force_vector(1) =
        static_cast<type_real>(-1.0) * this->f * std::cos(angle_in_rad);
    force_vector(2) = this->fc;
  } else {
    KOKKOS_ABORT_WITH_LOCATION("Cosserat force source array computation not "
                               "implemented for requested element type.");
  }

  return force_vector;
}

std::string
specfem::sources::cosserat_force<specfem::dimension::type::dim2>::print()
    const {

  std::ostringstream message;
  message << "- Cosserat Force Source: \n"
          << "    Source Location: \n"
          << "      x = " << type_real(this->x) << "\n"
          << "      z = " << type_real(this->z) << "\n"
          << "    Source Angle: " << type_real(this->angle) << "\n"
          << "    Source f: " << type_real(this->f) << "\n"
          << "    Source fc: " << type_real(this->fc) << "\n"
          << "    Source Time Function: \n"
          << this->forcing_function->print() << "\n";

  return message.str();
}

bool specfem::sources::cosserat_force<specfem::dimension::type::dim2>::
operator==(const specfem::sources::source<specfem::dimension::type::dim2>
               &other) const {

  // Try casting the other source to a cosserat_force source
  const auto *other_source = dynamic_cast<
      const specfem::sources::cosserat_force<specfem::dimension::type::dim2> *>(
      &other);

  // Check if cast was successful
  if (other_source == nullptr) {
    std::cout << "Other source is not a cosserat_force object" << std::endl;
    return false;
  }

  bool internal =
      specfem::utilities::almost_equal(this->f, other_source->f) &&
      specfem::utilities::almost_equal(this->x, other_source->x) &&
      specfem::utilities::almost_equal(this->z, other_source->z) &&
      specfem::utilities::almost_equal(this->angle, other_source->angle);

  if (!internal) {
    std::cout << "Cosserat force sources not equal" << std::endl;
  }

  return internal &&
         (*(this->forcing_function) == *(other_source->forcing_function));
}
bool specfem::sources::cosserat_force<specfem::dimension::type::dim2>::
operator!=(const specfem::sources::source<specfem::dimension::type::dim2>
               &other) const {
  return !(*this == other);
}
