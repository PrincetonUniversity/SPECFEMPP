#include "enumerations/interface.hpp"
#include "globals.h"
#include "source_time_function/interface.hpp"
#include "specfem/source.hpp"
#include "specfem_setup.hpp"
#include "utilities/interface.hpp"
#include "yaml-cpp/yaml.h"
#include <cmath>

// Static member definitions
const std::string
    specfem::sources::force<specfem::dimension::type::dim3>::name =
        "3-D force source";

std::vector<specfem::element::medium_tag>
specfem::sources::force<specfem::dimension::type::dim3>::get_supported_media()
    const {
  return { specfem::element::medium_tag::acoustic,
           specfem::element::medium_tag::elastic };
}

specfem::kokkos::HostView1d<type_real>
specfem::sources::force<specfem::dimension::type::dim3>::get_force_vector()
    const {

  // Get the medium tag that the source is located in
  specfem::element::medium_tag medium_tag = this->get_medium_tag();

  // Declare the force vector
  specfem::kokkos::HostView1d<type_real> force_vector;

  // Acoustic
  if (medium_tag == specfem::element::medium_tag::acoustic) {
    force_vector = specfem::kokkos::HostView1d<type_real>("force_vector", 1);
    force_vector(0) = fx;
  }
  // Elastic P-SV
  else if (medium_tag == specfem::element::medium_tag::elastic) {
    force_vector = specfem::kokkos::HostView1d<type_real>("force_vector", 3);
    force_vector(0) = fx;
    force_vector(1) = fy;
    force_vector(2) = fz;
  } else {
    KOKKOS_ABORT_WITH_LOCATION("3-D force source array computation not "
                               "implemented for requested element type.");
  }

  return force_vector;
}

std::string
specfem::sources::force<specfem::dimension::type::dim3>::print() const {

  std::ostringstream message;
  message << "- Force Source: \n"
          << "    Source Location: \n"
          << "      x = " << type_real(this->x) << "\n"
          << "      y = " << type_real(this->y) << "\n"
          << "      z = " << type_real(this->z) << "\n"
          << "    Force Vector: \n"
          << "      fx = " << type_real(this->fx) << "\n"
          << "      fy = " << type_real(this->fy) << "\n"
          << "      fz = " << type_real(this->fz) << "\n"
          << "    Source Time Function: \n"
          << this->forcing_function->print() << "\n";

  return message.str();
}

bool specfem::sources::force<specfem::dimension::type::dim3>::operator==(
    const specfem::sources::source<specfem::dimension::type::dim3> &other)
    const {

  // Try casting the other source to a force source
  const auto *other_source = dynamic_cast<
      const specfem::sources::force<specfem::dimension::type::dim3> *>(&other);

  // Check if cast was successful
  if (other_source == nullptr) {
    std::cout << "Other source is not a force object" << std::endl;
    return false;
  }

  return specfem::utilities::almost_equal(this->x, other_source->x) &&
         specfem::utilities::almost_equal(this->y, other_source->y) &&
         specfem::utilities::almost_equal(this->z, other_source->z) &&
         specfem::utilities::almost_equal(this->fx, other_source->fx) &&
         specfem::utilities::almost_equal(this->fy, other_source->fy) &&
         specfem::utilities::almost_equal(this->fz, other_source->fz) &&
         *(this->forcing_function) == *(other_source->forcing_function);
}
bool specfem::sources::force<specfem::dimension::type::dim3>::operator!=(
    const specfem::sources::source<specfem::dimension::type::dim3> &other)
    const {
  return !(*this == other);
}
