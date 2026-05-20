
#include "specfem/enums.hpp"
#include "specfem/setup.hpp"
#include "specfem/source.hpp"
#include "specfem/source_time_functions.hpp"
#include "specfem/utilities.hpp"
#include <cmath>

std::vector<specfem::element::medium_tag> specfem::sources::cosserat_force<
    specfem::element::dimension_tag::dim3>::get_supported_media() const {
  return { specfem::element::medium_tag::elastic_spin };
}

Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>
specfem::sources::cosserat_force<
    specfem::element::dimension_tag::dim3>::get_force_vector() const {

  // Get the medium tag that the source is located in
  specfem::element::medium_tag medium_tag = this->get_medium_tag();

  // Declare the force vector
  using ViewType =
      Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>;
  ViewType force_vector;

  // Only supporting elastic_psv_t medium for Cosserat force sources
  if (medium_tag == specfem::element::medium_tag::elastic_spin) {
    force_vector = ViewType("force_vector", 6);
    force_vector(0) = this->fx;
    force_vector(1) = this->fy;
    force_vector(2) = this->fz;
    force_vector(3) = this->fc_x;
    force_vector(4) = this->fc_y;
    force_vector(5) = this->fc_z;
  } else {
    KOKKOS_ABORT_WITH_LOCATION("Cosserat force source array computation not "
                               "implemented for requested element type.");
  }

  return force_vector;
}

std::string specfem::sources::cosserat_force<
    specfem::element::dimension_tag::dim3>::print_details() const {

  const auto gcoord = this->get_global_coordinates();

  std::ostringstream message;
  message << "- Cosserat Force Source: \n"
          << "    Source Location: \n"
          << "      x = " << gcoord.x << "\n"
          << "      y = " << gcoord.y << "\n"
          << "      z = " << gcoord.z << "\n"
          << "    Source fx: " << type_real(this->fx) << "\n"
          << "    Source fy: " << type_real(this->fy) << "\n"
          << "    Source fz: " << type_real(this->fz) << "\n"
          << "    Source fc_x: " << type_real(this->fc_x) << "\n"
          << "    Source fc_y: " << type_real(this->fc_y) << "\n"
          << "    Source fc_z: " << type_real(this->fc_z) << "\n"
          << "    Source Time Function: \n"
          << this->source_time_function->print() << "\n";

  return message.str();
}

bool specfem::sources::cosserat_force<specfem::element::dimension_tag::dim3>::
operator==(const specfem::sources::source<specfem::element::dimension_tag::dim3>
               &other) const {

  // Try casting the other source to a cosserat_force source
  const auto *other_source =
      dynamic_cast<const specfem::sources::cosserat_force<
          specfem::element::dimension_tag::dim3> *>(&other);

  // Check if cast was successful
  if (other_source == nullptr) {
    std::cout << "Other source is not a cosserat_force object" << std::endl;
    return false;
  }

  const auto gcoord = this->get_global_coordinates();
  const auto other_gcoord = other_source->get_global_coordinates();

  bool internal =
      specfem::utilities::is_close(this->fx, other_source->fx) &&
      specfem::utilities::is_close(this->fy, other_source->fy) &&
      specfem::utilities::is_close(this->fz, other_source->fz) &&
      specfem::utilities::is_close(this->fc_x, other_source->fc_x) &&
      specfem::utilities::is_close(this->fc_y, other_source->fc_y) &&
      specfem::utilities::is_close(this->fc_z, other_source->fc_z) &&
      specfem::utilities::is_close(gcoord.x, other_gcoord.x) &&
      specfem::utilities::is_close(gcoord.y, other_gcoord.y) &&
      specfem::utilities::is_close(gcoord.z, other_gcoord.z);

  if (!internal) {
    std::cout << "Cosserat force sources not equal" << std::endl;
  }

  return internal && (*(this->source_time_function) ==
                      *(other_source->source_time_function));
}
bool specfem::sources::cosserat_force<specfem::element::dimension_tag::dim3>::
operator!=(const specfem::sources::source<specfem::element::dimension_tag::dim3>
               &other) const {
  return !(*this == other);
}
