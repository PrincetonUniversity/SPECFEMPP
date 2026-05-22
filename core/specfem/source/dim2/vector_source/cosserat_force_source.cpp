
#include "specfem/enums.hpp"
#include "specfem/setup.hpp"
#include "specfem/source.hpp"
#include "specfem/source_time_functions.hpp"
#include "specfem/utilities.hpp"
#include <cmath>

std::vector<specfem::element::medium_tag> specfem::sources::cosserat_force<
    specfem::element::dimension_tag::dim2>::get_supported_media() const {
  return { specfem::element::medium_tag::elastic_psv_t };
}

Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>
specfem::sources::cosserat_force<
    specfem::element::dimension_tag::dim2>::get_force_vector() const {

  // Get the medium tag that the source is located in
  specfem::element::medium_tag medium_tag = this->get_medium_tag();

  // Declare the force vector
  using ViewType =
      Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>;
  ViewType force_vector;

  // Convert angle to radians
  type_real angle_in_rad = this->angle * Kokkos::numbers::pi_v<type_real> /
                           static_cast<type_real>(180.0);

  // Only supporting elastic_psv_t medium for Cosserat force sources
  if (medium_tag == specfem::element::medium_tag::elastic_psv_t) {
    force_vector = ViewType("force_vector", 3);
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

std::string specfem::sources::cosserat_force<
    specfem::element::dimension_tag::dim2>::print_details() const {
  std::ostringstream message;
  message << "    Source Angle: " << type_real(this->angle) << "\n"
          << "    Source f: " << type_real(this->f) << "\n"
          << "    Source fc: " << type_real(this->fc) << "\n";
  return message.str();
}

bool specfem::sources::cosserat_force<specfem::element::dimension_tag::dim2>::
operator==(const specfem::sources::source<specfem::element::dimension_tag::dim2>
               &other) const {

  // Try casting the other source to a cosserat_force source
  const auto *other_source =
      dynamic_cast<const specfem::sources::cosserat_force<
          specfem::element::dimension_tag::dim2> *>(&other);

  // Check if cast was successful
  if (other_source == nullptr) {
    std::cout << "Other source is not a cosserat_force object" << std::endl;
    return false;
  }

  // Compare input coordinates (identity depends solely on input, not mesh)
  const auto *c1 = this->get_input_coordinates();
  const auto *c2 = other_source->get_input_coordinates();
  bool coords_equal = (c1 && c2) ? (*c1 == *c2) : (!c1 && !c2);

  bool internal =
      coords_equal && specfem::utilities::is_close(this->f, other_source->f) &&
      specfem::utilities::is_close(this->angle, other_source->angle);

  if (!internal) {
    std::cout << "Cosserat force sources not equal" << std::endl;
  }

  return internal && (*(this->source_time_function) ==
                      *(other_source->source_time_function));
}
bool specfem::sources::cosserat_force<specfem::element::dimension_tag::dim2>::
operator!=(const specfem::sources::source<specfem::element::dimension_tag::dim2>
               &other) const {
  return !(*this == other);
}
