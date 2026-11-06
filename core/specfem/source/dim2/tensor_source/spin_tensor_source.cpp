

#include "specfem/enums.hpp"
#include "specfem/macros.hpp"
#include "specfem/setup.hpp"
#include "specfem/source.hpp"
#include "specfem/source_time_functions.hpp"
#include "specfem/utilities.hpp"
#include "yaml-cpp/yaml.h"
#include <cmath>
#include <stdexcept>

std::vector<specfem::element::medium_tag> specfem::sources::spin_tensor<
    specfem::element::dimension_tag::dim2>::get_supported_media() const {
  return { specfem::element::medium_tag::elastic_psv_t };
}

Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
specfem::sources::spin_tensor<
    specfem::element::dimension_tag::dim2>::get_source_tensor() const {

  // Get the medium tag that the source is located in
  specfem::element::medium_tag medium_tag = this->get_medium_tag();

  using ViewType =
      Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>;

  // Declare the source tensor
  ViewType source_tensor;

  // For elastic P-SV-T: 3x2 tensor with the displacement rows set to 0; the
  // rotation row carries the spin tensor, which drives the micro-rotation
  // field through the gradient contraction.
  if (medium_tag == specfem::element::medium_tag::elastic_psv_t) {
    source_tensor = ViewType("source_tensor", 3, 2);
    source_tensor(0, 0) = static_cast<type_real>(0.0);
    source_tensor(0, 1) = static_cast<type_real>(0.0);
    source_tensor(1, 0) = static_cast<type_real>(0.0);
    source_tensor(1, 1) = static_cast<type_real>(0.0);
    source_tensor(2, 0) = this->Mcyx;
    source_tensor(2, 1) = this->Mcyz;
  } else {
    KOKKOS_ABORT_WITH_LOCATION("Spin tensor source array computation not "
                               "implemented for requested element type.");
  }

  return source_tensor;
}

std::string specfem::sources::spin_tensor<
    specfem::element::dimension_tag::dim2>::print_details() const {
  std::ostringstream message;
  message << "(Mcyx, Mcyz) = ("
          << specfem::utilities::format_scientific(this->Mcyx, 6) << ", "
          << specfem::utilities::format_scientific(this->Mcyz, 6) << ")";
  return message.str();
}

bool specfem::sources::spin_tensor<specfem::element::dimension_tag::dim2>::
operator==(const specfem::sources::source<specfem::element::dimension_tag::dim2>
               &other) const {

  // Try casting the other source to a spin tensor source
  const auto *other_source = dynamic_cast<const specfem::sources::spin_tensor<
      specfem::element::dimension_tag::dim2> *>(&other);

  // Check if cast was successful
  if (other_source == nullptr) {
    std::cout << "Other source is not a spin tensor object" << std::endl;
    return false;
  }

  // Compare input coordinates (identity depends solely on input, not mesh)
  const auto *c1 = this->get_read_coordinates();
  const auto *c2 = other_source->get_read_coordinates();
  bool coords_equal = (c1 && c2) ? (*c1 == *c2) : (!c1 && !c2);

  bool internal =
      coords_equal &&
      specfem::utilities::is_close(this->Mcyx, other_source->Mcyx) &&
      specfem::utilities::is_close(this->Mcyz, other_source->Mcyz);

  if (!internal) {
    std::cout << "Spin tensor source not equal" << std::endl;
  }

  return internal && (*(this->source_time_function) ==
                      *(other_source->source_time_function));
}

bool specfem::sources::spin_tensor<specfem::element::dimension_tag::dim2>::
operator!=(const specfem::sources::source<specfem::element::dimension_tag::dim2>
               &other) const {
  return !(*this == other);
}
