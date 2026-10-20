#include "specfem/enums.hpp"
#include "specfem/setup.hpp"
#include "specfem/source.hpp"
#include "specfem/source_time_functions.hpp"
#include "specfem/utilities.hpp"
#include "yaml-cpp/yaml.h"
#include <cmath>

std::vector<specfem::element::medium_tag> specfem::sources::moment_tensor<
    specfem::element::dimension_tag::dim3>::get_supported_media() const {
  return { specfem::element::medium_tag::elastic,
           specfem::element::medium_tag::elastic_spin };
}

Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
specfem::sources::moment_tensor<
    specfem::element::dimension_tag::dim3>::get_source_tensor() const {

  // Get the medium tag that the source is located in
  specfem::element::medium_tag medium_tag = this->get_medium_tag();

  // Declare the source tensor
  using ViewType =
      Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>;
  ViewType source_tensor;

  // The lower-triangle slots carry Myx/Mzx/Mzy so that asymmetric tensors drive
  // the displacement dipole correctly; they equal Mxy/Mxz/Myz for the default
  // (symmetric) tensor.

  // For elastic: 3x3 tensor [[Mxx, Mxy, Mxz], [Myx, Myy, Myz], [Mzx, Mzy, Mzz]]
  if (medium_tag == specfem::element::medium_tag::elastic) {
    source_tensor = ViewType("source_tensor", 3, 3);
    source_tensor(0, 0) = this->Mxx;
    source_tensor(0, 1) = this->Mxy;
    source_tensor(0, 2) = this->Mxz;
    source_tensor(1, 0) = this->Myx;
    source_tensor(1, 1) = this->Myy;
    source_tensor(1, 2) = this->Myz;
    source_tensor(2, 0) = this->Mzx;
    source_tensor(2, 1) = this->Mzy;
    source_tensor(2, 2) = this->Mzz;
  }
  // For elastic_spin (Cosserat): 6x3 tensor. The three displacement rows carry
  // the moment tensor; the three rotation rows are zero (the rotational forcing
  // enters via the body couple instead).
  else if (medium_tag == specfem::element::medium_tag::elastic_spin) {
    source_tensor = ViewType("source_tensor", 6, 3);
    source_tensor(0, 0) = this->Mxx;
    source_tensor(0, 1) = this->Mxy;
    source_tensor(0, 2) = this->Mxz;
    source_tensor(1, 0) = this->Myx;
    source_tensor(1, 1) = this->Myy;
    source_tensor(1, 2) = this->Myz;
    source_tensor(2, 0) = this->Mzx;
    source_tensor(2, 1) = this->Mzy;
    source_tensor(2, 2) = this->Mzz;
    for (int j = 0; j < 3; ++j) {
      source_tensor(3, j) = static_cast<type_real>(0.0);
      source_tensor(4, j) = static_cast<type_real>(0.0);
      source_tensor(5, j) = static_cast<type_real>(0.0);
    }
  } else {
    KOKKOS_ABORT_WITH_LOCATION("Moment tensor source array computation not "
                               "implemented for requested element type.");
  }
  return source_tensor;
}

Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>
specfem::sources::moment_tensor<
    specfem::element::dimension_tag::dim3>::get_body_couple_vector() const {

  using ViewType =
      Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>;

  // Only Cosserat media (elastic_spin) carry rotational degrees of freedom.
  // The antisymmetric part of the moment tensor drives them via the monopole
  // term; all other media return an empty view. The axial vector of the
  // antisymmetric part follows the 3D Levi-Civita convention:
  //   (eps:M)_x = Myz - Mzy, (eps:M)_y = Mzx - Mxz, (eps:M)_z = Mxy - Myx.
  if (this->get_medium_tag() == specfem::element::medium_tag::elastic_spin) {
    ViewType body_couple("body_couple_vector", 6);
    body_couple(0) = static_cast<type_real>(0.0);
    body_couple(1) = static_cast<type_real>(0.0);
    body_couple(2) = static_cast<type_real>(0.0);
    body_couple(3) = this->Myz - this->Mzy;
    body_couple(4) = this->Mzx - this->Mxz;
    body_couple(5) = this->Mxy - this->Myx;
    return body_couple;
  }

  return {};
}

bool specfem::sources::moment_tensor<
    specfem::element::dimension_tag::dim3>::has_monopole_contribution() const {
  return this->get_medium_tag() == specfem::element::medium_tag::elastic_spin;
}

std::string specfem::sources::moment_tensor<
    specfem::element::dimension_tag::dim3>::print_details() const {
  std::ostringstream message;
  message << "(Mxx, Myy, Mzz, Mxy, Mxz, Myz, Myx, Mzx, Mzy) = ("
          << specfem::utilities::format_scientific(this->Mxx, 6) << ", "
          << specfem::utilities::format_scientific(this->Myy, 6) << ", "
          << specfem::utilities::format_scientific(this->Mzz, 6) << ", "
          << specfem::utilities::format_scientific(this->Mxy, 6) << ", "
          << specfem::utilities::format_scientific(this->Mxz, 6) << ", "
          << specfem::utilities::format_scientific(this->Myz, 6) << ", "
          << specfem::utilities::format_scientific(this->Myx, 6) << ", "
          << specfem::utilities::format_scientific(this->Mzx, 6) << ", "
          << specfem::utilities::format_scientific(this->Mzy, 6) << ")";
  return message.str();
}

bool specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>::
operator==(const specfem::sources::source<specfem::element::dimension_tag::dim3>
               &other) const {

  // Try casting the other source to a moment tensor source
  const auto *other_source = dynamic_cast<const specfem::sources::moment_tensor<
      specfem::element::dimension_tag::dim3> *>(&other);

  // Check if cast was successful
  if (other_source == nullptr) {
    std::cout << "Other source is not a moment tensor object" << std::endl;
    return false;
  }

  // Compare input coordinates (identity depends solely on input, not mesh)
  const auto *c1 = this->get_read_coordinates();
  const auto *c2 = other_source->get_read_coordinates();
  bool coords_equal = (c1 && c2) ? (*c1 == *c2) : (!c1 && !c2);

  bool internal = coords_equal &&
                  specfem::utilities::is_close(this->Mxx, other_source->Mxx) &&
                  specfem::utilities::is_close(this->Myy, other_source->Myy) &&
                  specfem::utilities::is_close(this->Mzz, other_source->Mzz) &&
                  specfem::utilities::is_close(this->Mxy, other_source->Mxy) &&
                  specfem::utilities::is_close(this->Mxz, other_source->Mxz) &&
                  specfem::utilities::is_close(this->Myz, other_source->Myz) &&
                  specfem::utilities::is_close(this->Myx, other_source->Myx) &&
                  specfem::utilities::is_close(this->Mzx, other_source->Mzx) &&
                  specfem::utilities::is_close(this->Mzy, other_source->Mzy);

  if (!internal) {
    std::cout << "3-D moment tensor source not equal" << std::endl;
  }

  return internal && (*(this->source_time_function) ==
                      *(other_source->source_time_function));
}

bool specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>::
operator!=(const specfem::sources::source<specfem::element::dimension_tag::dim3>
               &other) const {
  return !(*this == other);
}
