

#include "specfem/coordinate_systems/coordinates/cartesian_3d.hpp"
#include "specfem/coordinate_systems/coordinates/cartesian_with_depth_3d.hpp"
#include "specfem/enums.hpp"
#include "specfem/setup.hpp"
#include "specfem/source.hpp"
#include "specfem/source_time_functions.hpp"
// #include "utilities.cpp"
#include "yaml-cpp/yaml.h"
#include <cmath>
#include <stdexcept>
#include <tuple>

namespace {
// Extract (x,y,z) from a 3D source for comparison purposes.
// If the source has stored coordinates_, extract from them;
// otherwise use global_coordinates.
std::tuple<type_real, type_real, type_real> extract_xyz(
    const specfem::sources::source<specfem::element::dimension_tag::dim3>
        &src) {
  if (const auto *coords = src.get_coordinates()) {
    using namespace specfem::coordinate_systems;
    if (const auto *c = dynamic_cast<const cartesian_3d *>(coords)) {
      return { static_cast<type_real>(c->data.x),
               static_cast<type_real>(c->data.y),
               static_cast<type_real>(c->data.z) };
    }
    if (const auto *c = dynamic_cast<const cartesian_with_depth_3d *>(coords)) {
      return { static_cast<type_real>(c->x), static_cast<type_real>(c->y),
               static_cast<type_real>(-c->depth) };
    }
  }
  const auto gc = src.get_global_coordinates();
  return { gc.x, gc.y, gc.z };
}
} // namespace

std::vector<specfem::element::medium_tag> specfem::sources::moment_tensor<
    specfem::element::dimension_tag::dim3>::get_supported_media() const {
  return { specfem::element::medium_tag::elastic };
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

  // For elastic: 3x3 tensor [[Mxx, Mxz], [Mxz, Mzz]]

  if (medium_tag == specfem::element::medium_tag::elastic) {
    source_tensor = ViewType("source_tensor", 3, 3);
    source_tensor(0, 0) = this->Mxx;
    source_tensor(0, 1) = this->Mxy;
    source_tensor(0, 2) = this->Mxz;
    source_tensor(1, 0) = this->Mxy;
    source_tensor(1, 1) = this->Myy;
    source_tensor(1, 2) = this->Myz;
    source_tensor(2, 0) = this->Mxz;
    source_tensor(2, 1) = this->Myz;
    source_tensor(2, 2) = this->Mzz;
  } else {
    KOKKOS_ABORT_WITH_LOCATION("Moment tensor source array computation not "
                               "implemented for requested element type.");
  }
  return source_tensor;
}

std::string specfem::sources::moment_tensor<
    specfem::element::dimension_tag::dim3>::print_details() const {
  std::ostringstream message;
  message << "    Moment Tensor: \n"
          << "      Mxx = " << this->Mxx << "\n"
          << "      Myy = " << this->Myy << "\n"
          << "      Mzz = " << this->Mzz << "\n"
          << "      Mxy = " << this->Mxy << "\n"
          << "      Mxz = " << this->Mxz << "\n"
          << "      Myz = " << this->Myz << "\n";
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

  const auto [x1, y1, z1] = extract_xyz(*this);
  const auto [x2, y2, z2] = extract_xyz(*other_source);

  bool internal = specfem::utilities::is_close(this->Mxx, other_source->Mxx) &&
                  specfem::utilities::is_close(this->Myy, other_source->Myy) &&
                  specfem::utilities::is_close(this->Mzz, other_source->Mzz) &&
                  specfem::utilities::is_close(this->Mxy, other_source->Mxy) &&
                  specfem::utilities::is_close(this->Mxz, other_source->Mxz) &&
                  specfem::utilities::is_close(this->Myz, other_source->Myz) &&
                  specfem::utilities::is_close(x1, x2) &&
                  specfem::utilities::is_close(y1, y2) &&
                  specfem::utilities::is_close(z1, z2);

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
