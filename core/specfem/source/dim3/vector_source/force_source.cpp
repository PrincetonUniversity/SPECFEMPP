
#include "specfem/coordinate_systems/coordinates/cartesian_3d.hpp"
#include "specfem/coordinate_systems/coordinates/cartesian_with_depth_3d.hpp"
#include "specfem/enums.hpp"
#include "specfem/macros.hpp"
#include "specfem/setup.hpp"
#include "specfem/source.hpp"
#include "specfem/source_time_functions.hpp"
#include "specfem/utilities.hpp"
#include "yaml-cpp/yaml.h"
#include <cmath>
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

std::vector<specfem::element::medium_tag> specfem::sources::force<
    specfem::element::dimension_tag::dim3>::get_supported_media() const {
  return { specfem::element::medium_tag::acoustic,
           specfem::element::medium_tag::elastic };
}

Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>
specfem::sources::force<
    specfem::element::dimension_tag::dim3>::get_force_vector() const {

  // Get the medium tag that the source is located in
  specfem::element::medium_tag medium_tag = this->get_medium_tag();

  // Declare the force vector
  using ViewType =
      Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>;
  ViewType force_vector;

  // Acoustic
  if (medium_tag == specfem::element::medium_tag::acoustic) {
    force_vector = ViewType("force_vector", 1);
    force_vector(0) = fx;
  }
  // Elastic P-SV
  else if (medium_tag == specfem::element::medium_tag::elastic) {
    force_vector = ViewType("force_vector", 3);
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
specfem::sources::force<specfem::element::dimension_tag::dim3>::print_details()
    const {
  std::ostringstream message;
  message << "    Force Vector: \n"
          << "      fx = " << type_real(this->fx) << "\n"
          << "      fy = " << type_real(this->fy) << "\n"
          << "      fz = " << type_real(this->fz) << "\n";
  return message.str();
}

bool specfem::sources::force<specfem::element::dimension_tag::dim3>::operator==(
    const specfem::sources::source<specfem::element::dimension_tag::dim3>
        &other) const {

  // Try casting the other source to a force source
  const auto *other_source = dynamic_cast<
      const specfem::sources::force<specfem::element::dimension_tag::dim3> *>(
      &other);

  // Check if cast was successful
  if (other_source == nullptr) {
    std::cout << "Other source is not a force object" << std::endl;
    return false;
  }

  const auto [x1, y1, z1] = extract_xyz(*this);
  const auto [x2, y2, z2] = extract_xyz(*other_source);

  return specfem::utilities::is_close(x1, x2) &&
         specfem::utilities::is_close(y1, y2) &&
         specfem::utilities::is_close(z1, z2) &&
         specfem::utilities::is_close(this->fx, other_source->fx) &&
         specfem::utilities::is_close(this->fy, other_source->fy) &&
         specfem::utilities::is_close(this->fz, other_source->fz) &&
         *(this->source_time_function) == *(other_source->source_time_function);
}
bool specfem::sources::force<specfem::element::dimension_tag::dim3>::operator!=(
    const specfem::sources::source<specfem::element::dimension_tag::dim3>
        &other) const {
  return !(*this == other);
}
