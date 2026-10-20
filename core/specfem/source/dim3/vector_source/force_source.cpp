#include "specfem/enums.hpp"
#include "specfem/macros.hpp"
#include "specfem/setup.hpp"
#include "specfem/source.hpp"
#include "specfem/source_time_functions.hpp"
#include "specfem/utilities.hpp"
#include "yaml-cpp/yaml.h"
#include <cmath>
#include <type_traits>
#include <variant>

std::vector<specfem::element::medium_tag> specfem::sources::force<
    specfem::element::dimension_tag::dim3>::get_supported_media() const {
  return {
    specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::elastic,
    specfem::element::medium_tag::elastic_spin,
  };
}

Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>
specfem::sources::force<
    specfem::element::dimension_tag::dim3>::get_force_vector() const {

  // The resolved Cartesian force; throws if the source is still in its as-read
  // local frame (assembly-time resolution must run first).
  const auto &f = this->force_.cartesian();

  // Get the medium tag that the source is located in
  specfem::element::medium_tag medium_tag = this->get_medium_tag();

  // Declare the force vector
  using ViewType =
      Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>;
  ViewType force_vector;

  // Acoustic
  if (medium_tag == specfem::element::medium_tag::acoustic) {
    force_vector = ViewType("force_vector", 1);
    force_vector(0) = f.fx;
  }
  // Elastic P-SV
  else if (medium_tag == specfem::element::medium_tag::elastic) {
    force_vector = ViewType("force_vector", 3);
    force_vector(0) = f.fx;
    force_vector(1) = f.fy;
    force_vector(2) = f.fz;
  }
  // Elastic Spin
  else if (medium_tag == specfem::element::medium_tag::elastic_spin) {
    force_vector = ViewType("force_vector", 6);
    force_vector(0) = f.fx;
    force_vector(1) = f.fy;
    force_vector(2) = f.fz;
    force_vector(3) = 0.0; // rotational dofs for pure force source are zero
    force_vector(4) = 0.0;
    force_vector(5) = 0.0;
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
  // Echo the force in the frame it was read in (provenance).
  std::visit(
      [&message](const auto &force) {
        using T = std::decay_t<decltype(force)>;
        const auto fmt = [](type_real v) {
          return specfem::utilities::format_scientific(v, 6);
        };
        if constexpr (std::is_same_v<T, specfem::sources::GeographicForce>) {
          message << "(east, north, up) = (" << fmt(force.east) << ", "
                  << fmt(force.north) << ", " << fmt(force.up) << ")";
        } else {
          message << "(fx, fy, fz) = (" << fmt(force.fx) << ", "
                  << fmt(force.fy) << ", " << fmt(force.fz) << ")";
        }
      },
      this->force_.as_read());
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

  // Compare input coordinates (identity depends solely on input, not mesh)
  const auto *c1 = this->get_read_coordinates();
  const auto *c2 = other_source->get_read_coordinates();
  bool coords_equal = (c1 && c2) ? (*c1 == *c2) : (!c1 && !c2);

  // Compare the as-read forces: same frame and close components. Identity
  // depends on the input, not the resolved Cartesian value.
  const auto &a = this->force_.as_read();
  const auto &b = other_source->force_.as_read();
  bool force_equal = false;
  if (std::holds_alternative<specfem::sources::GeographicForce>(a) &&
      std::holds_alternative<specfem::sources::GeographicForce>(b)) {
    const auto &fa = std::get<specfem::sources::GeographicForce>(a);
    const auto &fb = std::get<specfem::sources::GeographicForce>(b);
    force_equal = specfem::utilities::is_close(fa.east, fb.east) &&
                  specfem::utilities::is_close(fa.north, fb.north) &&
                  specfem::utilities::is_close(fa.up, fb.up);
  } else if (std::holds_alternative<specfem::sources::CartesianForce>(a) &&
             std::holds_alternative<specfem::sources::CartesianForce>(b)) {
    const auto &fa = std::get<specfem::sources::CartesianForce>(a);
    const auto &fb = std::get<specfem::sources::CartesianForce>(b);
    force_equal = specfem::utilities::is_close(fa.fx, fb.fx) &&
                  specfem::utilities::is_close(fa.fy, fb.fy) &&
                  specfem::utilities::is_close(fa.fz, fb.fz);
  }

  return coords_equal && force_equal &&
         *(this->source_time_function) == *(other_source->source_time_function);
}
bool specfem::sources::force<specfem::element::dimension_tag::dim3>::operator!=(
    const specfem::sources::source<specfem::element::dimension_tag::dim3>
        &other) const {
  return !(*this == other);
}
