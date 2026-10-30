#include "specfem/enums.hpp"
#include "specfem/setup.hpp"
#include "specfem/source.hpp"
#include "specfem/source_time_functions.hpp"
#include "specfem/utilities.hpp"
#include "yaml-cpp/yaml.h"
#include <cmath>
#include <type_traits>
#include <variant>

specfem::sources::moment_tensor<
    specfem::element::dimension_tag::dim3>::CartesianTensor
specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>::
    read_cartesian_tensor(const YAML::Node &Node) {
  const auto Mxx = Node["Mxx"].as<type_real>();
  const auto Myy = Node["Myy"].as<type_real>();
  const auto Mzz = Node["Mzz"].as<type_real>();
  const auto Mxy = Node["Mxy"].as<type_real>();
  const auto Mxz = Node["Mxz"].as<type_real>();
  const auto Myz = Node["Myz"].as<type_real>();
  // Optional: asymmetric tensors set the lower-triangle components; otherwise
  // they default to their transposes (symmetric tensor).
  const auto Myx = Node["Myx"] ? Node["Myx"].as<type_real>() : Mxy;
  const auto Mzx = Node["Mzx"] ? Node["Mzx"].as<type_real>() : Mxz;
  const auto Mzy = Node["Mzy"] ? Node["Mzy"].as<type_real>() : Myz;
  return CartesianTensor{ Mxx, Myy, Mzz, Mxy, Mxz, Myz, Myx, Mzx, Mzy };
}

std::vector<specfem::element::medium_tag> specfem::sources::moment_tensor<
    specfem::element::dimension_tag::dim3>::get_supported_media() const {
  return { specfem::element::medium_tag::elastic,
           specfem::element::medium_tag::elastic_spin };
}

Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
specfem::sources::moment_tensor<
    specfem::element::dimension_tag::dim3>::get_source_tensor() const {

  // The resolved Cartesian tensor; throws if the source is still in its
  // as-read geographic frame (assembly-time resolution must run first).
  const auto &m = this->moment_.cartesian();

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
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        source_tensor(i, j) = m(i, j);
      }
    }
  }
  // For elastic_spin (Cosserat): 6x3 tensor. The three displacement rows carry
  // the moment tensor; the three rotation rows are zero (the rotational forcing
  // enters via the body couple instead).
  else if (medium_tag == specfem::element::medium_tag::elastic_spin) {
    source_tensor = ViewType("source_tensor", 6, 3);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        source_tensor(i, j) = m(i, j);
      }
    }
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
    const auto &m = this->moment_.cartesian();
    ViewType body_couple("body_couple_vector", 6);
    body_couple(0) = static_cast<type_real>(0.0);
    body_couple(1) = static_cast<type_real>(0.0);
    body_couple(2) = static_cast<type_real>(0.0);
    body_couple(3) = m(1, 2) - m(2, 1); // Myz - Mzy
    body_couple(4) = m(2, 0) - m(0, 2); // Mzx - Mxz
    body_couple(5) = m(0, 1) - m(1, 0); // Mxy - Myx
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
  // Echo the moment tensor in the frame it was read in (provenance).
  std::visit(
      [&message](const auto &tensor) {
        using T = std::decay_t<decltype(tensor)>;
        const auto fmt = [](type_real v) {
          return specfem::utilities::format_scientific(v, 6);
        };
        if constexpr (std::is_same_v<T, GeocentricTensor>) {
          // Geocentric axes are (theta, phi, r): report in CMTSOLUTION order.
          message << "(Mrr, Mtt, Mpp, Mrt, Mrp, Mtp) = (" << fmt(tensor(2, 2))
                  << ", " << fmt(tensor(0, 0)) << ", " << fmt(tensor(1, 1))
                  << ", " << fmt(tensor(0, 2)) << ", " << fmt(tensor(1, 2))
                  << ", " << fmt(tensor(0, 1)) << ")";
        } else {
          message << "(Mxx, Myy, Mzz, Mxy, Mxz, Myz, Myx, Mzx, Mzy) = ("
                  << fmt(tensor(0, 0)) << ", " << fmt(tensor(1, 1)) << ", "
                  << fmt(tensor(2, 2)) << ", " << fmt(tensor(0, 1)) << ", "
                  << fmt(tensor(0, 2)) << ", " << fmt(tensor(1, 2)) << ", "
                  << fmt(tensor(1, 0)) << ", " << fmt(tensor(2, 0)) << ", "
                  << fmt(tensor(2, 1)) << ")";
        }
      },
      this->moment_.as_read());
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

  // Compare the as-read tensors: same frame and close components. Identity
  // depends on the input, not the resolved Cartesian value.
  const bool tensor_equal = std::visit(
      [](const auto &a, const auto &b) {
        if constexpr (std::is_same_v<std::decay_t<decltype(a)>,
                                     std::decay_t<decltype(b)>>) {
          return specfem::sources::is_close(a, b);
        } else {
          return false;
        }
      },
      this->moment_.as_read(), other_source->moment_.as_read());

  bool internal = coords_equal && tensor_equal;

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
