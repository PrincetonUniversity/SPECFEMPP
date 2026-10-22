#include "specfem/enums.hpp"
#include "specfem/setup.hpp"
#include "specfem/source.hpp"
#include "specfem/source_time_functions.hpp"
#include "specfem/utilities.hpp"
#include "yaml-cpp/yaml.h"
#include <cmath>
#include <type_traits>
#include <variant>

specfem::sources::CartesianMomentTensor
specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>::
    read_cartesian_tensor(const YAML::Node &Node) {
  specfem::sources::CartesianMomentTensor tensor{
    Node["Mxx"].as<type_real>(), Node["Myy"].as<type_real>(),
    Node["Mzz"].as<type_real>(), Node["Mxy"].as<type_real>(),
    Node["Mxz"].as<type_real>(), Node["Myz"].as<type_real>()
  };
  // Optional: asymmetric tensors set the lower-triangle components; otherwise
  // they keep their symmetric defaults.
  if (Node["Myx"])
    tensor.Myx = Node["Myx"].as<type_real>();
  if (Node["Mzx"])
    tensor.Mzx = Node["Mzx"].as<type_real>();
  if (Node["Mzy"])
    tensor.Mzy = Node["Mzy"].as<type_real>();
  return tensor;
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
    source_tensor(0, 0) = m.Mxx;
    source_tensor(0, 1) = m.Mxy;
    source_tensor(0, 2) = m.Mxz;
    source_tensor(1, 0) = m.Myx;
    source_tensor(1, 1) = m.Myy;
    source_tensor(1, 2) = m.Myz;
    source_tensor(2, 0) = m.Mzx;
    source_tensor(2, 1) = m.Mzy;
    source_tensor(2, 2) = m.Mzz;
  }
  // For elastic_spin (Cosserat): 6x3 tensor. The three displacement rows carry
  // the moment tensor; the three rotation rows are zero (the rotational forcing
  // enters via the body couple instead).
  else if (medium_tag == specfem::element::medium_tag::elastic_spin) {
    source_tensor = ViewType("source_tensor", 6, 3);
    source_tensor(0, 0) = m.Mxx;
    source_tensor(0, 1) = m.Mxy;
    source_tensor(0, 2) = m.Mxz;
    source_tensor(1, 0) = m.Myx;
    source_tensor(1, 1) = m.Myy;
    source_tensor(1, 2) = m.Myz;
    source_tensor(2, 0) = m.Mzx;
    source_tensor(2, 1) = m.Mzy;
    source_tensor(2, 2) = m.Mzz;
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
    body_couple(3) = m.Myz - m.Mzy;
    body_couple(4) = m.Mzx - m.Mxz;
    body_couple(5) = m.Mxy - m.Myx;
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
        if constexpr (std::is_same_v<
                          T, specfem::sources::GeographicMomentTensor>) {
          message << "(Mrr, Mtt, Mpp, Mrt, Mrp, Mtp) = (" << fmt(tensor.Mrr)
                  << ", " << fmt(tensor.Mtt) << ", " << fmt(tensor.Mpp) << ", "
                  << fmt(tensor.Mrt) << ", " << fmt(tensor.Mrp) << ", "
                  << fmt(tensor.Mtp) << ")";
        } else {
          message << "(Mxx, Myy, Mzz, Mxy, Mxz, Myz, Myx, Mzx, Mzy) = ("
                  << fmt(tensor.Mxx) << ", " << fmt(tensor.Myy) << ", "
                  << fmt(tensor.Mzz) << ", " << fmt(tensor.Mxy) << ", "
                  << fmt(tensor.Mxz) << ", " << fmt(tensor.Myz) << ", "
                  << fmt(tensor.Myx) << ", " << fmt(tensor.Mzx) << ", "
                  << fmt(tensor.Mzy) << ")";
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
  const auto &a = this->moment_.as_read();
  const auto &b = other_source->moment_.as_read();
  bool tensor_equal = false;
  if (std::holds_alternative<specfem::sources::GeographicMomentTensor>(a) &&
      std::holds_alternative<specfem::sources::GeographicMomentTensor>(b)) {
    const auto &ta = std::get<specfem::sources::GeographicMomentTensor>(a);
    const auto &tb = std::get<specfem::sources::GeographicMomentTensor>(b);
    tensor_equal = specfem::utilities::is_close(ta.Mrr, tb.Mrr) &&
                   specfem::utilities::is_close(ta.Mtt, tb.Mtt) &&
                   specfem::utilities::is_close(ta.Mpp, tb.Mpp) &&
                   specfem::utilities::is_close(ta.Mrt, tb.Mrt) &&
                   specfem::utilities::is_close(ta.Mrp, tb.Mrp) &&
                   specfem::utilities::is_close(ta.Mtp, tb.Mtp);
  } else if (std::holds_alternative<specfem::sources::CartesianMomentTensor>(
                 a) &&
             std::holds_alternative<specfem::sources::CartesianMomentTensor>(
                 b)) {
    const auto &ta = std::get<specfem::sources::CartesianMomentTensor>(a);
    const auto &tb = std::get<specfem::sources::CartesianMomentTensor>(b);
    tensor_equal = specfem::utilities::is_close(ta.Mxx, tb.Mxx) &&
                   specfem::utilities::is_close(ta.Myy, tb.Myy) &&
                   specfem::utilities::is_close(ta.Mzz, tb.Mzz) &&
                   specfem::utilities::is_close(ta.Mxy, tb.Mxy) &&
                   specfem::utilities::is_close(ta.Mxz, tb.Mxz) &&
                   specfem::utilities::is_close(ta.Myz, tb.Myz) &&
                   specfem::utilities::is_close(ta.Myx, tb.Myx) &&
                   specfem::utilities::is_close(ta.Mzx, tb.Mzx) &&
                   specfem::utilities::is_close(ta.Mzy, tb.Mzy);
  }

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
