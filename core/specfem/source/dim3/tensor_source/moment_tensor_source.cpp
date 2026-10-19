#include "specfem/enums.hpp"
#include "specfem/setup.hpp"
#include "specfem/source.hpp"
#include "specfem/source_time_functions.hpp"
#include "specfem/utilities.hpp"
#include "yaml-cpp/yaml.h"
#include <cmath>
#include <type_traits>
#include <variant>

std::vector<specfem::element::medium_tag> specfem::sources::moment_tensor<
    specfem::element::dimension_tag::dim3>::get_supported_media() const {
  return { specfem::element::medium_tag::elastic };
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

  // For elastic: 3x3 tensor [[Mxx, Mxy, Mxz], [Mxy, Myy, Myz], [Mxz, Myz, Mzz]]
  if (medium_tag == specfem::element::medium_tag::elastic) {
    source_tensor = ViewType("source_tensor", 3, 3);
    source_tensor(0, 0) = m.Mxx;
    source_tensor(0, 1) = m.Mxy;
    source_tensor(0, 2) = m.Mxz;
    source_tensor(1, 0) = m.Mxy;
    source_tensor(1, 1) = m.Myy;
    source_tensor(1, 2) = m.Myz;
    source_tensor(2, 0) = m.Mxz;
    source_tensor(2, 1) = m.Myz;
    source_tensor(2, 2) = m.Mzz;
  } else {
    KOKKOS_ABORT_WITH_LOCATION("Moment tensor source array computation not "
                               "implemented for requested element type.");
  }
  return source_tensor;
}

std::string specfem::sources::moment_tensor<
    specfem::element::dimension_tag::dim3>::print_details() const {
  std::ostringstream message;
  // Echo the moment tensor in the frame it was read in (provenance).
  std::visit(
      [&message](const auto &tensor) {
        using T = std::decay_t<decltype(tensor)>;
        if constexpr (std::is_same_v<
                          T, specfem::sources::GeographicMomentTensor>) {
          message << "(Mrr, Mtt, Mpp, Mrt, Mrp, Mtp) = ("
                  << specfem::utilities::format_scientific(tensor.Mrr, 6)
                  << ", "
                  << specfem::utilities::format_scientific(tensor.Mtt, 6)
                  << ", "
                  << specfem::utilities::format_scientific(tensor.Mpp, 6)
                  << ", "
                  << specfem::utilities::format_scientific(tensor.Mrt, 6)
                  << ", "
                  << specfem::utilities::format_scientific(tensor.Mrp, 6)
                  << ", "
                  << specfem::utilities::format_scientific(tensor.Mtp, 6)
                  << ")";
        } else {
          message << "(Mxx, Myy, Mzz, Mxy, Mxz, Myz) = ("
                  << specfem::utilities::format_scientific(tensor.Mxx, 6)
                  << ", "
                  << specfem::utilities::format_scientific(tensor.Myy, 6)
                  << ", "
                  << specfem::utilities::format_scientific(tensor.Mzz, 6)
                  << ", "
                  << specfem::utilities::format_scientific(tensor.Mxy, 6)
                  << ", "
                  << specfem::utilities::format_scientific(tensor.Mxz, 6)
                  << ", "
                  << specfem::utilities::format_scientific(tensor.Myz, 6)
                  << ")";
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
                   specfem::utilities::is_close(ta.Myz, tb.Myz);
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
