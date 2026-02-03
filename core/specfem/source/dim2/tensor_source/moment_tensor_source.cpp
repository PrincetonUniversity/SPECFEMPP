#include "enumerations/interface.hpp"
#include "globals.h"
#include "kokkos_abstractions.h"
#include "specfem/macros.hpp"
#include "specfem/source.hpp"
#include "specfem/source_time_functions.hpp"
#include "specfem_setup.hpp"
// #include "utilities.cpp"
#include "yaml-cpp/yaml.h"
#include <cmath>
#include <stdexcept>

std::vector<specfem::element::medium_tag> specfem::sources::moment_tensor<
    specfem::dimension::type::dim2>::get_supported_media() const {
  return { specfem::element::medium_tag::elastic_psv,
           specfem::element::medium_tag::poroelastic,
           specfem::element::medium_tag::elastic_psv_t,
           specfem::element::medium_tag::electromagnetic_te };
}

Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
specfem::sources::moment_tensor<
    specfem::dimension::type::dim2>::get_source_tensor() const {

  // Get the medium tag that the source is located in
  specfem::element::medium_tag medium_tag = this->get_medium_tag();

  using ViewType =
      Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>;

  // Declare the source tensor
  ViewType source_tensor;

  // For elastic P-SV: 2x2 tensor [[Mxx, Mxz], [Mxz, Mzz]]
  if (medium_tag == specfem::element::medium_tag::elastic_psv) {
    source_tensor = ViewType("source_tensor", 2, 2);
    source_tensor(0, 0) = this->Mxx;
    source_tensor(0, 1) = this->Mxz;
    source_tensor(1, 0) = this->Mxz;
    source_tensor(1, 1) = this->Mzz;
  }
  // For poroelastic: 4x2 tensor using elastic moment tensor twice
  else if (medium_tag == specfem::element::medium_tag::poroelastic) {
    source_tensor = ViewType("source_tensor", 4, 2);
    source_tensor(0, 0) = this->Mxx;
    source_tensor(0, 1) = this->Mxz;
    source_tensor(1, 0) = this->Mxz;
    source_tensor(1, 1) = this->Mzz;
    source_tensor(2, 0) = this->Mxx;
    source_tensor(2, 1) = this->Mxz;
    source_tensor(3, 0) = this->Mxz;
    source_tensor(3, 1) = this->Mzz;
  }
  // For elastic P-SV-T: 3x2 tensor with third component set to 0
  else if (medium_tag == specfem::element::medium_tag::elastic_psv_t) {
    source_tensor = ViewType("source_tensor", 3, 2);
    source_tensor(0, 0) = this->Mxx;
    source_tensor(0, 1) = this->Mxz;
    source_tensor(1, 0) = this->Mxz;
    source_tensor(1, 1) = this->Mzz;
    source_tensor(2, 0) = static_cast<type_real>(0.0);
    source_tensor(2, 1) = static_cast<type_real>(0.0);
  }
  // For electromagnetic TE: 2x2 tensor [[Mxx, Mxz], [Mxz, Mzz]]
  else if (medium_tag == specfem::element::medium_tag::electromagnetic_te) {
    source_tensor = ViewType("source_tensor", 2, 2);
    source_tensor(0, 0) = this->Mxx;
    source_tensor(0, 1) = this->Mxz;
    source_tensor(1, 0) = this->Mxz;
    source_tensor(1, 1) = this->Mzz;
  } else {
    KOKKOS_ABORT_WITH_LOCATION("Moment tensor source array computation not "
                               "implemented for requested element type.");
  }

  return source_tensor;
}

std::string
specfem::sources::moment_tensor<specfem::dimension::type::dim2>::print() const {

  const auto gcoord = this->get_global_coordinates();

  std::ostringstream message;
  message << "- Moment Tensor Source: \n"
          << "    Source Location: \n"
          << "      x = " << gcoord.x << "\n"
          << "      z = " << gcoord.z << "\n"
          << "    Moment Tensor: \n"
          << "      Mxx, Mzz, Mxz = " << this->Mxx << ", " << this->Mzz << ", "
          << this->Mxz << "\n"
          << "    Source Time Function: \n"
          << this->source_time_function->print() << "\n";

  return message.str();
}

bool specfem::sources::moment_tensor<specfem::dimension::type::dim2>::
operator==(const specfem::sources::source<specfem::dimension::type::dim2>
               &other) const {

  // Try casting the other source to a moment tensor source
  const auto *other_source = dynamic_cast<
      const specfem::sources::moment_tensor<specfem::dimension::type::dim2> *>(
      &other);

  // Check if cast was successful
  if (other_source == nullptr) {
    std::cout << "Other source is not a moment tensor object" << std::endl;
    return false;
  }

  const auto gcoord = this->get_global_coordinates();
  const auto other_gcoord = other_source->get_global_coordinates();

  bool internal = specfem::utilities::is_close(this->Mxx, other_source->Mxx) &&
                  specfem::utilities::is_close(this->Mxz, other_source->Mxz) &&
                  specfem::utilities::is_close(this->Mzz, other_source->Mzz) &&
                  specfem::utilities::is_close(gcoord.x, other_gcoord.x) &&
                  specfem::utilities::is_close(gcoord.z, other_gcoord.z);

  if (!internal) {
    std::cout << "Moment tensor source not equal" << std::endl;
  }

  return internal && (*(this->source_time_function) ==
                      *(other_source->source_time_function));
}

bool specfem::sources::moment_tensor<specfem::dimension::type::dim2>::
operator!=(const specfem::sources::source<specfem::dimension::type::dim2>
               &other) const {
  return !(*this == other);
}
