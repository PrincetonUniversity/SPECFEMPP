#include "enumerations/interface.hpp"
#include "specfem/macros.hpp"
#include "specfem/source.hpp"
#include "specfem/source_time_functions.hpp"
#include "specfem_setup.hpp"
#include <cmath>

std::vector<specfem::element::medium_tag> specfem::sources::external<
    specfem::dimension::type::dim2>::get_supported_media() const {
  return {
    specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::elastic_psv,
    specfem::element::medium_tag::elastic_psv_t,
    specfem::element::medium_tag::elastic_sh,
    specfem::element::medium_tag::electromagnetic_te,
    specfem::element::medium_tag::poroelastic,
  };
}

Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>
specfem::sources::external<specfem::dimension::type::dim2>::get_force_vector()
    const {

  // Get the medium tag that the source is located in
  specfem::element::medium_tag medium_tag = this->get_medium_tag();

  // Declare the force vector
  using ViewType =
      Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>;
  ViewType force_vector;

  // Acoustic
  if (medium_tag == specfem::element::medium_tag::acoustic) {
    force_vector = ViewType("force_vector", 1);
    force_vector(0) = 1.0;
  }
  // Elastic SH
  else if (medium_tag == specfem::element::medium_tag::elastic_sh) {
    force_vector = ViewType("force_vector", 1);
    force_vector(0) = 1.0;
  }
  // Elastic P-SV
  else if (medium_tag == specfem::element::medium_tag::elastic_psv) {
    force_vector = ViewType("force_vector", 2);
    force_vector(0) = 1.0;
    force_vector(1) = 1.0;
  }
  // Poroelastic
  else if (medium_tag == specfem::element::medium_tag::poroelastic) {
    force_vector = ViewType("force_vector", 4);
    force_vector(0) = 1.0;
    force_vector(1) = 1.0;
    force_vector(2) = 1.0;
    force_vector(3) = 1.0;
  }
  // Electromagnetic TE
  else if (medium_tag == specfem::element::medium_tag::electromagnetic_te) {
    force_vector = ViewType("force_vector", 2);
    force_vector(0) = 1.0;
    force_vector(1) = 1.0;
  }
  // Elastic P-SV-T (note: external source uses 1.0 for all components, unlike
  // adjoint)
  else if (medium_tag == specfem::element::medium_tag::elastic_psv_t) {
    force_vector = ViewType("force_vector", 3);
    force_vector(0) = 1.0;
    force_vector(1) = 1.0;
    force_vector(2) = 1.0;
  } else {
    KOKKOS_ABORT_WITH_LOCATION("External source array computation not "
                               "implemented for requested element type.");
  }

  return force_vector;
}

std::string
specfem::sources::external<specfem::dimension::type::dim2>::print() const {

  const auto gcoord = this->get_global_coordinates();

  std::ostringstream message;
  message << "- External Source: \n"
          << "    Source Location: \n"
          << "      x = " << type_real(gcoord.x) << "\n"
          << "      z = " << type_real(gcoord.z) << "\n"
          << "    Source Time Function: \n"
          << this->source_time_function->print() << "\n";

  return message.str();
}
