#include "enumerations/interface.hpp"
#include "globals.h"
#include "specfem/algorithms.hpp"
#include "specfem/source.hpp"

std::vector<specfem::element::medium_tag> specfem::sources::adjoint_source<
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

specfem::kokkos::HostView1d<type_real> specfem::sources::adjoint_source<
    specfem::dimension::type::dim2>::get_force_vector() const {

  // Get the medium tag that the source is located in
  specfem::element::medium_tag medium_tag = this->get_medium_tag();

  // Declare the force vector
  specfem::kokkos::HostView1d<type_real> force_vector;

  // Acoustic
  if (medium_tag == specfem::element::medium_tag::acoustic) {
    force_vector = specfem::kokkos::HostView1d<type_real>("force_vector", 1);
    force_vector(0) = 1.0;
  }
  // Elastic SH
  else if (medium_tag == specfem::element::medium_tag::elastic_sh) {
    force_vector = specfem::kokkos::HostView1d<type_real>("force_vector", 1);
    force_vector(0) = 1.0;
  }
  // Elastic P-SV
  else if (medium_tag == specfem::element::medium_tag::elastic_psv) {
    force_vector = specfem::kokkos::HostView1d<type_real>("force_vector", 2);
    force_vector(0) = 1.0;
    force_vector(1) = 1.0;
  }
  // Poroelastic
  else if (medium_tag == specfem::element::medium_tag::poroelastic) {
    force_vector = specfem::kokkos::HostView1d<type_real>("force_vector", 4);
    force_vector(0) = 1.0;
    force_vector(1) = 1.0;
    force_vector(2) = 1.0;
    force_vector(3) = 1.0;
  }
  // Elastic P-SV-T
  else if (medium_tag == specfem::element::medium_tag::elastic_psv_t) {
    force_vector = specfem::kokkos::HostView1d<type_real>("force_vector", 3);
    force_vector(0) = 1.0;
    force_vector(1) = 1.0;
    force_vector(2) = static_cast<type_real>(0.0);
  }
  // Electromagnetic TE
  else if (medium_tag == specfem::element::medium_tag::electromagnetic_te) {
    force_vector = specfem::kokkos::HostView1d<type_real>("force_vector", 2);
    force_vector(0) = 1.0;
    force_vector(1) = 1.0;
  } else {
    KOKKOS_ABORT_WITH_LOCATION("Adjoint source array computation not "
                               "implemented for requested element type.");
  }

  return force_vector;
}

std::string
specfem::sources::adjoint_source<specfem::dimension::type::dim2>::print()
    const {

  const auto gcoord = this->get_global_coordinates();

  std::ostringstream message;
  message << "- Adjoint Source: \n"
          << "    Source Location: \n"
          << "      x = " << gcoord.x << "\n"
          << "      z = " << gcoord.z << "\n"
          << "    Source Time Function: \n"
          << this->source_time_function->print() << "\n";
  return message.str();
}
