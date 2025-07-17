#include "specfem/source/vector_source/external.hpp"
#include "enumerations/interface.hpp"
#include "source_time_function/interface.hpp"
#include "specfem_setup.hpp"
#include <cmath>

specfem::kokkos::HostView1d<type_real>
specfem::sources::external::get_force_vector() const {

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
  // Electromagnetic TE
  else if (medium_tag == specfem::element::medium_tag::electromagnetic_te) {
    force_vector = specfem::kokkos::HostView1d<type_real>("force_vector", 2);
    force_vector(0) = 1.0;
    force_vector(1) = 1.0;
  }
  // Elastic P-SV-T (note: external source uses 1.0 for all components, unlike
  // adjoint)
  else if (medium_tag == specfem::element::medium_tag::elastic_psv_t) {
    force_vector = specfem::kokkos::HostView1d<type_real>("force_vector", 3);
    force_vector(0) = 1.0;
    force_vector(1) = 1.0;
    force_vector(2) = 1.0;
  } else {
    KOKKOS_ABORT_WITH_LOCATION("External source array computation not "
                               "implemented for requested element type.");
  }

  return force_vector;
}

std::string specfem::sources::external::print() const {

  std::ostringstream message;
  message << "- External Source: \n"
          << "    Source Location: \n"
          << "      x = " << type_real(this->x) << "\n"
          << "      z = " << type_real(this->z) << "\n"
          << "    Source Time Function: \n"
          << this->forcing_function->print() << "\n";

  return message.str();
}
