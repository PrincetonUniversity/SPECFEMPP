#include "specfem/enums.hpp"
#include "specfem/macros.hpp"
#include "specfem/source.hpp"

std::vector<specfem::element::medium_tag> specfem::sources::adjoint_source<
    specfem::element::dimension_tag::dim3>::get_supported_media() const {
  return {
    specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::elastic,
  };
}

Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>
specfem::sources::adjoint_source<
    specfem::element::dimension_tag::dim3>::get_force_vector() const {

  // Get the medium tag that the source is located in
  specfem::element::medium_tag medium_tag = this->get_medium_tag();

  // Declare the force vector
  using ViewType =
      Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>;
  ViewType force_vector;

  // Acoustic: single scalar DOF (pressure)
  if (medium_tag == specfem::element::medium_tag::acoustic) {
    force_vector = ViewType("force_vector", 1);
    force_vector(0) = 1.0;
  }
  // Elastic: three DOFs (x, y, z)
  else if (medium_tag == specfem::element::medium_tag::elastic) {
    force_vector = ViewType("force_vector", 3);
    force_vector(0) = 1.0;
    force_vector(1) = 1.0;
    force_vector(2) = 1.0;
  } else {
    KOKKOS_ABORT_WITH_LOCATION("3-D adjoint source array computation not "
                               "implemented for requested element type.");
  }

  return force_vector;
}
