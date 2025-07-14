#include "specfem/assembly/jacobian_matrix.hpp"
#include "mesh/mesh.hpp"

specfem::assembly::jacobian_matrix<specfem::dimension::type::dim3>::
    jacobian_matrix(
        const specfem::mesh::jacobian_matrix<dimension_tag> &mesh_jacobian)
    : nspec(mesh_jacobian.nspec), ngllx(mesh_jacobian.ngllx),
      nglly(mesh_jacobian.nglly), ngllz(mesh_jacobian.ngllz),
      h_xix(mesh_jacobian.xix), h_xiy(mesh_jacobian.xiy),
      h_xiz(mesh_jacobian.xiz), h_etax(mesh_jacobian.etax),
      h_etay(mesh_jacobian.etay), h_etaz(mesh_jacobian.etaz),
      h_gammax(mesh_jacobian.gammax), h_gammay(mesh_jacobian.gammay),
      h_gammaz(mesh_jacobian.gammaz), h_jacobian(mesh_jacobian.jacobian),
      xix(Kokkos::ViewAllocateWithoutInitializing("xix"), nspec, ngllx, nglly,
          ngllz),
      xiy(Kokkos::ViewAllocateWithoutInitializing("xiy"), nspec, ngllx, nglly,
          ngllz),
      xiz(Kokkos::ViewAllocateWithoutInitializing("xiz"), nspec, ngllx, nglly,
          ngllz),
      etax(Kokkos::ViewAllocateWithoutInitializing("etax"), nspec, ngllx, nglly,
           ngllz),
      etay(Kokkos::ViewAllocateWithoutInitializing("etay"), nspec, ngllx, nglly,
           ngllz),
      etaz(Kokkos::ViewAllocateWithoutInitializing("etaz"), nspec, ngllx, nglly,
           ngllz),
      gammax(Kokkos::ViewAllocateWithoutInitializing("gammax"), nspec, ngllx,
             nglly, ngllz),
      gammay(Kokkos::ViewAllocateWithoutInitializing("gammay"), nspec, ngllx,
             nglly, ngllz),
      gammaz(Kokkos::ViewAllocateWithoutInitializing("gammaz"), nspec, ngllx,
             nglly, ngllz),
      jacobian(Kokkos::ViewAllocateWithoutInitializing("jacobian"), nspec,
               ngllx, nglly, ngllz) {
  this->sync_views();
  return;
}

void specfem::assembly::jacobian_matrix<
    specfem::dimension::type::dim3>::sync_views() {
  Kokkos::deep_copy(xix, h_xix);
  Kokkos::deep_copy(xiy, h_xiy);
  Kokkos::deep_copy(xiz, h_xiz);
  Kokkos::deep_copy(etax, h_etax);
  Kokkos::deep_copy(etay, h_etay);
  Kokkos::deep_copy(etaz, h_etaz);
  Kokkos::deep_copy(gammax, h_gammax);
  Kokkos::deep_copy(gammay, h_gammay);
  Kokkos::deep_copy(gammaz, h_gammaz);
  Kokkos::deep_copy(jacobian, h_jacobian);
}
