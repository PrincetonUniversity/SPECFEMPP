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
      xix(Kokkos::ViewAllocateWithoutInitializing("xix"), mesh_jacobian.nspec,
          mesh_jacobian.ngllz, mesh_jacobian.nglly, mesh_jacobian.ngllx),
      xiy(Kokkos::ViewAllocateWithoutInitializing("xiy"), mesh_jacobian.nspec,
          mesh_jacobian.ngllz, mesh_jacobian.nglly, mesh_jacobian.ngllx),
      xiz(Kokkos::ViewAllocateWithoutInitializing("xiz"), mesh_jacobian.nspec,
          mesh_jacobian.ngllz, mesh_jacobian.nglly, mesh_jacobian.ngllx),
      etax(Kokkos::ViewAllocateWithoutInitializing("etax"), mesh_jacobian.nspec,
           mesh_jacobian.ngllz, mesh_jacobian.nglly, mesh_jacobian.ngllx),
      etay(Kokkos::ViewAllocateWithoutInitializing("etay"), mesh_jacobian.nspec,
           mesh_jacobian.ngllz, mesh_jacobian.nglly, mesh_jacobian.ngllx),
      etaz(Kokkos::ViewAllocateWithoutInitializing("etaz"), mesh_jacobian.nspec,
           mesh_jacobian.ngllz, mesh_jacobian.nglly, mesh_jacobian.ngllx),
      gammax(Kokkos::ViewAllocateWithoutInitializing("gammax"),
             mesh_jacobian.nspec, mesh_jacobian.ngllz, mesh_jacobian.nglly,
             mesh_jacobian.ngllx),
      gammay(Kokkos::ViewAllocateWithoutInitializing("gammay"),
             mesh_jacobian.nspec, mesh_jacobian.ngllz, mesh_jacobian.nglly,
             mesh_jacobian.ngllx),
      gammaz(Kokkos::ViewAllocateWithoutInitializing("gammaz"),
             mesh_jacobian.nspec, mesh_jacobian.ngllz, mesh_jacobian.nglly,
             mesh_jacobian.ngllx),
      jacobian(Kokkos::ViewAllocateWithoutInitializing("jacobian"),
               mesh_jacobian.nspec, mesh_jacobian.ngllz, mesh_jacobian.nglly,
               mesh_jacobian.ngllx) {

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
