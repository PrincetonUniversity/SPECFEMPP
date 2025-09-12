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

std::tuple<bool, Kokkos::View<bool *, Kokkos::DefaultHostExecutionSpace> >
specfem::assembly::jacobian_matrix<
    specfem::dimension::type::dim3>::check_small_jacobian() const {
  Kokkos::View<bool *, Kokkos::DefaultHostExecutionSpace> small_jacobian(
      "specfem::assembly::jacobian_matrix::negative", nspec);

  Kokkos::deep_copy(small_jacobian, false);

  const type_real threshold = 1e-10;

  using PointJacobianMatrixType =
      specfem::point::jacobian_matrix<dimension_tag, true, false>;

  bool found = false;
  Kokkos::parallel_reduce(
      "specfem::assembly::jacobian_matrix::check_small_jacobian",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, nspec),
      [=, *this](const int &ispec, bool &l_found) {
        for (int iz = 0; iz < ngllz; ++iz) {
          for (int iy = 0; iy < nglly; ++iy) {
            for (int ix = 0; ix < ngllx; ++ix) {
              // Define the local_index
              const specfem::point::index<dimension_tag, false> index(ispec, iz,
                                                                      iy, ix);

              // Get the Jacobian determinant
              const auto jacobian = [&]() {
                PointJacobianMatrixType jacobian_matrix;
                specfem::assembly::load_on_host(index, *this, jacobian_matrix);
                return jacobian_matrix.jacobian;
              }();

              // Check if below threshold
              if (jacobian < threshold) {
                small_jacobian(ispec) = true;
                l_found = true;
                break;
              }
            }
          }
        }
      },
      found);

  return std::make_tuple(found, small_jacobian);
}
