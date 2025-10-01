#include "specfem/assembly/jacobian_matrix.hpp"
#include "mesh/mesh.hpp"

specfem::assembly::jacobian_matrix<specfem::dimension::type::dim3>::
    jacobian_matrix(
        const specfem::mesh::jacobian_matrix<dimension_tag> &mesh_jacobian)
    : xix("specfem::assembly::jacobian_matrix::xix", mesh_jacobian.nspec,
          mesh_jacobian.ngllz, mesh_jacobian.nglly, mesh_jacobian.ngllx),
      xiy("specfem::assembly::jacobian_matrix::xiy", mesh_jacobian.nspec,
          mesh_jacobian.ngllz, mesh_jacobian.nglly, mesh_jacobian.ngllx),
      xiz("specfem::assembly::jacobian_matrix::xiz", mesh_jacobian.nspec,
          mesh_jacobian.ngllz, mesh_jacobian.nglly, mesh_jacobian.ngllx),
      etax("specfem::assembly::jacobian_matrix::etax", mesh_jacobian.nspec,
           mesh_jacobian.ngllz, mesh_jacobian.nglly, mesh_jacobian.ngllx),
      etay("specfem::assembly::jacobian_matrix::etay", mesh_jacobian.nspec,
           mesh_jacobian.ngllz, mesh_jacobian.nglly, mesh_jacobian.ngllx),
      etaz("specfem::assembly::jacobian_matrix::etaz", mesh_jacobian.nspec,
           mesh_jacobian.ngllz, mesh_jacobian.nglly, mesh_jacobian.ngllx),
      gammax("specfem::assembly::jacobian_matrix::gammax", mesh_jacobian.nspec,
             mesh_jacobian.ngllz, mesh_jacobian.nglly, mesh_jacobian.ngllx),
      gammay("specfem::assembly::jacobian_matrix::gammay", mesh_jacobian.nspec,
             mesh_jacobian.ngllz, mesh_jacobian.nglly, mesh_jacobian.ngllx),
      gammaz("specfem::assembly::jacobian_matrix::gammaz", mesh_jacobian.nspec,
             mesh_jacobian.ngllz, mesh_jacobian.nglly, mesh_jacobian.ngllx),
      jacobian("specfem::assembly::jacobian_matrix::jacobian",
               mesh_jacobian.nspec, mesh_jacobian.ngllz, mesh_jacobian.nglly,
               mesh_jacobian.ngllx),
      h_xix(Kokkos::create_mirror_view(xix)),
      h_xiy(Kokkos::create_mirror_view(xiy)),
      h_xiz(Kokkos::create_mirror_view(xiz)),
      h_etax(Kokkos::create_mirror_view(etax)),
      h_etay(Kokkos::create_mirror_view(etay)),
      h_etaz(Kokkos::create_mirror_view(etaz)),
      h_gammax(Kokkos::create_mirror_view(gammax)),
      h_gammay(Kokkos::create_mirror_view(gammay)),
      h_gammaz(Kokkos::create_mirror_view(gammaz)),
      h_jacobian(Kokkos::create_mirror_view(jacobian)) {

  nspec = mesh_jacobian.nspec;
  ngllx = mesh_jacobian.ngllx;
  nglly = mesh_jacobian.nglly;
  ngllz = mesh_jacobian.ngllz;

  // Initialize the Kokkos view with single values
  Kokkos::deep_copy(h_xix, mesh_jacobian.xix_regular);
  Kokkos::deep_copy(h_xiy, mesh_jacobian.xix_regular);
  Kokkos::deep_copy(h_xiz, mesh_jacobian.xix_regular);
  Kokkos::deep_copy(h_etax, mesh_jacobian.xix_regular);
  Kokkos::deep_copy(h_etay, mesh_jacobian.xix_regular);
  Kokkos::deep_copy(h_etaz, mesh_jacobian.xix_regular);
  Kokkos::deep_copy(h_gammax, mesh_jacobian.xix_regular);
  Kokkos::deep_copy(h_gammay, mesh_jacobian.xix_regular);
  Kokkos::deep_copy(h_gammaz, mesh_jacobian.xix_regular);
  Kokkos::deep_copy(h_jacobian, mesh_jacobian.jacobian_regular);

  if (mesh_jacobian.nspec_irregular > 0) {

    for (int ispec = 0; ispec < nspec; ispec++) {
      // if element number is irregular
      if (mesh_jacobian.irregular_element_number(ispec)) {
        for (int iz = 0; iz < mesh_jacobian.ngllz; iz++) {
          for (int iy = 0; iy < mesh_jacobian.nglly; iy++) {
            for (int ix = 0; ix < mesh_jacobian.ngllx; ix++) {
              h_xix(ispec, iz, iy, ix) = mesh_jacobian.xix(ispec, iz, iy, ix);
              h_xiy(ispec, iz, iy, ix) = mesh_jacobian.xiy(ispec, iz, iy, ix);
              h_xiz(ispec, iz, iy, ix) = mesh_jacobian.xiz(ispec, iz, iy, ix);
              h_etax(ispec, iz, iy, ix) = mesh_jacobian.etax(ispec, iz, iy, ix);
              h_etay(ispec, iz, iy, ix) = mesh_jacobian.etay(ispec, iz, iy, ix);
              h_etaz(ispec, iz, iy, ix) = mesh_jacobian.etaz(ispec, iz, iy, ix);
              h_gammax(ispec, iz, iy, ix) =
                  mesh_jacobian.gammax(ispec, iz, iy, ix);
              h_gammay(ispec, iz, iy, ix) =
                  mesh_jacobian.gammay(ispec, iz, iy, ix);
              h_gammaz(ispec, iz, iy, ix) =
                  mesh_jacobian.gammaz(ispec, iz, iy, ix);
              h_jacobian(ispec, iz, iy, ix) =
                  mesh_jacobian.jacobian(ispec, iz, iy, ix);
            }
          }
        }
      }
    }
  }

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

  std::cout << "Checking for small Jacobian values..." << std::endl;

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
