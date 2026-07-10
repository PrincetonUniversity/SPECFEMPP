#include "specfem/algorithms/locate_point.hpp"

#include "specfem/assembly/mesh.hpp"
#include "specfem/jacobian.hpp"
#include "specfem/mesh.hpp"
#include "specfem/mesh_entity.hpp"
#include "specfem/mpi/mpi.hpp"
#include "specfem/point.hpp"

#include <Kokkos_Core.hpp>
#include <limits>
#include <stdexcept>
#include <tuple>

specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
specfem::algorithms::project_onto_surface(
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const specfem::mesh::acoustic_free_surface<
        specfem::element::dimension_tag::dim3> &surface,
    const specfem::point::global_coordinates<
        specfem::element::dimension_tag::dim3> &target,
    const specfem::algorithms::projection along) {

  // Only vertical (z) projection is implemented; see projection enum.
  if (along != specfem::algorithms::projection::along_z) {
    throw std::runtime_error(
        "specfem::algorithms::project_onto_surface: only projection::along_z "
        "is implemented");
  }

  constexpr double huge = std::numeric_limits<double>::max();

  const int nfaces = surface.nelem_acoustic_surface;

  double elevation = 0.0; // flat fallback when no free surface is found
  [[maybe_unused]] double elevation_distmin = huge; // only read in MPI builds

  if (nfaces > 0) {
    const auto &h_coord = mesh.h_coord;
    const auto &mesh_to_compute = mesh.h_mesh_to_compute;

    // Elements are cubic (ngllz == nglly == ngllx), so a face has ngll x ngll
    // GLL points regardless of orientation.
    const int ngll = mesh.element_grid.ngllx;
    const auto &element = mesh.element_grid;

    const auto node_xy = [&](int compute_ispec,
                             specfem::mesh_entity::dim3::type face, int ip,
                             int jp, double &nx, double &ny) {
      int iz, iy, ix;
      element.get_face_coordinates(face, ip, jp, iz, iy, ix);
      nx = h_coord(compute_ispec, iz, iy, ix, 0);
      ny = h_coord(compute_ispec, iz, iy, ix, 1);
    };

    // Stage 1: find the surface face whose node is nearest to (x, y). For the
    // usual SEM case (ngll >= 3) only interior nodes are considered, so nodes
    // shared between faces don't bias the choice; for ngll < 3 there are no
    // interior nodes, so all face nodes are used as a fallback.
    const int lo = (ngll > 2) ? 1 : 0;
    const int hi = (ngll > 2) ? ngll - 1 : ngll;
    double distmin = huge;
    int sel_face = -1;
    for (int f = 0; f < nfaces; ++f) {
      const int compute_ispec = mesh_to_compute(surface.index_mapping(f));
      const auto face = surface.type(f);
      for (int ip = lo; ip < hi; ++ip) {
        for (int jp = lo; jp < hi; ++jp) {
          double nx, ny;
          node_xy(compute_ispec, face, ip, jp, nx, ny);
          const double d = (target.x - nx) * (target.x - nx) +
                           (target.y - ny) * (target.y - ny);
          if (d < distmin) {
            distmin = d;
            sel_face = f;
          }
        }
      }
    }

    // Stage 2: intersect the vertical line through (x, y) with the chosen face.
    // The face fixes one reference coordinate; we Newton-solve the other two so
    // the interpolated (x, y) matches the target (z is ignored), then read z.
    if (sel_face >= 0) {
      const int compute_ispec =
          mesh_to_compute(surface.index_mapping(sel_face));
      const auto face = surface.type(sel_face);

      Kokkos::View<specfem::point::global_coordinates<
                       specfem::element::dimension_tag::dim3> *,
                   Kokkos::HostSpace>
          coorg("specfem::algorithms::project_onto_surface::coorg", mesh.ngnod);
      for (int i = 0; i < mesh.ngnod; ++i) {
        coorg(i).x = mesh.h_control_node_coordinates(compute_ispec, i, 0);
        coorg(i).y = mesh.h_control_node_coordinates(compute_ispec, i, 1);
        coorg(i).z = mesh.h_control_node_coordinates(compute_ispec, i, 2);
      }

      type_real xi = 0, eta = 0, gamma = 0;
      specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim3,
                                      true, false>
          jacobian;

      // Select the two free reference coordinates for this face and the inverse
      // Jacobian rows that map a horizontal (x, y) residual into their updates.
      // free1/free2 alias xi/eta/gamma and j1x..j2y alias members of
      // `jacobian`; because `jacobian` is reassigned in place below (same
      // object, members updated), those references stay valid across
      // iterations.
      auto [free1, free2, j1x, j1y, j2x, j2y] =
          [&xi, &eta, &gamma, &face,
           &jacobian]() -> std::tuple<type_real &, type_real &, type_real &,
                                      type_real &, type_real &, type_real &> {
        using specfem::mesh_entity::dim3::type;
        if (face == type::bottom || face == type::top) {
          gamma = (face == type::top) ? type_real(1) : type_real(-1);
          return { xi,           eta,           jacobian.xix,
                   jacobian.xiy, jacobian.etax, jacobian.etay };
        } else if (face == type::left || face == type::right) {
          xi = (face == type::right) ? type_real(1) : type_real(-1);
          return { eta,           gamma,           jacobian.etax,
                   jacobian.etay, jacobian.gammax, jacobian.gammay };
        } else { // front or back
          eta = (face == type::back) ? type_real(1) : type_real(-1);
          return { xi,           gamma,           jacobian.xix,
                   jacobian.xiy, jacobian.gammax, jacobian.gammay };
        }
      }();

      // Converged when the squared reference-coordinate update is negligible.
      // Accuracy is bounded by type_real precision (the mesh geometry is stored
      // in type_real); the 100-iteration cap below is the backstop.
      const type_real tol = type_real(1e-12);

      for (int iter = 0; iter < 100; ++iter) {
        const auto loc = specfem::jacobian::compute_locations(coorg, mesh.ngnod,
                                                              xi, eta, gamma);
        const type_real dx = target.x - loc.x;
        const type_real dy = target.y - loc.y;

        jacobian = specfem::jacobian::compute_jacobian(coorg, mesh.ngnod, xi,
                                                       eta, gamma);
        // Horizontal-only Gauss-Newton step (ignores z: vertical projection).
        const type_real d1 = j1x * dx + j1y * dy;
        const type_real d2 = j2x * dx + j2y * dy;

        free1 += d1;
        free2 += d2;
        free1 =
            (free1 > 1) ? type_real(1) : (free1 < -1 ? type_real(-1) : free1);
        free2 =
            (free2 > 1) ? type_real(1) : (free2 < -1 ? type_real(-1) : free2);

        if (d1 * d1 + d2 * d2 < tol)
          break;
      }

      const auto loc = specfem::jacobian::compute_locations(coorg, mesh.ngnod,
                                                            xi, eta, gamma);
      elevation = static_cast<double>(loc.z);
      elevation_distmin = distmin;
    }
  }

#ifdef SPECFEM_ENABLE_MPI
  // A source/receiver may project onto the free surface owned by another rank.
  // Select the elevation from the rank with the globally smallest distance.
  struct {
    double dist;
    int rank;
  } local{ elevation_distmin, specfem::MPI::get_rank() }, global;
  SPECFEM_MPI_SAFECALL(MPI_Allreduce(&local, &global, 1, MPI_DOUBLE_INT,
                                     MPI_MINLOC, specfem::MPI::communicator()));
  SPECFEM_MPI_SAFECALL(MPI_Bcast(&elevation, 1, MPI_DOUBLE, global.rank,
                                 specfem::MPI::communicator()));
#endif

  return { target.x, target.y, static_cast<type_real>(elevation) };
}
