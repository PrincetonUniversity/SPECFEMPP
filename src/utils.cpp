#include "../include/utils.h"
#include "../include/jacobian.h"
#include "../include/kokkos_abstractions.h"
#include "../include/specfem_mpi.h"
#include <tuple>
#include <vector>

std::vector<int> get_best_candidates(const int ispec_guess,
                                     const specfem::HostView3d<int> ibool) {

  const int nspec = ibool.extent(0);
  const int ngllx = ibool.extent(1);
  const int ngllz = ibool.extent(2);

  std::vector<int> iglob_guess;
  iglob_guess.push_back(ibool(ispec_guess, 0, 0));
  iglob_guess.push_back(ibool(ispec_guess, ngllz - 1, 0));
  iglob_guess.push_back(ibool(ispec_guess, 0, ngllx - 1));
  iglob_guess.push_back(ibool(ispec_guess, ngllz - 1, ngllx - 1));

  std::vector<int> ispec_candidates;
  ispec_candidates.push_back(ispec_guess);

  for (int ispec = 0; ispec < nspec; ispec++) {
    if (ispec == ispec_guess)
      continue;

    // loop over only corners
    for (int j : { 0, ngllz - 1 }) {
      for (int i : { 0, ngllx - 1 }) {
        // check if this element is in contact with initial guess
        if (std::find(iglob_guess.begin(), iglob_guess.end(),
                      ibool(ispec, j, i)) != iglob_guess.end()) {
          // do not count the element twice
          if (ispec_candidates.size() > 0 &&
              ispec_candidates[ispec_candidates.size() - 1] != ispec)
            ispec_candidates.push_back(ispec);
        }
      }
    }
  }

  return ispec_candidates;
}

std::tuple<int, int, int>
rough_location(const type_real x_source, const type_real z_source,
               const specfem::HostView3d<int> ibool,
               const specfem::HostView2d<type_real> coord) {

  /***
   *  Roughly locate closest quadrature point to the source
   ***/

  const int nspec = ibool.extent(0);
  const int ngllx = ibool.extent(1);
  const int ngllz = ibool.extent(2);
  type_real dist_min = std::numeric_limits<type_real>::max();
  type_real dist_squared;

  int ispec_selected, ix_selected, iz_selected;

  for (int ispec = 0; ispec < nspec; ispec++) {
    for (int j = 0; j < ngllz; j++) {
      for (int i = 0; i < ngllx; i++) {
        int iglob = ibool(ispec, j, i);
        dist_squared =
            (x_source - coord(0, iglob)) * (x_source - coord(0, iglob)) +
            (z_source - coord(1, iglob)) * (z_source - coord(1, iglob));
        if (dist_squared < dist_min) {
          ispec_selected = ispec;
          ix_selected = i;
          iz_selected = j;
          dist_min = dist_squared;
        }
      }
    }
  }

  return std::make_tuple(ix_selected, iz_selected, ispec_selected);
}

std::tuple<type_real, type_real>
get_best_location(const type_real x_source, const type_real z_source,
                  const specfem::HostView2d<type_real> coorg, type_real xi,
                  type_real gamma, const specfem::HostView2d<int> knods,
                  const int ispec) {

  int ngnod = knods.extent(0);

  specfem::HostView2d<type_real> s_coorg("get_best_location::s_coorg", ndim,
                                         ngnod);

  // Store s_coorg for better caching
  for (int in = 0; in < ngnod; in++) {
    s_coorg(0, in) = coorg(0, knods(in, ispec));
    s_coorg(1, in) = coorg(1, knods(in, ispec));
  }

  for (int iter_loop = 0; iter_loop < 5; iter_loop++) {
    auto [x, z] = jacobian::compute_locations(s_coorg, ngnod, xi, gamma);
    auto [xix, xiz, gammax, gammaz] =
        jacobian::compute_inverted_derivatives(s_coorg, ngnod, xi, gamma);

    type_real dx = -(x - x_source);
    type_real dz = -(z - z_source);

    type_real dxi = xix * dx + xiz * dz;
    type_real dgamma = gammax * dx + gammaz * dz;

    xi += dxi;
    gamma += dgamma;

    if (xi > 1.01)
      xi = 1.01;
    if (xi < -1.01)
      xi = -1.01;
    if (gamma > 1.01)
      gamma = 1.01;
    if (gamma < -1.01)
      gamma = -1.01;
  }

  return std::make_tuple(xi, gamma);
}

int get_islice(const int final_dist, const specfem::MPI::MPI *mpi) {

  int islice = 0;
  type_real glob_final_dist = mpi->all_reduce(final_dist, specfem::MPI::min);

  // check if the source lies on currrent proc
  if ((fabs(glob_final_dist - final_dist)) < 1e-6)
    islice = mpi->get_rank();

  std::vector<int> all_slice = mpi->gather(islice);

  // checking if multiple mpi ranks get assigned the source
  if (mpi->main_proc()) {
    bool already_assigned = false;
    for (int i = all_slice.size() - 1; i >= 0; i--) {
      // assign the source the maximum processor number
      // replicated from fortran code
      if (all_slice[i] == i && already_assigned) {
        mpi->cout("Multiple processors were assigned the same data");
        all_slice[i] = 0;
      } else if (all_slice[i] == i) {
        already_assigned = true;
      }
    }

    if (!already_assigned) {
      throw std::runtime_error("Source was not assigned to any slice");
    }
  }

  islice = mpi->scatter(all_slice);

  return islice;
}

std::tuple<type_real, type_real, int, int>
specfem::utilities::locate(const specfem::HostView2d<type_real> coord,
                           const specfem::HostMirror3d<int> ibool,
                           const specfem::HostMirror1d<type_real> xigll,
                           const specfem::HostMirror1d<type_real> zigll,
                           const int nproc, const type_real x_source,
                           const type_real z_source,
                           const specfem::HostView2d<type_real> coorg,
                           const specfem::HostView2d<int> knods,
                           const int npgeo, const specfem::MPI::MPI *mpi) {

  const int nspec = ibool.extent(0);
  const int ngllx = ibool.extent(1);
  const int ngllz = ibool.extent(2);
  // get closest quadrature point to source
  auto [ix_guess, iz_guess, ispec_guess] =
      rough_location(x_source, z_source, ibool, coord);
  int ngnod = knods.extent(0);

  assert(ispec_guess < nspec);

  // get best candidates to search
  auto ispec_candidates = get_best_candidates(ispec_guess, ibool);
  specfem::HostView2d<type_real> s_coorg("specfem::utilities::locate::s_coorg",
                                         ndim, ngnod);

  type_real final_dist = std::numeric_limits<type_real>::max();
  int ispec_selected_source;
  type_real xi_source, gamma_source;

  for (int i = 0; i < ispec_candidates.size(); i++) {
    if (i > 0) {
      ix_guess = int(ngllx / 2.0);
      iz_guess = int(ngllz / 2.0);
    }

    int ispec = ispec_candidates[i];

    type_real xi = xigll(ix_guess);
    type_real gamma = zigll(iz_guess);

    if (use_best_location)
      std::tie(xi, gamma) =
          get_best_location(x_source, z_source, coorg, xi, gamma, knods, ispec);

    // Store s_coorg for better caching
    for (int in = 0; in < ngnod; in++) {
      s_coorg(0, in) = coorg(0, knods(in, ispec));
      s_coorg(1, in) = coorg(1, knods(in, ispec));
    }

    auto [x, z] = jacobian::compute_locations(s_coorg, ngnod, xi, gamma);

    type_real final_distance_this_element = std::sqrt(
        (x_source - x) * (x_source - x) + (z_source - z) * (z_source - z));

    if (final_distance_this_element < final_dist) {
      ispec_selected_source = ispec;
      xi_source = xi;
      gamma_source = gamma;
      final_dist = final_distance_this_element;
    }
  }

#ifdef MPI_PARALLEL
  int islice = get_islice(final_dist, mpi);
#else
  int islice = 0;
#endif

  mpi->bcast(xi_source, islice);
  mpi->bcast(gamma_source, islice);
  mpi->bcast(ispec_selected_source, islice);

  return std::make_tuple(xi_source, gamma_source, ispec_selected_source,
                         islice);
}

void specfem::utilities::check_locations(const type_real x, const type_real z,
                                         const type_real xmin,
                                         const type_real xmax,
                                         const type_real zmin,
                                         const type_real zmax,
                                         const specfem::MPI::MPI *mpi) {
  // Check if the source is inside the domain

  type_real global_xmin = mpi->reduce(xmin, specfem::MPI::min);
  type_real global_xmax = mpi->reduce(xmax, specfem::MPI::max);
  type_real global_zmin = mpi->reduce(zmin, specfem::MPI::min);
  type_real global_zmax = mpi->reduce(xmax, specfem::MPI::max);

  if (mpi->get_main()) {
    if (x < global_xmin || x > global_xmax || z < global_zmin ||
        z > global_zmax) {
      std::ostringstream message;
      message << "Source at position (x,z) = " << x << " " << z
              << "Is located outside of mesh"
              << " Mesh dimenstions are (xmin, xmax, zmin, zmax) = " << xmin
              << " " << xmax << " " << zmin << " " << zmax;
      throw std::runtime_error(message.str());
    }
  }

  return;
}

int specfem::utilities::compute_nglob(const specfem::HostMirror3d<int> ibool) {

  const int nspec = ibool.extent(0);
  const int ngllz = ibool.extent(1);
  const int ngllx = ibool.extent(2);

  type_real nglob;
  Kokkos::parallel_reduce(
      "specfem::utils::compute_nglob",
      specfem::HostMDrange<3>({ 0, 0, 0 }, { nspec, ngllz, ngllx }),
      KOKKOS_LAMBDA(const int ispec, const int iz, const int ix,
                    type_real &l_nglob) {
        l_nglob =
            l_nglob > ibool(ispec, iz, ix) ? l_nglob : ibool(ispec, iz, ix);
      },
      Kokkos::Max<type_real>(nglob));

  return nglob;
}
