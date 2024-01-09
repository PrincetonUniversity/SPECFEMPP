// Find corner elements of the absorbing boundary
inline void find_corners(const specfem::kokkos::HostView1d<int> numabs,
                         const specfem::kokkos::HostView2d<bool> codeabs,
                         specfem::kokkos::HostView2d<bool> codeabscorner,
                         const int num_abs_boundary_faces,
                         const specfem::MPI::MPI *mpi) {
  int ncorner = 0;
  for (int inum = 0; inum < num_abs_boundary_faces; inum++) {
    if (codeabs(inum, 0)) {
      for (int inum_duplicate = 0; inum_duplicate < num_abs_boundary_faces;
           inum_duplicate++) {
        if (inum != inum_duplicate) {
          if (numabs(inum) == numabs(inum_duplicate)) {
            if (codeabs(inum_duplicate, 3)) {
              codeabscorner(inum, 1) = true;
              ncorner++;
            }
            if (codeabs(inum_duplicate, 1)) {
              codeabscorner(inum, 2) = true;
              ncorner++;
            }
          }
        }
      }
    }
    if (codeabs(inum, 2)) {
      for (int inum_duplicate = 0; inum_duplicate < num_abs_boundary_faces;
           inum_duplicate++) {
        if (inum != inum_duplicate) {
          if (numabs(inum) == numabs(inum_duplicate)) {
            if (codeabs(inum_duplicate, 3)) {
              codeabscorner(inum, 3) = true;
              ncorner++;
            }
            if (codeabs(inum_duplicate, 1)) {
              codeabscorner(inum, 4) = true;
              ncorner++;
            }
          }
        }
      }
    }
  }

  int ncorner_all = mpi->reduce(ncorner, specfem::MPI::sum);
  if (mpi->get_rank() == 0)
    assert(ncorner_all <= 4);
}

inline void calculate_ib(const specfem::kokkos::HostView2d<bool> code,
                         specfem::kokkos::HostView1d<int> ib_bottom,
                         specfem::kokkos::HostView1d<int> ib_top,
                         specfem::kokkos::HostView1d<int> ib_left,
                         specfem::kokkos::HostView1d<int> ib_right,
                         const int nelements) {

  int nspec_left = 0, nspec_right = 0, nspec_top = 0, nspec_bottom = 0;
  for (int inum = 0; inum < nelements; inum++) {
    if (code(inum, 0)) {
      ib_bottom(inum) = nspec_bottom;
      nspec_bottom++;
    } else if (code(inum, 1)) {
      ib_right(inum) = nspec_right;
      nspec_right++;
    } else if (code(inum, 2)) {
      ib_top(inum) = nspec_top;
      nspec_top++;
    } else if (code(inum, 3)) {
      ib_left(inum) = nspec_left;
      nspec_left++;
    } else {
      throw std::runtime_error("Incorrect acoustic boundary element type read");
    }
  }

  assert(nspec_left + nspec_right + nspec_bottom + nspec_top == nelements);
}
