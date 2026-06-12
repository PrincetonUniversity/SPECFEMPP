#include "assign_assembly_index_mapping.hpp"
#include "specfem/parallel_configuration.hpp"

namespace specfem::assembly::fields_impl {

template <>
void assign_assembly_index_mapping<specfem::element::dimension_tag::dim2>(
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh,
    const specfem::assembly::element_types<
        specfem::element::dimension_tag::dim2> &element_types,
    Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace>
        assembly_index_mapping,
    int &nglob, const specfem::element::medium_tag MediumTag) {
  const auto index_mapping = mesh.h_index_mapping;
  const int nspec = mesh.nspec;
  const int ngllz = mesh.element_grid.ngllz;
  const int ngllx = mesh.element_grid.ngllx;

  int count = 0;

  constexpr int chunk_size =
      specfem::parallel_configuration::storage_chunk_size;
  for (int ichunk = 0; ichunk < nspec; ichunk += chunk_size) {
    for (int ix = 0; ix < ngllx; ix++) {
      for (int iz = 0; iz < ngllz; iz++) {
        for (int ielement = 0; ielement < chunk_size; ielement++) {
          int ispec = ichunk + ielement;
          if (ispec >= nspec)
            break;
          const auto medium = element_types.get_medium_tag(ispec);
          if (medium == MediumTag) {
            const int global_index =
                index_mapping(ispec, iz, ix); // get global index
            // increase the count only if the global index is not already
            // counted
            /// static_cast<int>(medium::value) is the index of the medium in
            /// the enum class
            if (assembly_index_mapping(global_index) == -1) {
              assembly_index_mapping(global_index) = count;
              count++;
            }
          }
        }
      }
    }
  }

  nglob = count;

  return;
}

template <>
void assign_assembly_index_mapping<specfem::element::dimension_tag::dim3>(
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const specfem::assembly::element_types<
        specfem::element::dimension_tag::dim3> &element_types,
    Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace>
        assembly_index_mapping,
    int &nglob, const specfem::element::medium_tag MediumTag) {

  const auto index_mapping = mesh.h_index_mapping;
  const int nspec = mesh.nspec;
  const int ngllz = mesh.element_grid.ngllz;
  const int nglly = mesh.element_grid.nglly;
  const int ngllx = mesh.element_grid.ngllx;

  int count = 0;

  // Chunk-aware DOF numbering: iterate GLL points in outer loops and elements
  // within a chunk in the innermost loop.  For each (ichunk, iz, iy, ix), the
  // chunk_size elements are visited consecutively, so interior DOFs belonging
  // to the same GLL point within a chunk receive contiguous indices.  This
  // makes the SIMD gather in load_after_simd_dispatch stride-1 over the lane
  // dimension for interior points, collapsing 16 scattered loads into a single
  // cache-line read.
  constexpr int chunk_size =
      specfem::parallel_configuration::storage_chunk_size;

  for (int ichunk = 0; ichunk < nspec; ichunk += chunk_size) {
    for (int iz = 0; iz < ngllz; iz++) {
      for (int iy = 0; iy < nglly; iy++) {
        for (int ix = 0; ix < ngllx; ix++) {
          for (int ielement = 0; ielement < chunk_size; ielement++) {
            const int ispec = ichunk + ielement;
            if (ispec >= nspec)
              break;
            const auto medium = element_types.get_medium_tag(ispec);
            if (medium == MediumTag) {
              const int global_index = index_mapping(ispec, iz, iy, ix);
              if (assembly_index_mapping(global_index) == -1) {
                assembly_index_mapping(global_index) = count;
                count++;
              }
            }
          }
        }
      }
    }
  }

  nglob = count;

  return;
}

} // namespace specfem::assembly::fields_impl
