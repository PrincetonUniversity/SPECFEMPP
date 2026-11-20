#include "assign_assembly_index_mapping.hpp"
#include "specfem/parallel_configuration.hpp"

namespace specfem::assembly::fields_impl {

void assign_assembly_index_mapping(
    Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::HostSpace>
        h_index_mapping,
    const specfem::assembly::element_types<
        specfem::element::dimension_tag::dim2> &element_types,
    Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace> dedup_table,
    int &nglob, const specfem::element::medium_tag MediumTag,
    const int base_dof) {

  const int nspec = static_cast<int>(h_index_mapping.extent(0));
  const int ngllz = static_cast<int>(h_index_mapping.extent(1));
  const int ngllx = static_cast<int>(h_index_mapping.extent(2));

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
            const int old_global = h_index_mapping(ispec, iz, ix);
            if (dedup_table(old_global) == -1) {
              dedup_table(old_global) = count;
              h_index_mapping(ispec, iz, ix) = base_dof + count;
              count++;
            } else {
              h_index_mapping(ispec, iz, ix) =
                  base_dof + dedup_table(old_global);
            }
          }
        }
      }
    }
  }

  nglob = count;
}

void assign_assembly_index_mapping(
    Kokkos::View<const int ****, Kokkos::LayoutLeft, Kokkos::HostSpace>
        h_mesh_index_mapping,
    Kokkos::View<int ****, Kokkos::LayoutLeft, Kokkos::HostSpace>
        h_local_index_mapping,
    const specfem::datatype::ElementIndexRange &elements,
    Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace> dedup_table,
    int &nglob) {

  const int begin = elements.begin_index();
  const int end = elements.end_index();
  const int ngllz = static_cast<int>(h_mesh_index_mapping.extent(1));
  const int nglly = static_cast<int>(h_mesh_index_mapping.extent(2));
  const int ngllx = static_cast<int>(h_mesh_index_mapping.extent(3));

  int count = 0;

  constexpr int chunk_size =
      specfem::parallel_configuration::storage_chunk_size;

  // Visit points in the same chunk order as a scan of the whole mesh, so the
  // numbering does not depend on where the medium's range starts.
  for (int ichunk = (begin / chunk_size) * chunk_size; ichunk < end;
       ichunk += chunk_size) {
    for (int iz = 0; iz < ngllz; iz++) {
      for (int iy = 0; iy < nglly; iy++) {
        for (int ix = 0; ix < ngllx; ix++) {
          for (int ielement = 0; ielement < chunk_size; ielement++) {
            const int ispec = ichunk + ielement;
            if (ispec >= end)
              break;
            if (ispec < begin)
              continue;
            const int mesh_point = h_mesh_index_mapping(ispec, iz, iy, ix);
            if (dedup_table(mesh_point) == -1) {
              dedup_table(mesh_point) = count;
              count++;
            }
            h_local_index_mapping(ispec - begin, iz, iy, ix) =
                dedup_table(mesh_point);
          }
        }
      }
    }
  }

  nglob = count;
}

} // namespace specfem::assembly::fields_impl
