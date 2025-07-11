#include "specfem/assembly/mesh.hpp"

specfem::assembly::mesh_impl::points<specfem::dimension::type::dim3>::points(
    const specfem::mesh::mapping<dimension_tag> &mapping,
    const specfem::mesh::coordinates<dimension_tag> &coordinates)
    : nspec(mapping.nspec), ngllz(mapping.ngllz), nglly(mapping.nglly),
      ngllx(mapping.ngllx),
      index_mapping("specfem::assembly::mesh::points::index_mapping",
                    mapping.nspec, mapping.ngllz, mapping.nglly, mapping.ngllx),
      coord("specfem::assembly::mesh::points::coord", mapping.nspec,
            mapping.ngllz, mapping.nglly, mapping.ngllx, ndim),
      h_index_mapping(Kokkos::create_mirror_view(index_mapping)),
      h_coord(Kokkos::create_mirror_view(coord)) {

  // Initialize index mapping
  this->h_index_mapping = mapping.ibool;

  // Initialize coordinates
  Kokkos::parallel_for(
      "specfem::assembly::mesh::points::initialize_coordinates",
      Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace,
                            Kokkos::Rank<4> >(
          { 0, 0, 0, 0 },
          { mapping.nspec, mapping.ngllz, mapping.nglly, mapping.ngllx }),
      [&](const int ispec, const int iz, const int iy, const int ix) {
        const int iglob = h_index_mapping(ispec, iz, iy, ix);
        this->h_coord(ispec, iz, iy, ix, 0) = coordinates.x(iglob);
        this->h_coord(ispec, iz, iy, ix, 1) = coordinates.y(iglob);
        this->h_coord(ispec, iz, iy, ix, 2) = coordinates.z(iglob);
      });

  Kokkos::fence();

  // Copy to device
  Kokkos::deep_copy(this->coord, this->h_coord);
  Kokkos::deep_copy(this->index_mapping, this->h_index_mapping);
}
