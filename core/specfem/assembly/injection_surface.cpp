#include "injection_surface.hpp"
#include "injection_surface/dim3/injection_surface.tpp"

bool specfem::assembly::injection_surface_impl::is_face(
    const specfem::mesh_entity::dim3::type type) {
  using face = specfem::mesh_entity::dim3::type;
  return type == face::top || type == face::bottom || type == face::left ||
         type == face::right || type == face::front || type == face::back;
}

bool specfem::assembly::injection_surface_impl::stacey_keeps_face(
    const specfem::assembly::boundaries<specfem::element::dimension_tag::dim3>
        &boundaries,
    const int ispec, const int iz, const int iy, const int ix,
    const std::array<type_real, 3> &normal) {
  const int local = boundaries.h_stacey_index_mapping(ispec);
  if (local < 0 || boundaries.stacey.h_face_weight(local, iz, iy, ix) == 0)
    return true;
  for (int d = 0; d < 3; ++d)
    if (boundaries.stacey.h_face_normal(local, iz, iy, ix, d) != normal[d])
      return false;
  return true;
}

template specfem::assembly::
    InjectionSurface<specfem::element::dimension_tag::dim3>::InjectionSurface(
        const specfem::mesh::cartesian3d_mesh &mesh,
        const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
            &mesh_assembly,
        const specfem::assembly::jacobian_matrix<
            specfem::element::dimension_tag::dim3> &jacobian_matrix,
        const specfem::assembly::element_types<
            specfem::element::dimension_tag::dim3> &element_types,
        const specfem::assembly::boundaries<
            specfem::element::dimension_tag::dim3> &boundaries);

template specfem::assembly::
    InjectionSurface<specfem::element::dimension_tag::dim3>::InjectionSurface(
        const specfem::mesh::globe3d_mesh &mesh,
        const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
            &mesh_assembly,
        const specfem::assembly::jacobian_matrix<
            specfem::element::dimension_tag::dim3> &jacobian_matrix,
        const specfem::assembly::element_types<
            specfem::element::dimension_tag::dim3> &element_types,
        const specfem::assembly::boundaries<
            specfem::element::dimension_tag::dim3> &boundaries);
