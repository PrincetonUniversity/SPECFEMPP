#include "injection_surface.hpp"
#include "injection_surface/dim3/injection_surface.tpp"

bool specfem::assembly::injection_surface_impl::is_face(
    const specfem::mesh_entity::dim3::type type) {
  using face = specfem::mesh_entity::dim3::type;
  return type == face::top || type == face::bottom || type == face::left ||
         type == face::right || type == face::front || type == face::back;
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
