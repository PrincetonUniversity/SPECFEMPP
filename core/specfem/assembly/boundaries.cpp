#include "boundaries.hpp"
#include "boundaries/dim2/boundaries.tpp"
#include "boundaries/dim2/impl/acoustic_free_surface.tpp"
#include "boundaries/dim2/impl/stacey.tpp"
#include "boundaries/dim3/boundaries.tpp"
#include "boundaries/dim3/impl/acoustic_free_surface.tpp"
#include "boundaries/dim3/impl/stacey.tpp"

template specfem::assembly::boundaries<specfem::element::dimension_tag::dim3>::
    boundaries(
        const int nspec, const int ngllz, const int nglly, const int ngllx,
        const specfem::mesh::cartesian3d_mesh &mesh,
        const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
            &mesh_assembly,
        const specfem::assembly::jacobian_matrix<
            specfem::element::dimension_tag::dim3> &jacobian_matrix);

template specfem::assembly::boundaries<specfem::element::dimension_tag::dim3>::
    boundaries(
        const int nspec, const int ngllz, const int nglly, const int ngllx,
        const specfem::mesh::globe3d_mesh &mesh,
        const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
            &mesh_assembly,
        const specfem::assembly::jacobian_matrix<
            specfem::element::dimension_tag::dim3> &jacobian_matrix);
