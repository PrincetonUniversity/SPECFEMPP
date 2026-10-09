#include "specfem/assembly/coordinate_conversion.hpp"

#include "specfem/assembly/coordinate_conversion/dim2/coordinate_conversion.tpp"
#include "specfem/assembly/coordinate_conversion/dim3/coordinate_conversion.tpp"

// Cartesian target (source/receiver placement) -- one per model. These cover
// the sources constructor (dim2/Cartesian2D, dim3/Cartesian3D, dim3/Globe3D)
// and the dim3 receivers constructor (Cartesian3D, Globe3D).
template specfem::point::global_coordinates<
    specfem::element::dimension_tag::dim2>
specfem::assembly::to<specfem::coordinate_systems::cartesian_coordinates<
                          specfem::element::dimension_tag::dim2>,
                      specfem::simulation::model::Cartesian2D>(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim2> &,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &,
    const specfem::mesh::mesh<specfem::simulation::model::Cartesian2D> &);

template specfem::point::global_coordinates<
    specfem::element::dimension_tag::dim3>
specfem::assembly::to<specfem::coordinate_systems::cartesian_coordinates<
                          specfem::element::dimension_tag::dim3>,
                      specfem::simulation::model::Cartesian3D>(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &,
    const specfem::mesh::mesh<specfem::simulation::model::Cartesian3D> &);

template specfem::point::global_coordinates<
    specfem::element::dimension_tag::dim3>
specfem::assembly::to<specfem::coordinate_systems::cartesian_coordinates<
                          specfem::element::dimension_tag::dim3>,
                      specfem::simulation::model::Globe3D>(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &,
    const specfem::mesh::mesh<specfem::simulation::model::Globe3D> &);

// Cartesian -> geocentric (globe).
template specfem::coordinate_systems::geocentric_coordinates
specfem::assembly::to<specfem::coordinate_systems::geocentric_coordinates,
                      specfem::simulation::model::Globe3D>(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &,
    const specfem::mesh::mesh<specfem::simulation::model::Globe3D> &);

// Cartesian -> geographic (regional UTM, and globe).
template specfem::coordinate_systems::geographic_coordinates
specfem::assembly::to<specfem::coordinate_systems::geographic_coordinates,
                      specfem::simulation::model::Cartesian3D>(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &,
    const specfem::mesh::mesh<specfem::simulation::model::Cartesian3D> &);

template specfem::coordinate_systems::geographic_coordinates
specfem::assembly::to<specfem::coordinate_systems::geographic_coordinates,
                      specfem::simulation::model::Globe3D>(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &,
    const specfem::mesh::mesh<specfem::simulation::model::Globe3D> &);
