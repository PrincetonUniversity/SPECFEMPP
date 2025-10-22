// generate_data function generates wavefield data for compute_wavefield tests
// We pick a single element for different medium_tag and property_tag
// combinations and assign 1.0 to all the quadrature points in the element

#pragma once
#include "enumerations/interface.hpp"
#include "specfem/point.hpp"

// Factor for arbitrary field values
template <specfem::wavefield::type component> struct field_factor;

template <> struct field_factor<specfem::wavefield::type::displacement> {
  static constexpr type_real f_iz[] = { 1.1, 1.7 };
  static constexpr type_real f_ix[] = { 2.6, 0.0 };
  static constexpr type_real f_c[] = { 0.0, 2.3 };
};

template <> struct field_factor<specfem::wavefield::type::velocity> {
  static constexpr type_real f_iz[] = { 0.0, 0.0 };
  static constexpr type_real f_ix[] = { 0.0, 0.0 };
  static constexpr type_real f_c[] = { 1.0, 1.0 };
};

template <> struct field_factor<specfem::wavefield::type::acceleration> {
  static constexpr type_real f_iz[] = { 1.3, 0.0 };
  static constexpr type_real f_ix[] = { 0.0, 0.0 };
  static constexpr type_real f_c[] = { 0.0, 0.0 };
};

template <specfem::wavefield::type component, typename PointType>
void assign_field(PointType &point, const int num_components, const int iz,
                  const int ix) {
  for (int i = 0; i < num_components; i++) {
    point(i) = field_factor<component>::f_iz[i] * iz +
               field_factor<component>::f_ix[i] * ix +
               field_factor<component>::f_c[i];
  }
}

const type_real jacobian_fac[] = { 1.0, 1.4 };

template <specfem::wavefield::type component,
          specfem::wavefield::simulation_field type,
          specfem::element::medium_tag medium,
          specfem::element::property_tag property>
void generate_data(
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly,
    std::vector<int> &ispecs) {

  auto field = assembly.fields.template get_simulation_field<type>();

  const int ngllx = assembly.mesh.element_grid.ngllx;
  const int ngllz = assembly.mesh.element_grid.ngllz;

  const auto elements =
      assembly.element_types.get_elements_on_host(medium, property);

  constexpr int num_components =
      specfem::element::attributes<specfem::dimension::type::dim2,
                                   medium>::components;

  using PointDisplacementType =
      specfem::point::displacement<specfem::dimension::type::dim2, medium,
                                   false>;
  using PointVelocityType =
      specfem::point::velocity<specfem::dimension ::type::dim2, medium, false>;
  using PointAccelerationType =
      specfem::point::acceleration<specfem::dimension::type::dim2, medium,
                                   false>;

  using IndexType =
      specfem::point::index<specfem::dimension::type::dim2, false>;

  const int nelements = elements.size();

  if (nelements == 0)
    return;

  const int ispec = elements(nelements / 2);
  ispecs.push_back(ispec);

  for (int iz = 0; iz < ngllz; iz++) {
    for (int ix = 0; ix < ngllx; ix++) {
      const IndexType index(ispec, iz, ix);

      PointDisplacementType displacement;
      PointVelocityType velocity;
      PointAccelerationType acceleration;

      for (int icomp = 0; icomp < num_components; icomp++) {
        assign_field<specfem::wavefield::type::displacement>(
            displacement, num_components, iz, ix);
        assign_field<specfem::wavefield::type::velocity>(
            velocity, num_components, iz, ix);
        assign_field<specfem::wavefield::type::acceleration>(
            acceleration, num_components, iz, ix);
      }

      specfem::assembly::store_on_host(index, field, displacement, velocity,
                                       acceleration);
    }
  }

  field.copy_to_device();
}

template <specfem::wavefield::type component,
          specfem::wavefield::simulation_field type>
std::vector<int> generate_data(
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly) {

  std::vector<int> ispecs;

  generate_data<component, type, specfem::element::medium_tag::elastic_psv,
                specfem::element::property_tag::isotropic>(assembly, ispecs);

  generate_data<component, type, specfem::element::medium_tag::acoustic,
                specfem::element::property_tag::isotropic>(assembly, ispecs);

  return ispecs;
}
