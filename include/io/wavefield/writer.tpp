#pragma once

#include "compute/interface.hpp"
#include "enumerations/interface.hpp"
#include "io/wavefield/writer.hpp"
#include "utilities/strings.hpp"

template <typename OutputLibrary>
specfem::io::wavefield_writer<OutputLibrary>::wavefield_writer(
    const std::string output_folder)
    : output_folder(output_folder), file(typename OutputLibrary::File(
                                        output_folder + "/ForwardWavefield")) {}

template <typename OutputLibrary>
void specfem::io::wavefield_writer<OutputLibrary>::write(
    specfem::compute::assembly &assembly) {

  auto &forward = assembly.fields.forward;
  auto &mesh = assembly.mesh;
  auto &element_types = assembly.element_types;

  using DomainView =
      Kokkos::View<type_real *, Kokkos::LayoutLeft, Kokkos::HostSpace>;

  const int ngllz = mesh.points.ngllz;
  const int ngllx = mesh.points.ngllx;

  typename OutputLibrary::Group base_group =
      file.createGroup(std::string("/Coordinates"));

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2),
       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC)),
      {
        const auto &field = forward.get_field<_medium_tag_>();

        typename OutputLibrary::Group group = base_group.createGroup(
            specfem::element::to_string(_medium_tag_));

        const auto element_indices =
            element_types.get_elements_on_host(_medium_tag_);
        const int n_elements = element_indices.size();

        DomainView x("xcoordinates", field.h_field.extent(0));
        DomainView z("zcoordinates", field.h_field.extent(0));

        for (int i = 0; i < n_elements; i++) {
          const int ispec = element_indices(i);
          for (int iz = 0; iz < ngllz; iz++) {
            for (int ix = 0; ix < ngllx; ix++) {
              const int iglob =
                  forward.template get_iglob<false>(i, iz, ix, _medium_tag_);
              x(iglob) = mesh.points.h_coord(0, ispec, iz, ix);
              z(iglob) = mesh.points.h_coord(1, ispec, iz, ix);
            }
          }
        }

        group.createDataset("X", x).write();
        group.createDataset("Z", z).write();
      });

  file.flush();

  std::cout << "Coordinates written to " << output_folder + "/ForwardWavefield"
            << std::endl;
}

template <typename OutputLibrary>
void specfem::io::wavefield_writer<OutputLibrary>::write(
    specfem::compute::assembly &assembly, const int istep) {
  auto &forward = assembly.fields.forward;
  auto &boundary_values = assembly.boundary_values;
  auto &element_types = assembly.element_types;
  auto &mesh = assembly.mesh;

  using DomainView =
      Kokkos::View<type_real *, Kokkos::LayoutLeft, Kokkos::HostSpace>;

  const int ngllz = mesh.points.ngllz;
  const int ngllx = mesh.points.ngllx;

  forward.copy_to_host();
  boundary_values.copy_to_host();

  typename OutputLibrary::Group base_group = file.createGroup(
      std::string("/Step") + specfem::utilities::to_zero_lead(istep, 6));

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2),
       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC)),
      {
        const auto &field = forward.get_field<_medium_tag_>();

        typename OutputLibrary::Group group = base_group.createGroup(
            specfem::element::to_string(_medium_tag_));

        if (_medium_tag_ == specfem::element::medium_tag::acoustic) {
          group.createDataset("Potential", field.h_field).write();
          group.createDataset("PotentialDot", field.h_field_dot).write();
          group.createDataset("PotentialDotDot", field.h_field_dot_dot).write();
        } else {
          group.createDataset("Displacement", field.h_field).write();
          group.createDataset("Velocity", field.h_field_dot).write();
          group.createDataset("Acceleration", field.h_field_dot_dot).write();
        }
      });

  typename OutputLibrary::Group boundary = base_group.createGroup("Boundary");
  typename OutputLibrary::Group stacey = boundary.createGroup("Stacey");

  {
    const auto element_indices = element_types.get_elements_on_host(
        specfem::element::medium_tag::poroelastic);
    const int n_elements = element_indices.size();
    DomainView x("xcoordinates", poroelastic_field.h_field.extent(0));
    DomainView z("zcoordinates", poroelastic_field.h_field.extent(0));
    for (int i = 0; i < n_elements; i++) {
      const int ispec = element_indices(i);
      for (int iz = 0; iz < ngllz; iz++) {
        for (int ix = 0; ix < ngllx; ix++) {
          const int iglob = forward.template get_iglob<false>(i, iz, ix, specfem::element::medium_tag::poroelastic);
          x(iglob) = mesh.points.h_coord(0, ispec, iz, ix);
          z(iglob) = mesh.points.h_coord(1, ispec, iz, ix);
        }
      }
    }
    poroelastic.createDataset("X", x).write();
    poroelastic.createDataset("Z", z).write();
  }

  stacey
      .createDataset("IndexMapping",
                     boundary_values.stacey.h_property_index_mapping)
      .write();
  stacey
      .createDataset("ElasticAcceleration",
                     boundary_values.stacey.elastic.h_values)
      .write();
  stacey
      .createDataset("AcousticAcceleration",
                     boundary_values.stacey.acoustic.h_values)
      .write();
  stacey
      .createDataset("PoroelasticAcceleration",
                     boundary_values.stacey.poroelastic.h_values)
      .write();

  file.flush();

  std::cout << "Wavefield written to " << output_folder + "/ForwardWavefield"
            << std::endl;
}
