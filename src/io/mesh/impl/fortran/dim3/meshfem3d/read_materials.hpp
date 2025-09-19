#include "io/mesh/impl/fortran/dim3/meshfem3d/read_materials.hpp"
#include "mesh/mesh.hpp"

specfem::mesh::meshfem3d::Materials<specfem::dimension::type::dim3>
specfem::io::mesh::impl::fortran::dim3::meshfem3d::read_materials(
    std::ifstream &stream, const specfem::MPI::MPI *mpi) {

  using MaterialsType =
      specfem::mesh::meshfem3d::Materials<specfem::dimension::type::dim3>;

  MaterialsType materials;

  // TODO (Rohit : TOMOGRAPHIC_MATERIALS)
  // We are currently not reading undefined materials which use tomographic
  // models. Add support for reading these materials later.
  int num_materials, num_undefined_materials;

  specfem::io::read_fortran_line(stream, &num_materials,
                                 &num_undefined_materials);

  for (int imat = 0; imat < num_materials; ++imat) {
    std::vector<type_real> material_properties(17, 0.0);
    specfem::io::read_fortran_line(stream, &material_properties);

    const int material_id = static_cast<int>(material_properties[6]);
    switch (material_id) {
    case 1: // Elastic or Acoustic material
    {
      const type_real rho = material_properties[0];
      const type_real vp = material_properties[1];
      const type_real vs = material_properties[2];
      const type_real Qkappa = material_properties[3];
      const type_real Qmu = material_properties[4];
      const int is_anisotropic = static_cast<int>(material_properties[5]);
      if (is_anisotropic <= 0) {
        if (specfem::utilities::is_close(vs, 0.0)) {
          // Acoustic material
          specfem::material::material<specfem::element::medium_tag::acoustic,
                                      specfem::element::property_tag::isotropic>
              material(rho, vp, Qkappa);
          materials.add_material(material);
        } else if (vs > 0.0) {
          // Isotropic elastic material
          specfem::material::material<specfem::element::medium_tag::elastic,
                                      specfem::element::property_tag::isotropic>
              material(rho, vp, vs, Qkappa, Qmu);
          materials.add_material(material);
        } else {
          throw std::runtime_error(
              "Shear wave velocity (Vs) cannot be negative for elastic "
              "materials.");
        }
      } else {
        // Anisotropic elastic material
        // TODO (Rohit: ANISOTROPIC_MATERIALS): Add support for anisotropic
        // materials
        throw std::runtime_error("Anisotropic elastic materials are not "
                                 "supported yet for 3D simulations.");
      }
      break;
    }
    case 2: {
      // Poroelastic material
      // TODO (Rohit: POROELASTIC_MATERIALS): Add support for poroelastic
      // materials
      throw std::runtime_error(
          "Poroelastic materials are not supported yet for 3D simulations.");
      break;
    }
    default:
      throw std::runtime_error("Unknown material ID: " +
                               std::to_string(material_id));
    }
  }

  // TODO (Rohit: TOMOGRAPHIC_MATERIALS): Add support for reading tomographic
  // materials
  for (int imat = 0; imat < num_undefined_materials; ++imat) {
    std::vector<type_real> dummy(6);
    specfem::io::read_fortran_line(stream, &dummy);
  }

  return materials;
}
