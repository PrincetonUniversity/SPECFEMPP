#include "specfem/io/mesh/impl/fortran/dim3/read_materials.hpp"
#include "specfem/io/fortranio/interface.hpp"
#include "specfem/medium_container.hpp"
#include "specfem/mesh.hpp"
#include <Kokkos_Core.hpp>
#include <fstream>
#include <tuple>
#include <vector>

std::tuple<int, int, int, int,
           Kokkos::View<int **, Kokkos::LayoutLeft, Kokkos::HostSpace>,
           specfem::mesh::materials<specfem::element::dimension_tag::dim3> >
specfem::io::mesh::impl::fortran::dim3::read_materials(
    std::ifstream &stream, const int ngnod, const bool attenuation_enabled) {

  using MaterialsType =
      specfem::mesh::materials<specfem::element::dimension_tag::dim3>;

  MaterialsType materials;

  // TODO (Rohit : TOMOGRAPHIC_MATERIALS)
  // We are currently not reading undefined materials which use tomographic
  // models. Add support for reading these materials later.
  int num_materials, num_undefined_materials;

  specfem::io::fortran_read_line(stream, &num_materials,
                                 &num_undefined_materials);

  std::vector<typename MaterialsType::material_specification> mapping;

  for (int imat = 0; imat < num_materials; ++imat) {
    std::vector<double> material_properties(17, 0.0);
    specfem::io::fortran_read_line(stream, &material_properties);

    const int material_id = static_cast<int>(material_properties[6]);
    switch (material_id) {
    case 1: // Acoustic
    case 2: // Elastic
    {
      const type_real rho = material_properties[0];
      const type_real vp = material_properties[1];
      const type_real vs = material_properties[2];
      const type_real Qkappa = material_properties[3];
      const type_real Qmu = material_properties[4];
      const int is_anisotropic = static_cast<int>(material_properties[5]);
      if (is_anisotropic <= 0) {
        if (specfem::utilities::is_close(vs, static_cast<type_real>(0.0))) {
          // Acoustic material
          if (material_id != 1) {
            throw std::runtime_error(
                "Shear wave velocity (Vs) cannot be zero for elastic "
                "materials.");
          }

          if (!((std::abs(Qmu - 9999.0) < 1e-6) || (std::abs(Qmu) < 1e-6))) {
            std::ostringstream error_message;
            error_message
                << "Qmu should be set to 9999 or 0 for acoustic materials. "
                << "Found Qmu = " << Qmu << " for material index " << imat
                << "." << "[" << __FILE__ << ":" << __LINE__ << "]\n";
            throw std::runtime_error(error_message.str());
          }

          if (!attenuation_enabled || (std::abs(Qkappa - 9999.0) < 1e-6)) {

            specfem::medium_container::material<
                specfem::element::dimension_tag::dim3,
                specfem::element::medium_tag::acoustic,
                specfem::element::property_tag::isotropic,
                specfem::element::attenuation_tag::none>
                material(rho, vp, static_cast<type_real>(0.0));
            const int index = materials.add_material(material);
            mapping.push_back({ specfem::element::medium_tag::acoustic,
                                specfem::element::property_tag::isotropic,
                                specfem::element::attenuation_tag::none, index,
                                imat });
          } else {
            specfem::medium_container::material<
                specfem::element::dimension_tag::dim3,
                specfem::element::medium_tag::acoustic,
                specfem::element::property_tag::isotropic,
                specfem::element::attenuation_tag::constant_isotropic>
                material(rho, vp, Qkappa, static_cast<type_real>(0.0));
            const int index = materials.add_material(material);
            mapping.push_back(
                { specfem::element::medium_tag::acoustic,
                  specfem::element::property_tag::isotropic,
                  specfem::element::attenuation_tag::constant_isotropic, index,
                  imat });
          }
        } else if (vs > 0.0) {
          // Isotropic elastic material
          if (material_id != 2) {
            throw std::runtime_error(
                "Shear wave velocity (Vs) cannot be nonzero for acoustic "
                "materials.");
          }

          if (!attenuation_enabled || (std::abs(Qkappa - 9999.0) < 1e-6 &&
                                       std::abs(Qmu - 9999.0) < 1e-6)) {
            specfem::medium_container::material<
                specfem::element::dimension_tag::dim3,
                specfem::element::medium_tag::elastic,
                specfem::element::property_tag::isotropic,
                specfem::element::attenuation_tag::none>
                material(rho, vs, vp, static_cast<type_real>(0.0));
            const int index = materials.add_material(material);
            mapping.push_back({ specfem::element::medium_tag::elastic,
                                specfem::element::property_tag::isotropic,
                                specfem::element::attenuation_tag::none, index,
                                imat });
          } else {
            specfem::medium_container::material<
                specfem::element::dimension_tag::dim3,
                specfem::element::medium_tag::elastic,
                specfem::element::property_tag::isotropic,
                specfem::element::attenuation_tag::constant_isotropic>
                material(rho, vs, vp, Qmu, Qkappa, static_cast<type_real>(0.0));
            const int index = materials.add_material(material);
            mapping.push_back(
                { specfem::element::medium_tag::elastic,
                  specfem::element::property_tag::isotropic,
                  specfem::element::attenuation_tag::constant_isotropic, index,
                  imat });
          }

        } else {
          throw std::runtime_error("Shear wave velocity (Vs) cannot be "
                                   "negative for any "
                                   "material.");
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
    case 3: {
      // Poroelastic material
      // TODO (Rohit: POROELASTIC_MATERIALS): Add support for poroelastic
      // materials
      throw std::runtime_error(
          "Poroelastic materials are not supported yet for 3D simulations.");
      break;
    }
    case 4: {
      // Elastic Cosserat material
      const type_real rho = material_properties[0];
      const type_real kappa = material_properties[1];
      const type_real mu = material_properties[2];
      const type_real nu = material_properties[3];
      const type_real j = material_properties[4];
      const type_real lambda_c = material_properties[5];
      // material_properties[6] is the material ID
      const type_real mu_c = material_properties[7];
      const type_real nu_c = material_properties[8];

      const type_real lambda = kappa - static_cast<type_real>(2.0) /
                                           static_cast<type_real>(3.0) * mu;

      if (mu < 0.0 || mu_c < 0.0 || 3 * lambda + 2 * mu < 0.0 ||
          3 * lambda_c + 2 * mu_c < 0.0) {
        throw std::runtime_error(
            "Invalid elastic parameters for Cosserat material. mu, mu_c, "
            "3*lambda + 2*mu, and 3*lambda_c + 2*mu_c must be non-negative.");
      }
      specfem::medium_container::material<
          specfem::element::dimension_tag::dim3,
          specfem::element::medium_tag::elastic_spin,
          specfem::element::property_tag::isotropic_cosserat,
          specfem::element::attenuation_tag::none>
          material(rho, kappa, mu, nu, j, lambda_c, mu_c, nu_c);
      const int index = materials.add_material(material);
      mapping.push_back({ specfem::element::medium_tag::elastic_spin,
                          specfem::element::property_tag::isotropic_cosserat,
                          specfem::element::attenuation_tag::none, index,
                          imat });
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
    specfem::io::fortran_read_line(stream, &dummy);
  }

  int nspec;
  specfem::io::fortran_read_line(stream, &nspec);
  Kokkos::View<int **, Kokkos::LayoutLeft, Kokkos::HostSpace>
      control_node_index("specfem::mesh::control_node_index", nspec, ngnod);

  int ngllz, nglly, ngllx;
  specfem::io::fortran_read_line(stream, &ngllz, &nglly, &ngllx);

  materials.material_index_mapping.resize(nspec);
  materials.nspec = nspec;
  for (int ispec = 0; ispec < nspec; ++ispec) {
    int index;
    int database_index;
    int tomographic_model;
    std::vector<int> control_nodes(ngnod, 0);
    specfem::io::fortran_read_line(stream, &index, &database_index,
                                   &tomographic_model, &control_nodes);
    if (index < 1 || index > nspec) {
      throw std::runtime_error("Error reading material indices");
    }
    if (database_index < 1 || database_index > num_materials) {
      throw std::runtime_error("Error reading material indices");
    }
    if (database_index < 0 && tomographic_model == 1) {
      // Deprecated funcitionality within MESHFEM3D
      throw std::runtime_error(
          "Interfaces are deprecated within 3D simulations.");
    }
    if (database_index < 0 && tomographic_model == 2) {
      // TODO (Rohit: TOMOGRAPHIC_MATERIALS): Add support for reading
      // tomographic materials
      throw std::runtime_error(
          "Tomographic materials are not supported yet for 3D simulations.");
    }
    materials.material_index_mapping[index - 1] = mapping[database_index - 1];
    for (int inode = 0; inode < ngnod; ++inode) {
      control_node_index(index - 1, inode) = control_nodes[inode] - 1;
    }
  }

  return std::make_tuple(nspec, ngllz, nglly, ngllx, control_node_index,
                         materials);
}
