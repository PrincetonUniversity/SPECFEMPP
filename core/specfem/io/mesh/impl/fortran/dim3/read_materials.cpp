#include "specfem/io/mesh/impl/fortran/dim3/read_materials.hpp"
#include "specfem/io/fortranio/interface.hpp"
#include "specfem/medium_container.hpp"
#include "specfem/mesh.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

specfem::io::mesh::impl::fortran::dim3_impl::stiffness_array
specfem::io::mesh::impl::fortran::dim3_impl::model_aniso(const int iflag_aniso,
                                                         const type_real rho,
                                                         const type_real vp,
                                                         const type_real vs) {

  // Perturbation factors, mirroring the parameters at the top of SPECFEM3D's
  // model_aniso.f90. Edit these to change the anisotropic halfspace model.
  // Only related to body waves; one-zeta term.
  constexpr type_real factor_cs1p_a = 0.2;
  constexpr type_real factor_cs1sv_a = 0.0;
  constexpr type_real factor_cs1sh_n = 0.0;
  // Three-zeta term.
  constexpr type_real factor_cs3_l = 0.0;
  // Relative to Love waves; four-zeta terms.
  constexpr type_real factor_n = 0.0;
  constexpr type_real factor_e_n = 0.0;
  // Relative to Rayleigh waves; two-zeta terms.
  constexpr type_real factor_a = 0.0;
  constexpr type_real factor_c = 0.0;
  constexpr type_real factor_f = 0.0;
  constexpr type_real factor_h_f = 0.0;
  constexpr type_real factor_b_a = 0.0;
  // Relative to both Love and Rayleigh waves; two-zeta terms.
  constexpr type_real factor_l = 0.0;
  constexpr type_real factor_g_l = 0.0;

  constexpr int anisotropy_model1 = 1;
  constexpr int anisotropy_model2 = 2;

  if (iflag_aniso > anisotropy_model2) {
    std::ostringstream message;
    message << "Anisotropy model flag " << iflag_aniso
            << " is not supported; expected 0, " << anisotropy_model1 << " or "
            << anisotropy_model2 << ".";
    throw std::runtime_error(message.str());
  }

  // Love parameters of the unperturbed (isotropic) medium. See Dziewonski &
  // Anderson (1981), PEPI 25, 297-356, page 305.
  const type_real vph = vp;
  const type_real vpv = vp;
  const type_real vsh = vs;
  const type_real vsv = vs;
  const type_real eta_aniso = 1.0;

  const type_real aa = rho * vph * vph;
  const type_real cc = rho * vpv * vpv;
  const type_real nn = rho * vsh * vsh;
  const type_real ll = rho * vsv * vsv;
  const type_real ff = eta_aniso * (aa - 2.0 * ll);

  // Anisotropic perturbation; notation follows Chen & Tromp (2006), appendix
  // A, page 1151.
  type_real a_param = 0.0, c_param = 0.0, an = 0.0, al = 0.0, f_param = 0.0;
  type_real c1p = 0.0, c1sv = 0.0, c1sh = 0.0;
  type_real s1p = 0.0, s1sv = 0.0, s1sh = 0.0;
  type_real gc = 0.0, gs = 0.0, bc = 0.0, bs = 0.0, hc = 0.0, hs = 0.0;
  type_real c3 = 0.0, s3 = 0.0, ec = 0.0, es = 0.0;

  if (iflag_aniso <= 0) {
    // No anisotropic perturbation: the isotropic-equivalent Voigt matrix.
    a_param = aa;
    c_param = cc;
    an = nn;
    al = ll;
    f_param = ff;
  } else {
    // Both perturbation models share every zeta-dependent term; model 2 adds a
    // further 10% to the zeta-independent parameters.
    const type_real extra = (iflag_aniso == anisotropy_model2) ? 0.1 : 0.0;

    a_param = aa * (1.0 + factor_a + extra);
    c_param = cc * (1.0 + factor_c + extra);
    an = nn * (1.0 + factor_n + extra);
    al = ll * (1.0 + factor_l + extra);
    f_param = ff * (1.0 + factor_f + extra);

    c1p = factor_cs1p_a * aa;
    c1sv = factor_cs1sv_a * aa;
    c1sh = factor_cs1sh_n * nn;

    gc = factor_g_l * ll;
    bc = factor_b_a * aa;
    hc = factor_h_f * ff;

    c3 = factor_cs3_l * ll;
    ec = factor_e_n * nn;
  }

  // Elastic tensor in the local geographic frame (1 South, 2 East, 3 up).
  const type_real d11 = a_param + ec + bc;
  const type_real d12 = a_param - 2.0 * an - ec;
  const type_real d13 = f_param + hc;
  const type_real d14 = s3 + 2.0 * s1sh + 2.0 * s1p;
  const type_real d15 = 2.0 * c1p + c3;
  const type_real d16 = -bs / 2.0 - es;
  const type_real d22 = a_param + ec - bc;
  const type_real d23 = f_param - hc;
  const type_real d24 = 2.0 * s1p - s3;
  const type_real d25 = 2.0 * c1p - 2.0 * c1sh - c3;
  const type_real d26 = -bs / 2.0 + es;
  const type_real d33 = c_param;
  const type_real d34 = 2.0 * (s1p - s1sv);
  const type_real d35 = 2.0 * (c1p - c1sv);
  const type_real d36 = -hs;
  const type_real d44 = al - gc;
  const type_real d45 = -gs;
  const type_real d46 = c1sh - c3;
  const type_real d55 = al + gc;
  const type_real d56 = s3 - s1sh;
  const type_real d66 = an - ec;

  // Rotate to the global Cartesian frame used by the solver
  // (1 East, 2 North, 3 up).
  return { d22, d12,  d23, -d25, d24, -d26, d11, d13, -d15, d14, -d16,
           d33, -d35, d34, -d36, d55, -d45, d56, d44, -d46, d66 };
}

std::tuple<int, int, int, int,
           Kokkos::View<int **, Kokkos::LayoutLeft, Kokkos::HostSpace>,
           specfem::mesh::materials<specfem::element::dimension_tag::dim3>>
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
                "Shear wave velocity (Vs) cannot be zero for elastic "
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
                material(rho, vs, vp, Qkappa, Qmu, static_cast<type_real>(0.0));
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
        // Anisotropic elastic material. The database carries only the
        // anisotropy flag alongside (rho, vp, vs), so the 21 stiffnesses are
        // reconstructed by the model_aniso port above, exactly as SPECFEM3D's
        // generate_databases does.
        if (material_id != 2) {
          throw std::runtime_error(
              "Anisotropic materials must be elastic (domain_id 2).");
        }

        if (specfem::utilities::is_close(vs, static_cast<type_real>(0.0))) {
          throw std::runtime_error("Shear wave velocity (Vs) cannot be zero "
                                   "for anisotropic elastic materials.");
        }

        if (vs < 0.0) {
          throw std::runtime_error("Shear wave velocity (Vs) cannot be "
                                   "negative for any material.");
        }

        if (attenuation_enabled && !(std::abs(Qkappa - 9999.0) < 1e-6 &&
                                     std::abs(Qmu - 9999.0) < 1e-6)) {
          std::ostringstream error_message;
          error_message
              << "Attenuation is not implemented for 3D anisotropic elastic "
              << "materials. Set Q_Kappa and Q_mu to 9999 for material index "
              << imat << ". " << "[" << __FILE__ << ":" << __LINE__ << "]\n";
          throw std::runtime_error(error_message.str());
        }

        const auto c = specfem::io::mesh::impl::fortran::dim3_impl::model_aniso(
            is_anisotropic, rho, vp, vs);

        specfem::medium_container::material<
            specfem::element::dimension_tag::dim3,
            specfem::element::medium_tag::elastic,
            specfem::element::property_tag::anisotropic,
            specfem::element::attenuation_tag::none>
            material(rho, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8],
                     c[9], c[10], c[11], c[12], c[13], c[14], c[15], c[16],
                     c[17], c[18], c[19], c[20]);
        const int index = materials.add_material(material);
        mapping.push_back({ specfem::element::medium_tag::elastic,
                            specfem::element::property_tag::anisotropic,
                            specfem::element::attenuation_tag::none, index,
                            imat });
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
