#include "compute/interface.hpp"
#include "kokkos_abstractions.h"
#include "material/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <memory>

// specfem::compute::properties::properties(const int nspec, const int ngllz,
//                                          const int ngllx)
//     : rho(specfem::kokkos::DeviceView3d<type_real>(
//           "specfem::compute::properties::rho", nspec, ngllz, ngllx)),
//       mu(specfem::kokkos::DeviceView3d<type_real>(
//           "specfem::compute::properties::mu", nspec, ngllz, ngllx)),
//       kappa(specfem::kokkos::DeviceView3d<type_real>(
//           "specfem::compute::properties::kappa", nspec, ngllz, ngllx)),
//       qmu(specfem::kokkos::HostView3d<type_real>(
//           "specfem::compute::properties::qmu", nspec, ngllz, ngllx)),
//       qkappa(specfem::kokkos::HostView3d<type_real>(
//           "specfem::compute::properties::qkappa", nspec, ngllz, ngllx)),
//       rho_vp(specfem::kokkos::HostView3d<type_real>(
//           "specfem::compute::properties::rho_vp", nspec, ngllz, ngllx)),
//       rho_vs(specfem::kokkos::HostView3d<type_real>(
//           "specfem::compute::properties::rho_vs", nspec, ngllz, ngllx)),
//       lambdaplus2mu(specfem::kokkos::DeviceView3d<type_real>(
//           "specfem::compute::properties::lambdaplus2mu", nspec, ngllz,
//           ngllx)),
//       ispec_type(specfem::kokkos::DeviceView1d<specfem::enums::element::type>(
//           "specfem::compute::properties::ispec_type", nspec)),
//       rho_inverse(specfem::kokkos::DeviceView3d<type_real>(
//           "specfem::compute::properties::rho_inverse", nspec, ngllz, ngllx)),
//       lambdaplus2mu_inverse(specfem::kokkos::DeviceView3d<type_real>(
//           "specfem::compute::properties::lambdaplus2mu_inverse", nspec,
//           ngllz, ngllx)) {

//   h_rho = Kokkos::create_mirror_view(rho);
//   h_mu = Kokkos::create_mirror_view(mu);
//   h_kappa = Kokkos::create_mirror_view(kappa);
//   h_lambdaplus2mu = Kokkos::create_mirror_view(lambdaplus2mu);
//   h_ispec_type = Kokkos::create_mirror_view(ispec_type);
//   h_rho_inverse = Kokkos::create_mirror_view(rho_inverse);
//   h_lambdaplus2mu_inverse =
//   Kokkos::create_mirror_view(lambdaplus2mu_inverse);
// };

// specfem::compute::properties::acoustic_isotropic::acoustic_isotropic(
//     const int nspec, const int n_acoustic, const int ngllz, const int ngllx,
//     const specfem::mesh::materials &materials,
//     const specfem::kokkos::HostView1d<int> property_material_mapping)
//     : nspec(n_acoustic), ngllz(ngllz), ngllx(ngllx),
//       kappa("specfem::compute::properties::acoustic_isotropic::kappa",
//             n_acoustic, ngllz, ngllx),
//       rho_inverse(
//           "specfem::compute::properties::acoustic_isotropic::rho_inverse",
//           n_acoustic, ngllz, ngllx),
//       lambdaplus2mu_inverse("specfem::compute::properties::acoustic_isotropic::"
//                             "lambdaplus2mu_inverse",
//                             n_acoustic, ngllz, ngllx),
//       h_kappa(Kokkos::create_mirror_view(kappa)),
//       h_rho_inverse(Kokkos::create_mirror_view(rho_inverse)),
//       h_lambdaplus2mu_inverse(
//           Kokkos::create_mirror_view(lambdaplus2mu_inverse)) {

//   int acoustic_count = 0;
//   for (int ispec = 0; ispec < nspec; ++ispec) {
//     const auto material_specification =
//     materials.material_index_mapping(ispec); const int index =
//     material_specification.index; if ((material_specification.type ==
//          specfem::enums::element::type::acoustic) &&
//         (material_specification.property ==
//          specfem::enums::element::property::isotropic)) {
//       property_index_mapping(ispec) = acoustic_count;
//       for (int iz = 0; iz < ngllz; ++iz) {
//         for (int ix = 0; ix < ngllx; ++ix) {
//           auto property =
//               materials.acoustic_isotropic.material_properties(index)
//                   .get_properties();
//           h_rho_inverse(acoustic_count, iz, ix) = property.rho_inverse;
//           h_lambdaplus2mu_inverse(acoustic_count, iz, ix) =
//               property.lambdaplus2mu_inverse;
//           h_kappa(acoustic_count, iz, ix) = property.kappa;
//         }
//       }
//       acoustic_count++;
//     }
//   }

//   assert(acoustic_count == n_acoustic);

//   Kokkos::deep_copy(kappa, h_kappa);
//   Kokkos::deep_copy(rho_inverse, h_rho_inverse);
//   Kokkos::deep_copy(lambdaplus2mu_inverse, h_lambdaplus2mu_inverse);

//   return;
// }

// specfem::compute::properties::elastic_isotropic::elastic_isotropic(
//     const int nspec, const int n_elastic, const int ngllz, const int ngllx,
//     const specfem::mesh::materials &materials,
//     const specfem::kokkos::HostView1d<int> property_material_mapping)
//     : nspec(n_elastic), ngllz(ngllz), ngllx(ngllx),
//       rho("specfem::compute::properties::elastic_isotropic::rho", n_elastic,
//           ngllz, ngllx),
//       mu("specfem::compute::properties::elastic_isotropic::mu", n_elastic,
//          ngllz, ngllx),
//       lambdaplus2mu("specfem::compute::properties::elastic_isotropic::"
//                     "lambdaplus2mu",
//                     n_elastic, ngllz, ngllx),
//       h_rho(Kokkos::create_mirror_view(rho)),
//       h_mu(Kokkos::create_mirror_view(mu)),
//       h_lambdaplus2mu(Kokkos::create_mirror_view(lambdaplus2mu)) {

//   int elastic_count = 0;
//   for (int ispec = 0; ispec < nspec; ++ispec) {
//     const auto material_specification =
//     materials.material_index_mapping(ispec); const int index =
//     material_specification.index; if ((material_specification.type ==
//          specfem::enums::element::type::elastic) &&
//         (material_specification.property ==
//          specfem::enums::element::property::isotropic)) {
//       property_index_mapping(ispec) = elastic_count;
//       for (int iz = 0; iz < ngllz; ++iz) {
//         for (int ix = 0; ix < ngllx; ++ix) {
//           auto property =
//           materials.elastic_isotropic.material_properties(index)
//                               .get_properties();
//           h_rho(elastic_count, iz, ix) = property.rho;
//           h_mu(elastic_count, iz, ix) = property.mu;
//           h_lambdaplus2mu(elastic_count, iz, ix) = property.lambdaplus2mu;
//         }
//       }
//       elastic_count++;
//     }
//   }

//   assert(elastic_count == n_elastic);

//   Kokkos::deep_copy(rho, h_rho);
//   Kokkos::deep_copy(mu, h_mu);
//   Kokkos::deep_copy(lambdaplus2mu, h_lambdaplus2mu);

//   return;
// }

specfem::compute::properties::properties(
    const int nspec, const int ngllz, const int ngllx,
    const specfem::mesh::materials &materials)
    : nspec(nspec), ngllz(ngllz), ngllx(ngllx),
      property_index_mapping(
          "specfem::compute::properties::property_index_mapping", nspec),
      h_property_index_mapping(
          Kokkos::create_mirror_view(property_index_mapping)) {

  // compute total number of elastic and acoustic spectral elements
  int n_elastic;
  int n_acoustic;

  Kokkos::parallel_reduce(
      "specfem::mesh::mesh::print", this->nspec,
      KOKKOS_LAMBDA(const int ispec, int &n_elastic, int &n_acoustic) {
        if (materials.material_index_mapping(ispec).type ==
            specfem::enums::element::type::elastic) {
          n_elastic++;
        } else if (materials.material_index_mapping(ispec).type ==
                   specfem::enums::element::type::acoustic) {
          n_acoustic++;
        }
      },
      n_elastic, n_acoustic);

  assert(n_elastic + n_acoustic == nspec);

  acoustic_isotropic = specfem::compute::impl::properties::material_property<
      specfem::enums::element::type::acoustic,
      specfem::enums::element::property_tag::isotropic>(
      nspec, n_acoustic, ngllz, ngllx, materials, h_property_index_mapping);

  elastic_isotropic = specfem::compute::impl::properties::material_property<
      specfem::enums::element::type::elastic,
      specfem::enums::element::property_tag::isotropic>(
      nspec, n_elastic, ngllz, ngllx, materials, h_property_index_mapping);

  Kokkos::deep_copy(property_index_mapping, h_property_index_mapping);

  return;
}

// specfem::compute::properties::properties(
//     const specfem::kokkos::HostView1d<int> kmato,
//     const std::vector<std::shared_ptr<specfem::material::material> >
//     &materials, const int nspec, const int ngllx, const int ngllz) {
//   // Setup compute::properties properties
//   // UPDATEME::
//   //           acoustic materials
//   //           poroelastic materials
//   //           axisymmetric materials
//   //           anisotropic materials

//   std::vector<int> index_mapping(materials.size(), -1);

//   std::vector<specfem::material::elastic_material> elastic_materials;
//   int index = 0;
//   for (const int i = 0; i < materials.size(); ++i) {
//     if (materials[i]->get_type() == specfem::enums::element::type::elastic) {
//       elastic_materials.push_back(
//           *(std::dynamic_pointer_cast<specfem::material::elastic_material>(
//               materials[i])));
//       index_mapping[i] = index;
//       index++;
//     }
//   }

//   std::vector<specfem::material::acoustic_material> acoustic_materials;
//   index = 0;
//   for (const int i = 0; i < materials.size(); ++i) {
//     if (materials[i]->get_type() == specfem::enums::element::type::acoustic)
//     {
//       acoustic_materials.push_back(
//           *(std::dynamic_pointer_cast<specfem::material::acoustic_material>(
//               materials[i])));
//       index_mapping[i] = index;
//       index++;
//     }
//   }

//   assert(elastic_materials.size() + acoustic_materials.size() ==
//          materials.size());

//   int n_elastic = 0;
//   int n_acoustic = 0;

//   // compute total number of elastic and acoustic spectral elements
//   for (int ispec = 0; ispec < nspec; ++ispec) {
//     const int imat = kmato(ispec);
//     if (materials[imat]->get_type() ==
//     specfem::enums::element::type::elastic) {
//       ispec_mapping(ispec) = n_elastic;
//       n_elastic++;
//     } else if (materials[imat]->get_type() ==
//                specfem::enums::element::type::acoustic) {
//       ispec_mapping(ispec) = n_acoustic;
//       n_acoustic++;
//     }
//   }

//   assert(n_elastic + n_acoustic == nspec);

//   specfem::kokkos::HostView1d<int> kmato_elastic_isotropic(
//       "specfem::compute::properties::kmato_elastic_isotropic", n_elastic);

//   index = 0;
//   for (int ispec = 0; ispec < nspec; ++ispec) {
//     const int imat = kmato(ispec);
//     if (materials[imat]->get_type() ==
//     specfem::enums::element::type::elastic) {
//       kmato_elastic_isotropic(index) = index_mapping[imat];
//       index++;
//     }
//   }

//   specfem::compute::properties::elastic_isotropic elastic_isotropic(
//       n_elastic, ngllx, ngllz, elastic_materials, kmato_elastic_isotropic);

//   specfem::kokkos::HostView1d<int> kmato_acoustic_isotropic(
//       "specfem::compute::properties::kmato_acoustic_isotropic", n_acoustic);

//   index = 0;
//   for (int ispec = 0; ispec < nspec; ++ispec) {
//     const int imat = kmato(ispec);
//     if (materials[imat]->get_type() ==
//         specfem::enums::element::type::acoustic) {
//       kmato_acoustic_isotropic(index) = index_mapping[imat];
//       index++;
//     }
//   }

//   specfem::compute::properties::acoustic_isotropic acoustic_isotropic(
//       n_acoustic, ngllx, ngllz, acoustic_materials,
//       kmato_acoustic_isotropic);

//   *this = specfem::compute::properties(nspec, ngllz, ngllx);

//   Kokkos::parallel_for(
//       "specfem::compute::properties::properties",
//       specfem::kokkos::HostMDrange<3>({ 0, 0, 0 }, { nspec, ngllz, ngllx }),
//       [=](const int ispec, const int iz, const int ix) {
//         const int imat = kmato(ispec);
//         utilities::return_holder holder = materials[imat]->get_properties();
//         auto [rho, mu, kappa, qmu, qkappa, lambdaplus2mu] =
//             std::make_tuple(holder.rho, holder.mu, holder.kappa, holder.qmu,
//                             holder.qkappa, holder.lambdaplus2mu);
//         this->h_rho(ispec, iz, ix) = rho;
//         this->h_mu(ispec, iz, ix) = mu;
//         this->h_kappa(ispec, iz, ix) = kappa;

//         this->qmu(ispec, iz, ix) = qmu;
//         this->qkappa(ispec, iz, ix) = qkappa;

//         type_real vp = std::sqrt((kappa + mu) / rho);
//         type_real vs = std::sqrt(mu / rho);

//         this->rho_vp(ispec, iz, ix) = rho * vp;
//         this->rho_vs(ispec, iz, ix) = rho * vs;
//         this->h_rho_inverse(ispec, iz, ix) = 1.0 / rho;
//         this->h_lambdaplus2mu_inverse(ispec, iz, ix) = 1.0 / lambdaplus2mu;
//         this->h_lambdaplus2mu(ispec, iz, ix) = lambdaplus2mu;
//       });

//   Kokkos::parallel_for(
//       "setup_compute::properties_ispec", specfem::kokkos::HostRange(0,
//       nspec),
//       [=](const int ispec) {
//         const int imat = kmato(ispec);
//         this->h_ispec_type(ispec) = materials[imat]->get_ispec_type();
//       });

//   this->sync_views();
// }

// void specfem::compute::properties::sync_views() {
//   Kokkos::deep_copy(rho, h_rho);
//   Kokkos::deep_copy(mu, h_mu);
//   Kokkos::deep_copy(kappa, h_kappa);
//   Kokkos::deep_copy(lambdaplus2mu, h_lambdaplus2mu);
//   Kokkos::deep_copy(ispec_type, h_ispec_type);
//   Kokkos::deep_copy(rho_inverse, h_rho_inverse);
//   Kokkos::deep_copy(lambdaplus2mu_inverse, h_lambdaplus2mu_inverse);

//   return;
// }
