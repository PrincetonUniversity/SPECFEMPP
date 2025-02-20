#pragma once
#include "enumerations/dimension.hpp"
#include "mesh/mesh_base.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace mesh {

template <> struct materials<specfem::dimension::type::dim3> {

  constexpr static auto dimension =
      specfem::dimension::type::dim3; ///< Dimension
                                      ///< type

  int nspec; ///< Number of spectral elements
  int ngllx; ///< Number of GLL points in x
  int nglly; ///< Number of GLL points in y
  int ngllz; ///< Number of GLL points in z

  bool acoustic;    ///< Acoustic simulation
  bool elastic;     ///< Elastic simulation
  bool poroelastic; ///< Poroelastic simulation

  template <typename T> using View4D = Kokkos::View<T ****, Kokkos::HostSpace>;
  template <typename T> using View5D = Kokkos::View<T *****, Kokkos::HostSpace>;

  View4D<type_real> rho;   ///< Density rho
  View4D<type_real> kappa; ///< Bulk modulus kappa
  View4D<type_real> mu;    ///< Shear modulus mu

  View4D<type_real> rho_vp; ///< For Stacey ABC
  View4D<type_real> rho_vs; ///< For Stacey ABC

  // Poroelastic properties
  View5D<type_real> poro_rho;   ///< Poroelastic density rho
  View5D<type_real> poro_kappa; ///< Poroelastic bulk modulus kappa
  View5D<type_real> poro_perm;  ///< Poroelastic permeability perm

  // Poroelastic properties
  View4D<type_real> poro_eta;      ///< Poroelastic storage coefficient eta
  View4D<type_real> poro_tort;     ///< Poroelastic tortuosity tort
  View4D<type_real> poro_phi;      ///< Poroelastic porosity phi
  View4D<type_real> poro_rho_vpI;  ///< Poroelastic rho_vpI
  View4D<type_real> poro_rho_vpII; ///< Poroelastic rho_vspII
  View4D<type_real> poro_rho_vsI;  ///< Poroelastic rho_vsI

  // Anisotropic properties
  bool anisotropic = false;
  View4D<type_real> c11;
  View4D<type_real> c12;
  View4D<type_real> c13;
  View4D<type_real> c14;
  View4D<type_real> c15;
  View4D<type_real> c16;
  View4D<type_real> c22;
  View4D<type_real> c23;
  View4D<type_real> c24;
  View4D<type_real> c25;
  View4D<type_real> c26;
  View4D<type_real> c33;
  View4D<type_real> c34;
  View4D<type_real> c35;
  View4D<type_real> c36;
  View4D<type_real> c44;
  View4D<type_real> c45;
  View4D<type_real> c46;
  View4D<type_real> c55;
  View4D<type_real> c56;
  View4D<type_real> c66;

  /**
   * @name Constructors
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  materials() = default;
  /**
   * @brief Constructor used to allocate views
   *
   * @param nspec Number of nodes
   */

  materials(const int nspec, const int ngllx, const int nglly, const int ngllz,
            const bool acoustic, const bool elastic, const bool poroelastic,
            const bool anisotropic)
      : nspec(nspec), ngllx(ngllx), nglly(nglly), ngllz(ngllz),
        acoustic(acoustic), elastic(elastic), poroelastic(poroelastic),
        anisotropic(anisotropic),
        rho("specfem::mesh::materials::rho", nspec, ngllx, nglly, ngllz),
        kappa("specfem::mesh::materials::kappa", nspec, ngllx, nglly, ngllz),
        mu("specfem::mesh::materials::mu", nspec, ngllx, nglly, ngllz) {
    if (elastic) {
      rho_vp = View4D<type_real>("specfem::mesh::materials::rho_vp", nspec,
                                 ngllx, nglly, ngllz);
      rho_vs = View4D<type_real>("specfem::mesh::materials::rho_vs", nspec,
                                 ngllx, nglly, ngllz);

      if (anisotropic) {
        c11 = View4D<type_real>("specfem::mesh::materials::c11", nspec, ngllx,
                                nglly, ngllz);
        c12 = View4D<type_real>("specfem::mesh::materials::c12", nspec, ngllx,
                                nglly, ngllz);
        c13 = View4D<type_real>("specfem::mesh::materials::c13", nspec, ngllx,
                                nglly, ngllz);
        c14 = View4D<type_real>("specfem::mesh::materials::c14", nspec, ngllx,
                                nglly, ngllz);
        c15 = View4D<type_real>("specfem::mesh::materials::c15", nspec, ngllx,
                                nglly, ngllz);
        c16 = View4D<type_real>("specfem::mesh::materials::c16", nspec, ngllx,
                                nglly, ngllz);
        c22 = View4D<type_real>("specfem::mesh::materials::c22", nspec, ngllx,
                                nglly, ngllz);
        c23 = View4D<type_real>("specfem::mesh::materials::c23", nspec, ngllx,
                                nglly, ngllz);
        c24 = View4D<type_real>("specfem::mesh::materials::c24", nspec, ngllx,
                                nglly, ngllz);
        c25 = View4D<type_real>("specfem::mesh::materials::c25", nspec, ngllx,
                                nglly, ngllz);
        c26 = View4D<type_real>("specfem::mesh::materials::c26", nspec, ngllx,
                                nglly, ngllz);
        c33 = View4D<type_real>("specfem::mesh::materials::c33", nspec, ngllx,
                                nglly, ngllz);
        c34 = View4D<type_real>("specfem::mesh::materials::c34", nspec, ngllx,
                                nglly, ngllz);
        c35 = View4D<type_real>("specfem::mesh::materials::c35", nspec, ngllx,
                                nglly, ngllz);
        c36 = View4D<type_real>("specfem::mesh::materials::c36", nspec, ngllx,
                                nglly, ngllz);
        c44 = View4D<type_real>("specfem::mesh::materials::c44", nspec, ngllx,
                                nglly, ngllz);
        c45 = View4D<type_real>("specfem::mesh::materials::c45", nspec, ngllx,
                                nglly, ngllz);
        c46 = View4D<type_real>("specfem::mesh::materials::c46", nspec, ngllx,
                                nglly, ngllz);
        c55 = View4D<type_real>("specfem::mesh::materials::c55", nspec, ngllx,
                                nglly, ngllz);
        c56 = View4D<type_real>("specfem::mesh::materials::c56", nspec, ngllx,
                                nglly, ngllz);
        c66 = View4D<type_real>("specfem::mesh::materials::c66", nspec, ngllx,
                                nglly, ngllz);
      }
    }
    if (poroelastic) {
      // Hardcoded array sizes
      poro_rho = View5D<type_real>("specfem::mesh::materials::poro_rho", nspec,
                                   2, ngllx, nglly, ngllz);
      poro_kappa = View5D<type_real>("specfem::mesh::materials::poro_kappa",
                                     nspec, 3, ngllx, nglly, ngllz);
      poro_perm = View5D<type_real>("specfem::mesh::materials::poro_perm",
                                    nspec, 6, ngllx, nglly, ngllz);
      poro_eta = View4D<type_real>("specfem::mesh::materials::poro_eta", nspec,
                                   ngllx, nglly, ngllz);
      poro_tort = View4D<type_real>("specfem::mesh::materials::poro_tort",
                                    nspec, ngllx, nglly, ngllz);
      poro_phi = View4D<type_real>("specfem::mesh::materials::poro_phi", nspec,
                                   ngllx, nglly, ngllz);
      poro_rho_vpI = View4D<type_real>("specfem::mesh::materials::poro_rho_vpI",
                                       nspec, ngllx, nglly, ngllz);
      poro_rho_vpII =
          View4D<type_real>("specfem::mesh::materials::poro_rho_vpII", nspec,
                            ngllx, nglly, ngllz);
      poro_rho_vsI = View4D<type_real>("specfem::mesh::materials::poro_rho_vsI",
                                       nspec, ngllx, nglly, ngllz);
    }
  };

  void print();
};
} // namespace mesh
} // namespace specfem
