#pragma once
#include "enumerations/dimension.hpp"
#include "mesh/mesh_base.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace mesh {

/**
 * @brief Struct to store materials for a 3D mesh
 *
 */
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
  bool anisotropic = false; ///< Anisotropic simulation
  View4D<type_real> c11;    ///< Anisotropic c11
  View4D<type_real> c12;    ///< Anisotropic c12
  View4D<type_real> c13;    ///< Anisotropic c13
  View4D<type_real> c14;    ///< Anisotropic c14
  View4D<type_real> c15;    ///< Anisotropic c15
  View4D<type_real> c16;    ///< Anisotropic c16
  View4D<type_real> c22;    ///< Anisotropic c22
  View4D<type_real> c23;    ///< Anisotropic c23
  View4D<type_real> c24;    ///< Anisotropic c24
  View4D<type_real> c25;    ///< Anisotropic c25
  View4D<type_real> c26;    ///< Anisotropic c26
  View4D<type_real> c33;    ///< Anisotropic c33
  View4D<type_real> c34;    ///< Anisotropic c34
  View4D<type_real> c35;    ///< Anisotropic c35
  View4D<type_real> c36;    ///< Anisotropic c36
  View4D<type_real> c44;    ///< Anisotropic c44
  View4D<type_real> c45;    ///< Anisotropic c45
  View4D<type_real> c46;    ///< Anisotropic c46
  View4D<type_real> c55;    ///< Anisotropic c55
  View4D<type_real> c56;    ///< Anisotropic c56
  View4D<type_real> c66;    ///< Anisotropic c66

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
   * @brief Construct a new materials object
   *
   * @param nspec  Number of spectral elements
   * @param ngllx  Number of GLL points in x
   * @param nglly  Number of GLL points in y
   * @param ngllz  Number of GLL points in z
   * @param acoustic  whether the simulation is acoustic
   * @param elastic  whether the simulation is elastic
   * @param poroelastic  whether the simulation is poroelastic
   * @param anisotropic  whether the simulation is anisotropic
   */
  materials(const int nspec, const int ngllz, const int nglly, const int ngllx,
            const bool acoustic, const bool elastic, const bool poroelastic,
            const bool anisotropic)
      : nspec(nspec), ngllx(ngllx), nglly(nglly), ngllz(ngllz),
        acoustic(acoustic), elastic(elastic), poroelastic(poroelastic),
        anisotropic(anisotropic),
        rho("specfem::mesh::materials::rho", nspec, ngllz, nglly, ngllx),
        kappa("specfem::mesh::materials::kappa", nspec, ngllz, nglly, ngllx),
        mu("specfem::mesh::materials::mu", nspec, ngllz, nglly, ngllx) {
    if (elastic) {
      rho_vp = View4D<type_real>("specfem::mesh::materials::rho_vp", nspec,
                                 ngllz, nglly, ngllx);
      rho_vs = View4D<type_real>("specfem::mesh::materials::rho_vs", nspec,
                                 ngllz, nglly, ngllx);

      if (anisotropic) {
        c11 = View4D<type_real>("specfem::mesh::materials::c11", nspec, ngllz,
                                nglly, ngllx);
        c12 = View4D<type_real>("specfem::mesh::materials::c12", nspec, ngllz,
                                nglly, ngllx);
        c13 = View4D<type_real>("specfem::mesh::materials::c13", nspec, ngllz,
                                nglly, ngllx);
        c14 = View4D<type_real>("specfem::mesh::materials::c14", nspec, ngllz,
                                nglly, ngllx);
        c15 = View4D<type_real>("specfem::mesh::materials::c15", nspec, ngllz,
                                nglly, ngllx);
        c16 = View4D<type_real>("specfem::mesh::materials::c16", nspec, ngllz,
                                nglly, ngllx);
        c22 = View4D<type_real>("specfem::mesh::materials::c22", nspec, ngllz,
                                nglly, ngllx);
        c23 = View4D<type_real>("specfem::mesh::materials::c23", nspec, ngllz,
                                nglly, ngllx);
        c24 = View4D<type_real>("specfem::mesh::materials::c24", nspec, ngllz,
                                nglly, ngllx);
        c25 = View4D<type_real>("specfem::mesh::materials::c25", nspec, ngllz,
                                nglly, ngllx);
        c26 = View4D<type_real>("specfem::mesh::materials::c26", nspec, ngllz,
                                nglly, ngllx);
        c33 = View4D<type_real>("specfem::mesh::materials::c33", nspec, ngllz,
                                nglly, ngllx);
        c34 = View4D<type_real>("specfem::mesh::materials::c34", nspec, ngllz,
                                nglly, ngllx);
        c35 = View4D<type_real>("specfem::mesh::materials::c35", nspec, ngllz,
                                nglly, ngllx);
        c36 = View4D<type_real>("specfem::mesh::materials::c36", nspec, ngllz,
                                nglly, ngllx);
        c44 = View4D<type_real>("specfem::mesh::materials::c44", nspec, ngllz,
                                nglly, ngllx);
        c45 = View4D<type_real>("specfem::mesh::materials::c45", nspec, ngllz,
                                nglly, ngllx);
        c46 = View4D<type_real>("specfem::mesh::materials::c46", nspec, ngllz,
                                nglly, ngllx);
        c55 = View4D<type_real>("specfem::mesh::materials::c55", nspec, ngllz,
                                nglly, ngllx);
        c56 = View4D<type_real>("specfem::mesh::materials::c56", nspec, ngllz,
                                nglly, ngllx);
        c66 = View4D<type_real>("specfem::mesh::materials::c66", nspec, ngllz,
                                nglly, ngllx);
      }
    }
    if (poroelastic) {
      // Hardcoded array sizes
      poro_rho = View5D<type_real>("specfem::mesh::materials::poro_rho", nspec,
                                   2, ngllz, nglly, ngllx);
      poro_kappa = View5D<type_real>("specfem::mesh::materials::poro_kappa",
                                     nspec, 3, ngllz, nglly, ngllx);
      poro_perm = View5D<type_real>("specfem::mesh::materials::poro_perm",
                                    nspec, 6, ngllz, nglly, ngllx);
      poro_eta = View4D<type_real>("specfem::mesh::materials::poro_eta", nspec,
                                   ngllz, nglly, ngllx);
      poro_tort = View4D<type_real>("specfem::mesh::materials::poro_tort",
                                    nspec, ngllz, nglly, ngllx);
      poro_phi = View4D<type_real>("specfem::mesh::materials::poro_phi", nspec,
                                   ngllz, nglly, ngllx);
      poro_rho_vpI = View4D<type_real>("specfem::mesh::materials::poro_rho_vpI",
                                       nspec, ngllz, nglly, ngllx);
      poro_rho_vpII =
          View4D<type_real>("specfem::mesh::materials::poro_rho_vpII", nspec,
                            ngllz, nglly, ngllx);
      poro_rho_vsI = View4D<type_real>("specfem::mesh::materials::poro_rho_vsI",
                                       nspec, ngllz, nglly, ngllx);
    }
  };

  ///@}

  /**
   * @brief Print basic information about the materials
   *
   */
  std::string print();
};
} // namespace mesh
} // namespace specfem
