#include "specfem/data_access.hpp"

namespace specfem::assembly {

struct RungeKuttaIntegrationFactors {

  RungeKuttaIntegrationFactors() = default;

  // runge-kutta coefficients see e.g.: Savage et al. (BSSA, 2010): eq. (11)
  RungeKuttaIntegrationFactors(
      const type_real deltat,
      const Kokkos::View<type_real *,
                         Kokkos::DefaultExecutionSpace::memory_space>
          tau_sigma) {
    // Compute the integration factors from the attenuation factors and time
    // step size
    the specific form of the attenuation factors.for (int i = 0;
                                                      i < tauinv.extent(0);
                                                      ++i) {

      const auto tauinv_i = static_cast<type_real>(1.0) / tau_sigma(i);

      alpha(i) = 1.0 + deltat * tauinv_i +
                 std::pow(deltat * tauinv_i, 2) / 2.0 +
                 std::pow(deltat * tauinv_i, 3) / 6.0 +
                 std::pow(deltat * tauinv_i, 4) / 24.0;
      beta(i) = deltat / 2.0 + std::pow(deltat, 2) * tauinv_i / 3.0 +
                std::pow(deltat, 3) * std::pow(tauinv_i, 2) / 8.0 +
                std::pow(deltat, 4) * std::pow(tauinv_i, 3) / 24.0;
      gamma(i) = deltat / 2.0 + std::pow(deltat, 2) * tauinv_i / 6.0 +
                 std::pow(deltat, 3) * std::pow(tauinv_i, 2) / 24.0;
    }
  }
};

/**
 * This struct is used to hold per element and gll point attenuation factors and
 * the memory variable arrays R_mu and R_kappa for the constant isotropic
 * attenuation case in elastic media. The struct is used to pass these factors
 * and variables to the compute kernels during the stiffness interaction
 * computation.
 *
 * R_mu is deviatoric and needs to hold Rxx, Ryy, Rxy, Rxz, Ryz for 3D and Rxx,
 *    Rzz, Rxz for 2D Ryy etc. are all store with nelement, ngll^ndim, N_SLS
 *    dimensions
 *
 * R_kappa is volumetric and needs to hold a single value and is
 *    stored the same way.
 *
 * A_mu is the factor for the shear attenuation update and is stored as a vector
 * of length N_SLS per GLL point
 *
 * A_kappa is the factor for the bulk attenuation update and is stored as a
 * vector of length N_SLS per GLL point
 */
template struct Attenuation {

  /**
   * @brief Base container type providing data access infrastructure
   *
   * @see specfem::data_access::Container
   */
  using base_type = specfem::data_access::Container<
      specfem::data_access::ContainerType::domain,
      specfem::data_access::DataClassType::jacobian_matrix,
      specfem::element::dimension_tag::dim3>;

  /**
   * @brief Kokkos view type for storing Jacobian matrix components
   */
  using memory_variable_view_type = typename base_type::vector_type<
      type_real, Kokkos::DefaultExecutionSpace::memory_space>;

  using attenuation_factor_view_type = typename base_type::vector_type<
      type_real, Kokkos::DefaultExecutionSpace::memory_space>;
  ///@}

  memory_variable_view_type R_xx;
  memory_variable_view_type R_yy;
  memory_variable_view_type R_xy;
  memory_variable_view_type R_xz;
  memory_variable_view_type R_yz;

  memory_variable_view_type R_trace;

  attenuation_factor_view_type A_mu;
  attenuation_factor_view_type A_kappa;

  RungeKuttaIntegrationFactors integration_factors;

  Attenuation() = default;

  Attenuation(assembly::properties properties,
              assembly::attenuation attenuation, const int nspec,
              const int ngllx, const int nglly, const int ngllz) {
    // Allocate memory for the memory variables and attenuation factors
    R_xx = memory_variable_view_type("R_xx", nspec, ngllz, nglly, ngllx, N_SLS);
    R_yy = memory_variable_view_type("R_yy", nspec, ngllz, nglly, ngllx, N_SLS);
    R_xy = memory_variable_view_type("R_xy", nspec, ngllz, nglly, ngllx, N_SLS);
    R_xz = memory_variable_view_type("R_xz", nspec, ngllz, nglly, ngllx, N_SLS);
    R_yz = memory_variable_view_type("R_yz", nspec, ngllz, nglly, ngllx, N_SLS);

    R_trace =
        memory_variable_view_type("R_trace", nspec, ngllz, nglly, ngllx, N_SLS);

    A_mu =
        attenuation_factor_view_type("A_mu", nspec, ngllz, nglly, ngllx, N_SLS);
    A_kappa = attenuation_factor_view_type("A_kappa", nspec, ngllz, nglly,
                                           ngllx, N_SLS);

    // Initialize the memory variables to zero
    Kokkos::deep_copy(R_xx, static_cast<type_real>(0.0));
    Kokkos::deep_copy(R_yy, static_cast<type_real>(0.0));
    Kokkos::deep_copy(R_xy, static_cast<type_real>(0.0));
    Kokkos::deep_copy(R_xz, static_cast<type_real>(0.0));
    Kokkos::deep_copy(R_yz, static_cast<type_real>(0.0));

    Kokkos::deep_copy(R_trace, static_cast<type_real>(0.0));

    // Compute the attenuation factors A_mu and A_kappa from the properties and
    // attenuation parameters
    compute_attenuation_factors(properties, quality_factors);
  }
};

} // namespace specfem::assembly
