#pragma once

#include "constants.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>

namespace specfem {
namespace attenuation {

/**
 * @brief Result containing real (\f$A\f$) and imaginary (\f$B\f$) moduli from
 * Maxwell solid computation
 *
 * For a standard linear solid (SLS), the complex modulus can be written as:
 * \f[
 *   M^*(\omega) = M_R \left( 1 + \sum_i \frac{(\tau_{\epsilon_i} -
 * \tau_{\sigma_i}) \omega^2 \tau_{\sigma_i}}{1 + \omega^2 \tau_{\sigma_i}^2}
 * \right)
 *              + i M_R \sum_i \frac{(\tau_{\epsilon_i} - \tau_{\sigma_i})
 * \omega}{1 + \omega^2 \tau_{\sigma_i}^2}
 * \f]
 *
 * \f$real\f$ represents the real part (storage modulus ratio),
 * \f$imag\f$ represents the imaginary part (loss modulus ratio).
 * The quality factor \f$Q = real / imag = 1 / \tan\delta\f$.
 *
 * @tparam NF Number of frequencies
 *
 */
template <int NF> struct MaxwellFactors {
  Kokkos::View<type_real[NF], Kokkos::LayoutRight, Kokkos::HostSpace> real;
  Kokkos::View<type_real[NF], Kokkos::LayoutRight, Kokkos::HostSpace> imag;
};

/**
 * @brief Compute Maxwell solid moduli for a series of standard linear solids
 *
 * This function computes the real (\f$real\f$) and imaginary (\f$imag\f$) parts
 * of the viscoelastic modulus for a generalized Maxwell solid (also known as
 * generalized Zener model) consisting of \f$N_\text{SLS}\f$ standard linear
 * solids in parallel.
 *
 * The formulas match Jeroen's Attenuation notes (43)-(44), with 1/L
 * normalization:
 *
 * For angular frequency \f$\omega = 2 \pi f\f$:
 *
 * \f[real(\omega) = \frac{1}{L} \sum_{i=1}^{L} \frac{1 + \omega^2
 * \tau_{\epsilon_i} \tau_{\sigma_i}}{1 + \omega^2 \tau_{\sigma_i}^2}\f]
 *
 * \f[imag(\omega) = \frac{1}{L} \sum_{i=1}^{L} \frac{\omega (\tau_{\epsilon_i}
 * -
 * \tau_{\sigma_i})}{1 + \omega^2 \tau_{\sigma_i}^2}\f]
 *
 * where \f$L = N_\text{SLS}\f$. At low frequency, \f$A\f$ approaches \f$1\f$.
 * The quality factor \f$Q = A/B\f$ is independent of the normalization.
 * \f$B\f$ is independent of the normalization.
 *
 * @note the real part is often referred to as \f$ M_1 \f$ and the imaginary
 * part as \f$ M_2 \f$ in the literature, but we use \f$real\f$ and \f$imag\f$
 * here to avoid confusion with the actual \f$ M_1 \f$ and \f$ M_2 \f$ moduli
 * of the material, which would require multiplying by the relaxed modulus
 * \f$M_R\f$.
 *
 * The quality factor \f$Q = A / B\f$, so \f$\tan(\delta) = B / A = 1 / Q\f$
 *
 * @tparam NF Number of frequencies to evaluate
 * @tparam N_SLS Number of standard linear solids
 *
 * @param f View containing \f$N_F\f$ frequencies \f$f\f$ (in Hz, not
 * \f$\log_{10}\f$)
 * @param tau_s View containing \f$N_\text{SLS}\f$ stress relaxation times
 * \f$\tau_\sigma\f$
 * @param tau_eps View containing \f$N_\text{SLS}\f$ strain relaxation times
 * \f$\tau_\epsilon\f$
 *
 * @return MaxwellFactors containing \f$real\f$ (real) and \f$imag\f$
 * (imaginary) moduli
 *
 * @code
 * // Example: compute moduli for 3 SLS over a frequency range
 * constexpr int NF = 100;
 * constexpr int N_SLS = 3;
 * Kokkos::View<type_real[NF], ...> f("f");
 * Kokkos::View<type_real[N_SLS], ...> tau_s("tau_s");
 * Kokkos::View<type_real[N_SLS], ...> tau_eps("tau_eps"); // ... fill in values
 * ... auto moduli = maxwell<NF, N_SLS>(f, tau_s, tau_eps); // Q at
 * frequency i is approximately moduli.real(i) / moduli.imag(i)
 * @endcode
 */
template <int NF, int N_SLS>
MaxwellFactors<NF>
maxwell(Kokkos::View<type_real[NF], Kokkos::LayoutRight, Kokkos::HostSpace> f,
        Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
            tau_s,
        Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
            tau_eps) {

  MaxwellFactors<NF> result;
  result.real =
      Kokkos::View<type_real[NF], Kokkos::LayoutRight, Kokkos::HostSpace>(
          "real");
  result.imag =
      Kokkos::View<type_real[NF], Kokkos::LayoutRight, Kokkos::HostSpace>(
          "imag");

  for (int i = 0; i < NF; ++i) {
    // Angular frequency: w = 2 * pi * f
    const type_real w = 2.0 * pi * f(i);
    const type_real w2 = w * w;

    type_real A_sum = 0.0;
    type_real B_sum = 0.0;

    for (int j = 0; j < N_SLS; ++j) {
      const type_real tau_s_j = tau_s(j);
      const type_real tau_eps_j = tau_eps(j);
      const type_real tau_s_j_sq = tau_s_j * tau_s_j;
      const type_real denom = 1.0 + w2 * tau_s_j_sq;

      // Real part: (1 + w^2 * tau_eps * tau_s) / denom
      A_sum += (1.0 + w2 * tau_eps_j * tau_s_j) / denom;

      // Imaginary part: w * (tau_eps - tau_s) / denom
      B_sum += w * (tau_eps_j - tau_s_j) / denom;
    }

    // Apply 1/L normalization per Jeroen Tromp's Attenuation notes
    result.real(i) = A_sum / N_SLS;
    result.imag(i) = B_sum / N_SLS;
  }

  return result;
}

} // namespace attenuation
} // namespace specfem
