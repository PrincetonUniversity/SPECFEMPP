#ifndef _CONSTANTS_HPP
#define _CONSTANTS_HPP

#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <string>

const type_real pi = 2 * Kokkos::acos(0.0);

namespace specfem {

namespace wave {
enum type {
  p_sv, ///< P-SV wave
  sh    ///< SH wave
};
} // namespace wave

} // namespace specfem

namespace specfem {
namespace build_configuration {
namespace chunk {
constexpr static int chunk_size = 32;
constexpr static int num_chunks = 1;
constexpr static int num_threads = 160;
constexpr static int vector_lanes = 1;
}
}


namespace constants::empirical {
  /**
   * @brief Source decay rate to mimic a triangle source time function
   *
   * We mimic a triangle of half duration equal to half_duration_triangle using a
   * Gaussian having a very close shape, as explained in Figure 4.2 of the manual.
   * This source decay rate to mimic an equivalent triangle was found by trial and
   * error.
   *
   * @note From globalcmt.org: The source duration is generally estimated using an
   * empirically determined relationship such that the duration increases as the
   * cube root of the scalar moment. Specifically, we currently use a relationship
   * where the half duration for an event with moment 10**24 is 1.05 seconds, and
   * for an event with moment 10**27 is 10.5 seconds.
   *
   * @see https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/allorder.ndk_explained
   */
  const type_real SOURCE_DECAY_MIMIC_TRIANGLE = 1.62800;
}

}

#endif
