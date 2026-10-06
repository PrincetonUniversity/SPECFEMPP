#pragma once

#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <stdexcept>

namespace specfem {
namespace utilities {
namespace fft {

/**
 * @brief Precomputed tables required by the radix-2 Cooley–Tukey FFT.
 *
 * Build with @c make_fft_tables on the host; the resulting Views can be
 * passed (by their raw @c .data() pointers) into the device-callable
 * @c transform / @c inverse functions.
 */
struct FftTables {
  Kokkos::View<int *> bit_reversal; ///< Bit-reversal recurrence table, size
                                    ///< power+1 (index 1..power used).
  Kokkos::View<Kokkos::complex<double> *> twiddles; ///< Flattened twiddle
                                                    ///< factors, size 2^power
                                                    ///< - 1.
  int power; ///< log2 of transform length.
};

/**
 * @brief Build FFT twiddle-factor and bit-reversal tables on the host, then
 * copy them to device Views.
 *
 * The twiddle factor for level @p l (1-based) and block @p iblock (1-based)
 * is stored at index @c ((1<<(l-1)) - 1) + (iblock-1).
 *
 * @param power Transform length is @c lx = 2^power.
 * @param zign  Sign convention: use @c -1.0 for the forward transform
 *              (@c e^{-iωt}) and @c +1.0 for the inverse.
 * @return Fully populated @c FftTables with device-resident Views.
 * @throws std::runtime_error if @p power > 30.
 */
inline FftTables make_fft_tables(int power, double zign) {
  if (power > 30) {
    throw std::runtime_error("specfem::utilities::fft::make_fft_tables: power "
                             "> 30 is not supported");
  }

  constexpr double two_pi = 2.0 * 3.141592653589793;

  const int lx = 1 << power; // transform length

  // ---- mpow table (1-based, indices 1..power) ----
  // mpow[i] = 2^(power - i)  for i = 1..power; mpow[0] unused.
  Kokkos::View<int *, Kokkos::HostSpace> h_bit_reversal("h_bit_reversal",
                                                        power + 1);
  for (int i = 1; i <= power; ++i) {
    h_bit_reversal(i) = 1 << (power - i);
  }
  h_bit_reversal(0) = 0; // unused sentinel

  // ---- twiddle factors ----
  // Total entries: sum_{l=1}^{power} 2^(l-1) = 2^power - 1.
  const int n_twiddles = lx - 1;
  Kokkos::View<Kokkos::complex<double> *, Kokkos::HostSpace> h_twiddles(
      "h_twiddles", n_twiddles);

  for (int l = 1; l <= power; ++l) {
    const int nblock = 1 << (l - 1); // 2^(l-1)
    const int offset = nblock - 1;   // (1<<(l-1)) - 1

    int k = 0;
    for (int iblock = 1; iblock <= nblock; ++iblock) {
      const double v =
          zign * two_pi * static_cast<double>(k) / static_cast<double>(lx);
      h_twiddles(offset + (iblock - 1)) =
          Kokkos::complex<double>(Kokkos::cos(v), -Kokkos::sin(v));

      // Advance k using the same bit-reversal recurrence as the reference.
      int ii = 2;
      for (; ii <= power; ++ii) {
        if (k < h_bit_reversal(ii))
          break;
        k -= h_bit_reversal(ii);
      }
      // If we exhausted the loop (ii > power), use ii = power.
      if (ii > power)
        ii = power;
      k += h_bit_reversal(ii);
    }
  }

  // ---- deep-copy to device ----
  FftTables tables;
  tables.power = power;
  tables.bit_reversal = Kokkos::View<int *>("bit_reversal", power + 1);
  tables.twiddles =
      Kokkos::View<Kokkos::complex<double> *>("twiddles", n_twiddles);

  Kokkos::deep_copy(tables.bit_reversal, h_bit_reversal);
  Kokkos::deep_copy(tables.twiddles, h_twiddles);

  return tables;
}

/**
 * @brief In-place radix-2 Cooley–Tukey FFT (Fortran @c e^{-iωt} convention).
 *
 * Operates on a raw pointer to @p lx = 2^power complex samples.  Safe to call
 * from inside a Kokkos kernel when @p data points to per-thread scratch.
 *
 * Twiddle layout: level @p l, block @p iblock → index
 * @c ((1<<(l-1)) - 1) + (iblock-1).
 *
 * @param power       log2 of transform length.
 * @param data        Pointer to @c 2^power in-place samples.
 * @param zign        Sign used when tables were built (+1 or -1).
 * @param dt          Time-step for scaling: forward multiplies by @c dt,
 *                    inverse multiplies by @c 1/(lx*dt).
 * @param bit_reversal Bit-reversal table from @c
 * FftTables::bit_reversal.data().
 * @param twiddles    Twiddle table from @c FftTables::twiddles.data().
 */
KOKKOS_INLINE_FUNCTION void transform(int power, Kokkos::complex<double> *data,
                                      double zign, double dt,
                                      const int *bit_reversal,
                                      const Kokkos::complex<double> *twiddles) {
  const int lx = 1 << power;

  // ---- butterfly passes ----
  for (int l = 1; l <= power; ++l) {
    const int nblock = 1 << (l - 1);
    const int lblock = lx / nblock;
    const int lbhalf = lblock / 2;
    const int offset = nblock - 1; // twiddle index base for this level

    for (int iblock = 1; iblock <= nblock; ++iblock) {
      const Kokkos::complex<double> wk = twiddles[offset + (iblock - 1)];
      const int istart = lblock * (iblock - 1);

      for (int i = 1; i <= lbhalf; ++i) {
        const int j = istart + i;  // 1-based → index j-1
        const int jh = j + lbhalf; // 1-based → index jh-1

        const Kokkos::complex<double> q = data[jh - 1] * wk;
        data[jh - 1] = data[j - 1] - q;
        data[j - 1] = data[j - 1] + q;
      }
    }
  }

  // ---- bit-reversal permutation ----
  {
    int k = 0;
    for (int j = 1; j <= lx; ++j) {
      if (k < j - 1) { // swap 0-based indices k and j-1
        const Kokkos::complex<double> tmp = data[j - 1];
        data[j - 1] = data[k];
        data[k] = tmp;
      }
      // Advance k with the recurrence.
      int ii = 1;
      for (; ii <= power; ++ii) {
        if (k < bit_reversal[ii])
          break;
        k -= bit_reversal[ii];
      }
      if (ii > power)
        ii = power;
      k += bit_reversal[ii];
    }
  }

  // ---- dt scaling ----
  if (zign > 0.0) {
    for (int i = 0; i < lx; ++i) {
      data[i] *= dt;
    }
  } else {
    const double inv_scale = 1.0 / (static_cast<double>(lx) * dt);
    for (int i = 0; i < lx; ++i) {
      data[i] *= inv_scale;
    }
  }
}

/**
 * @brief Apply Hermitian symmetry to the second half of a spectrum
 * (the @c rspec routine from the Fortran reference).
 *
 * Sets @c s[nhalf] = 0 (Nyquist imaginary to zero), forces @c s[0] to be
 * real, then mirrors the first half into the second half as
 * @c s[nhalf+i] = conj(s[nhalf+2-i-1]) for @p i = 1..nhalf.
 *
 * @param s      Pointer to the complex spectrum array of length @c 2*nhalf.
 * @param nhalf  Half-length of the transform (@c lx/2 = 2^(power-1)).
 */
KOKKOS_INLINE_FUNCTION void restructure_spectrum(Kokkos::complex<double> *s,
                                                 int nhalf) {
  const int n1 = nhalf + 1; // 1-based n1

  // s[n1-1] (0-based: nhalf) = 0
  s[n1 - 1] = Kokkos::complex<double>(0.0, 0.0);

  // s[0] is forced real
  s[0] = Kokkos::complex<double>(s[0].real(), 0.0);

  // Reference (1-based): s[np2 + i - 1] = conj(s[np2 + 2 - i - 1]) for
  // i=1..np2, with np2 = nhalf. In 0-based indices:
  //   lhs = nhalf + i - 1,  rhs = (nhalf + 2 - i) - 1 = nhalf + 1 - i
  // e.g. i=2 -> s[nhalf+1] = conj(s[nhalf-1]) (Hermitian mirror about nhalf).
  for (int i = 1; i <= nhalf; ++i) {
    s[nhalf + i - 1] = Kokkos::conj(s[nhalf + 1 - i]);
  }
}

/**
 * @brief Compute the inverse FFT of a complex spectrum and store the real part.
 *
 * Applies @c restructure_spectrum, then @c transform, then extracts the real
 * parts into @p out.
 *
 * @param power       log2 of transform length @c lx = 2^power.
 * @param s           In/out complex spectrum of length @c lx (modified in
 * place).
 * @param zign        Sign convention (should match the sign used during the
 *                    forward transform; pass @c -1.0 for the @c e^{-iωt}
 * convention).
 * @param dt          Time-step used in the corresponding forward transform.
 * @param out         Output buffer of length @c lx; receives the real part of
 * each transformed sample.
 * @param bit_reversal Bit-reversal table from @c
 * FftTables::bit_reversal.data().
 * @param twiddles    Twiddle table from @c FftTables::twiddles.data().
 */
KOKKOS_INLINE_FUNCTION void inverse(int power, Kokkos::complex<double> *s,
                                    double zign, double dt, double *out,
                                    const int *bit_reversal,
                                    const Kokkos::complex<double> *twiddles) {
  const int lx = 1 << power;
  const int nhalf = lx / 2;

  restructure_spectrum(s, nhalf);
  transform(power, s, zign, dt, bit_reversal, twiddles);

  for (int i = 0; i < lx; ++i) {
    out[i] = s[i].real();
  }
}

} // namespace fft
} // namespace utilities
} // namespace specfem
