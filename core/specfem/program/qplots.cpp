#include "specfem/attenuation.hpp"
#include "specfem/program.hpp"
#include "specfem/utilities.hpp"
#include "specfem_setup.hpp"

bool specfem::program::qplots(const type_real Q, const type_real minfreq,
                              const type_real maxfreq,
                              const type_real min_plot_freq,
                              const type_real max_plot_freq,
                              std::string output_dir) {

  if (Q <= 0) {
    specfem::Logger::error("Q must be positive for qplots");
    return false;
  }
  if (minfreq <= 0 || maxfreq <= 0) {
    specfem::Logger::error("Frequencies must be positive for qplots");
    return false;
  }
  if (minfreq >= maxfreq) {
    specfem::Logger::error("minfreq must be less than maxfreq for qplots");
    return false;
  }
  if (min_plot_freq <= 0 || max_plot_freq <= 0) {
    specfem::Logger::error("Plot frequencies must be positive for qplots");
    return false;
  }

  // Placeholder implementation for Q plot generation
  std::ostringstream message;
  message << "Generating Q plots with Q=" << Q << ", minfreq=" << minfreq
          << ", maxfreq=" << maxfreq;
  specfem::Logger::info(message.str());

  // Create directory for Q plots
  std::filesystem::path qplot_dir = output_dir;

  if (!std::filesystem::exists(qplot_dir)) {
    std::filesystem::create_directory(qplot_dir);
    specfem::Logger::info("Created directory: " + qplot_dir.string());
  } else {
    specfem::Logger::info("Directory already exists: " + qplot_dir.string());
  }

  // Get tau_sigma
  const auto tau_sigma = specfem::attenuation::compute_tau_sigma<
      specfem::constants::empirical::N_SLS>(1.0 / maxfreq, 1.0 / minfreq);

  // Compute tau_eps for the given Q
  const auto tau_eps = specfem::attenuation::compute_tau_eps<
      specfem::constants::empirical::N_SLS>(Q, tau_sigma, 1.0 / maxfreq,
                                            1.0 / minfreq);

  // Print tau_sigma and tau_eps for debugging
  specfem::Logger::info("Computed tau_sigma:");
  for (int j = 0; j < specfem::constants::empirical::N_SLS; ++j) {
    specfem::Logger::info("tau_sigma[" + std::to_string(j) +
                          "] = " + std::to_string(tau_sigma(j)));
  }
  specfem::Logger::info("Computed tau_eps:");
  for (int j = 0; j < specfem::constants::empirical::N_SLS; ++j) {
    specfem::Logger::info("tau_eps[" + std::to_string(j) +
                          "] = " + std::to_string(tau_eps(j)));
  }

  // Generate logspace frequencies for plotting
  const int NF = 1000;
  const auto frequencies =
      specfem::utilities::logspace<NF>(min_plot_freq, max_plot_freq);

  // Get the Q inverse from the Maxwell real and imaginary factors.
  const auto moduli =
      specfem::attenuation::maxwell<NF, specfem::constants::empirical::N_SLS>(
          frequencies, tau_sigma, tau_eps);

  // Struct holding Q_inverse
  Kokkos::View<type_real[NF], Kokkos::LayoutRight, Kokkos::HostSpace> Q_inverse(
      "Q_inverse", NF);

  // Compute Q^-1 = imag / real
  for (int i = 0; i < NF; ++i) {
    Q_inverse(i) = moduli.imag(i) / moduli.real(i);
  }

  // Individual Q^-1 and real modulus for each SLS
  Kokkos::View<type_real[NF][specfem::constants::empirical::N_SLS],
               Kokkos::LayoutRight, Kokkos::HostSpace>
      individual_Q_inverse("individual_Q_inverse", NF);
  Kokkos::View<type_real[NF][specfem::constants::empirical::N_SLS],
               Kokkos::LayoutRight, Kokkos::HostSpace>
      individual_real("individual_real", NF);

  for (int j = 0; j < specfem::constants::empirical::N_SLS; ++j) {
    std::pair<int, int> subview_range(j, j + 1);
    auto maxwell_factors_j = specfem::attenuation::maxwell<NF, 1>(
        frequencies, Kokkos::subview(tau_sigma, subview_range),
        Kokkos::subview(tau_eps, subview_range));
    for (int i = 0; i < NF; ++i) {
      individual_Q_inverse(i, j) =
          maxwell_factors_j.imag(i) / maxwell_factors_j.real(i);
      individual_real(i, j) = maxwell_factors_j.real(i);
    }
  }

  // Open file for writing
  std::ofstream qplot_file(qplot_dir /
                           ("Q_inverse_Q" + std::to_string(Q) + ".txt"));
  if (!qplot_file.is_open()) {
    throw std::runtime_error("Failed to open file for writing Q plot");
  }

  // Write header
  qplot_file << "# Frequency(Hz), Q^-1, M1";
  for (int j = 0; j < specfem::constants::empirical::N_SLS; ++j) {
    qplot_file << ", Q^-1 - SLS " << j;
  }
  for (int j = 0; j < specfem::constants::empirical::N_SLS; ++j) {
    qplot_file << ", M1 - SLS " << j;
  }
  qplot_file << "\n";

  // Write frequencies, Q^-1, real modulus, and per-SLS values
  for (int i = 0; i < NF; ++i) {
    qplot_file << std::scientific << frequencies(i) << "," << std::scientific
               << Q_inverse(i) << "," << std::scientific << moduli.real(i);

    for (int j = 0; j < specfem::constants::empirical::N_SLS; ++j) {
      qplot_file << "," << std::scientific << individual_Q_inverse(i, j);
    }
    for (int j = 0; j < specfem::constants::empirical::N_SLS; ++j) {
      qplot_file << "," << std::scientific << individual_real(i, j);
    }
    qplot_file << "\n";
  }
  qplot_file.close();

  specfem::Logger::info(
      "Q plot values generated successfully: " +
      (qplot_dir / ("Q_inverse_Q" + std::to_string(Q) + ".txt")).string());
  specfem::Logger::info("You can use this data to create a plot of Q^-1 vs "
                        "frequency using your preferred plotting tool.");

  specfem::Logger::info("Q plot generation completed successfully");

  return true;
}
