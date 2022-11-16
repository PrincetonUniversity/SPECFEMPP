#ifndef PARAMS_H
#define PARAMS_H

#include "../include/config.h"
#include <string>

namespace specfem {
struct parameters {
  std::string title, model, save_model, seismotype, tomography_file, mesh_file,
      nodes_coords_file, materials_file, free_surface_file, axial_elements_file,
      absorbing_surface_file, acoustic_forcing_surface_file,
      absorbing_cpml_file, tangential_detection_curve_file, interfacesfile;
  int simulation_type, noise_tomography, nproc, partitioning_type, ngnod, nstep,
      time_stepping_scheme, setup_with_binary_database, n_sls,
      nt_dump_attenuation, nsources, noise_source_time_function_type,
      ntstep_between_output_sample, ntstep_between_output_seismos,
      nreceiversets, ntstep_between_compute_kernels, nbmodels, nx_param,
      nbregions, nelem_PML_thickness;
  bool save_forward, axisym, p_sv, gpu_mode, attenuation_viscoelastic,
      attenuation_viscoacoustic, use_solvopt, attenuation_poro_fluid_part,
      undo_attenuation_and_or_PML, force_normal_to_surface, initialfield,
      add_Bielak_conditions_bottom, add_Bielak_conditions_right,
      add_Bielak_conditions_top, add_Bielak_conditions_left, acoustic_forcing,
      write_moving_sources_database, use_trick_for_better_pressure,
      save_ASCII_seismograms, save_binary_seismograms_single,
      rec_normal_to_surface, save_binary_seismograms_double, su_format,
      use_existing_stations, save_ASCII_kernels, approximate_hess_KL,
      no_backward_reconstruction, PML_boundary_conditions, rotate_PML_activate,
      PML_parameter_adjustment, stacey_absorbing_boundary_conditions,
      add_periodic_conditions, read_external_mesh, absorbbottom, absorbright,
      absorbtop, absorbleft, read_velocities_at_f0;
  type_real dt, attenuation_f0_reference, Q0_poroelastic, freq0_poroelastic,
      user_t0, anglerec, rotate_PML_angle, k_min_PML, k_max_PML,
      damping_change_factor_acoustic, damping_change_factor_elastic,
      periodic_horiz_dist, xmin_param, xmax_param;
};
} // namespace specfem

#endif
