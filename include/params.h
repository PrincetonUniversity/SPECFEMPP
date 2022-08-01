#include "../include/config.h"
#include <string>

namespace parameters {
struct parameters {
  std::string title, model, save_model, seismotype, tomography_file, mesh_file,
      nodes_coords_file, materials_file, free_surface_file, axial_elements_file,
      absorbing_surface_file, acoustic_forcing_surface_file,
      absorbing_cpml_file, tangential_detection_curve_file, interfacesfile;
  int simulation_type, noise_tomography, nproc, partitioning_type, ngnod, nstep,
      time_stepping_scheme, setup_with_binary_databases, n_sls,
      nt_dump_attenuation, nsources, noise_source_time_fuctions_type,
      ntstep_between_output_sample, nt_step_between_seismos, receiversets,
      ntstep_between_compute_kernels, nbmodels, nx_param, nbregions,
      nelem_PML_thickness;
  bool save_forward, axisym, p_sv, gpu_mode, attenuation_viscoelastic,
      attenuation_viscoacoustic, use_solvopt, attenuation_poro_fluid_part,
      undo_attenuation_and_or_PML, fource_normal_to_surface, initialfield,
      add_Bielak_conditions_bottom, add_Bielak_conditions_right,
      add_Bielak_conditions_top, add_bielak_conditions_left, acoustic_forcing,
      write_moving_sources_database, use_trick_for_better_pressure,
      save_ASCII_seismos, save_binary_seismograms_single, rec_normal_to_surface,
      savebiary_seismograms_double, su_format, use_existing_stations,
      save_ASCII_kernels, approximate_hess_KL, no_backwward_reconstruction,
      pml_boundary_conditions, rotate_PML_activate, PML_parameter_adjustment,
      stacey_absorbing_boundary_conditions, add_periodic_conditions,
      read_external_mesh, absorbbottom, absorbright, absorbtop, absorbleft;
  real_type dt, attenuation_f0_reference, read_velocities_at_f0, Q0_poroelastic,
      freq0_poroelastic, user_t0, angle_rec, rotate_PML_angle, k_min_PML,
      k_max_PML, damping_change_factor_acoustic, damping_change_factor_elastic,
      periodic_horiz_dist, xmin_param, xmax_param;
}
} // namespace parameters
