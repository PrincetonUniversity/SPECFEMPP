#include "../include/parameter_reader.h"
#include <fstream>
#include <iostream>

IO::param_file::param_file(std::string param_file) {
  this->filename = param_file;
}

void IO::param_file::open() { this->stream.open(this->filename); }

void IO::param_file::close() { this->stream.close(); }

std::string IO::param_file::param_read(std::string name,
                                       bool start_from_beginning = true) {

  std::string string_read;
  //  Trim the parameter name
  std::string namecopy = name.Trim();
  //  Remove dot terminated prefix
  size_t pos = namecopy.find('.');
  if (pos == std::string::npos) {
    std::string namecopy2 = namecopy;
  } else {
    std::string namecopy2 = namecopy.substr(pos);
  }

  std::regex pattern(
      "^[ \t]*([^# \t]+)[ \t]*=[ \t]*([^# \t]+([ \t]+[^# \t]+)*)");

  if (start_from_beginning)
    this->stream.seekg(0, ios::beg);
  std::string line;
  std::cmatch parameter;
  while (getline(stream, line)) {

    std::regex_search(line, parameter, pattern);

    if (parameter.size() == 0)
      continue;

    if (parameter[1] != namecopy2) {
      continue;
    }

    string_read = parameter[1];

    return string_read;
  }

  return string_read;
}

void IO::param_file::read(int &value, std::string name,
                          bool start_from_beginning = true) {
  std::string string_read = IO::param_file::param_read(name);
  if (string_read == "")
    throw std::runtime_error("Missing parameter " + name);
  value = std::stoi(string_read);
  return;
}

void IO::param_file::read(type_real &value, std::string name,
                          bool start_from_beginning = true) {
  std::string string_read = IO::param_file::param_read(name);
  if (string_read == "")
    throw std::runtime_error("Missing parameter " + name);
  value = std::atof(string_read);
}

void IO::param_file::read(bool &value, std::string name,
                          bool start_from_beginning = true) {
  std::string string_read = IO::param_file::param_read(name);
  if (string_read == "")
    throw std::runtime_error("Missing parameter " + name);
  if (string_read == ".false.") {
    value = false;
    return;
  } else if (string_read == ".true") {
    value = true;
    return;
  }

  throw std::runtime_error("Missing parameter " + name);
  return;
}

void IO::param_file::read(std::string value, std::string name,
                          bool start_from_beginning = true) {
  value = IO::param_file::param_read(name);
  if (value == "")
    throw std::runtime_error("Missing parameter " + name);
  return;
}

void IO::read_parameters_file(std::string param_file,
                              specfem::parameters::parameters &param) {

  IO::param_file::param_file param_file(param_file);
  param_file.open();

  bool some_parameters_are_missing = false;
  std::string dummy;

  try {
    param_file.read(param.title, "title");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.simulation_type, "SIMULATION_TYPE");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.noise_tomography, "NOISE_TOMOGRAPHY");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.save_forward, "SAVE_FORWARD");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.save_forward, "SAVE_FORWARD");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.nproc, "NPROC");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.partitioning_type, "PARTITIONING_TYPE");
  } catch std::runtime_error &e {
    try {
      param_file.read(param.partioning_type, "partitioning_method");
      std::cout
          << "Warning: Deprecated parameter partitioning_method found in "
             "Par_file. \n          Please use parameter PARTITIONING_TYPE "
             "in future..."
          << std::endl;
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }
  }

  try {
    param_file.read(param.ngnod, "NGNOD");
  } catch std::runtime_error &e {
    try {
      param_file.read(param.ngnod, "ngnod");
      std::cout << "Warning: Deprecated parameter ngnod found in "
                   "Par_file. \n          Please use parameter NGNOD "
                   "in future..."
                << std::endl;
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }
  }

  try {
    param_file.read(param.nstep, "NSTEP");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.dt, "DT");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.time_stepping_scheme, "time_stepping_scheme");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.axisym, "AXISYM");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.p_sv, "P_SV");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.gpu_mode, "GPU_MODE");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.setup_with_binary_database,
                    "setup_with_binary_database");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.model, "MODEL");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.save_model, "SAVE_MODEL");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.setup_with_binary_database,
                    "setup_with_binary_database");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  /*

          Attenuation

  */

  try {
    param_file.read(param.attenuation_viscoelastic, "ATTENUATION_VISCOELASTIC");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.attenuation_viscoacoustic,
                    "ATTENUATION_VISCOACOUSTIC");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.nsls, "NSLS");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.attenuation_f0_reference, "ATTENUATION_f0_REFERENCE");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(dummy, "f0_REFERENCE");
    throw std::runtime_error(
        "Parameter f0_attenuation in the Par_file is now called "
        "ATTENUATION_f0_REFERENCE \n in order to use the same name as in the "
        "3D code (SPECFEM3D). \n Please rename it in your Par_file and start "
        "the code again.")
  } catch std::runtime_error &e {
    continue;
  }

  try {
    param_file.read(param.read_velocities_at_f0, "READ_VELOCITIES_AT_f0");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.use_solvopt, "USE_SOLVOPT");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.attenuation_poro_fluid_part,
                    "ATTENUATION_PORO_FLUID_PART");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.Q0_poroelastic, "Q0_poroelastic");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.freq0_poroelastic, "freq0_poroelastic");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.undo_attenuation_and_or_PML,
                    "UNDO_ATTENUATION_AND_OR_PML");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.nt_dump_attenuation, "NT_DUMP_ATTENUATION");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  /*

      sources

  */

  try {
    param_file.read(param.nsources, "NSOURCES");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.force_normal_to_surface, "force_normal_to_surface");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.initialfield, "initialfield");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.add_Bielak_conditions_bottom,
                    "add_Bielak_conditions_bottom");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.add_Bielak_conditions_right,
                    "add_Bielak_conditions_right");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.add_Bielak_conditions_top,
                    "add_Bielak_conditions_top");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.add_Bielak_conditions_left,
                    "add_Bielak_conditions_left");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.acoustic_forcing, "ACOUSTIC_FORCING");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.noise_source_time_function_type,
                    "noise_source_time_function_type");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.write_moving_sources_database,
                    "write_moving_sources_database");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  /*

      Receivers

  */

  try {
    param_file.read(param.seismotype, "seismotype");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.ntstep_between_output_sample,
                    "NTSTEP_BETWEEN_OUTPUT_SAMPLE");
  } catch std::runtime_error &e {
    try {
      param_file.read(param.ntstep_between_output_sample, "subsamp_seismos");
      std::cout << "Warning: Deprecated parameter subsamp_seismos found in "
                   "Par_file. \n          Please use parameter "
                   "NTSTEP_BETWEEN_OUTPUT_SAMPLE in future..."
                << std::endl;
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }
  }

  try {
    param_file.read(param.use_trick_for_better_pressure,
                    "USE_TRICK_FOR_BETTER_PRESSURE");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.use_trick_for_better_pressure,
                    "USE_TRICK_FOR_BETTER_PRESSURE");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.ntstep_between_output_seismos,
                    "NTSTEP_BETWEEN_OUTPUT_SEISMOS");
  } catch std::runtime_error &e {
    try {
      param_file.read(param.ntstep_between_output_sample,
                      "NSTEP_BETWEEN_OUTPUT_SEISMOS");
      std::cout << "Warning: Deprecated parameter NSTEP_BETWEEN_OUTPUT_SEISMOS "
                   "found in "
                   "Par_file. \n          Please use parameter "
                   "NTSTEP_BETWEEN_OUTPUT_SEISMOS in future..."
                << std::endl;
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }
  }

  try {
    param_file.read(param.user_t0, "USER_T0");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.save_ASCII_seismograms, "save_ASCII_seismograms");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.save_binary_seismograms_single,
                    "save_binary_seismograms_single");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.save_binary_seismograms_double,
                    "save_binary_seismograms_double");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.su_format, "SU_FORMAT");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.use_existing_stations, "use_existing_STATIONS");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.nreceiversets, "nreceiversets");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.anglerec, "anglerec");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.rec_normal_to_surface, "rec_normal_to_surface");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  /*

      adjoint kernel

  */

  try {
    param_file.read(param.save_ASCII_kernels, "save_ASCII_kernels");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.ntstep_between_compute_kernels,
                    "NTSTEP_BETWEEN_COMPUTE_KERNELS");
  } catch std::runtime_error &e {
    try {
      param_file.read(param.ntstep_between_compute_kernels,
                      "NSTEP_BETWEEN_COMPUTE_KERNELS");
      std::cout
          << "Warning: Deprecated parameter NSTEP_BETWEEN_COMPUTE_KERNELS "
             "found in "
             "Par_file. \n          Please use parameter "
             "NTSTEP_BETWEEN_COMPUTE_KERNELS in future..."
          << std::endl;
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }
  }

  try {
    param_file.read(param.approximate_hess_KL, "APPROXIMATE_HESS_KL");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.no_backward_reconstruction,
                    "NO_BACKWARD_RECONSTRUCTION");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  /*

      Boundary Conditions

  */

  try {
    param_file.read(param.PML_boudary_conditions, "PML_BOUNDARY_CONDITIONS");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.nelem_PML_thickness, "NELEM_PML_THICKNESS");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.rotate_PML_activate, "ROTATE_PML_ACTIVATE");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.rotate_PML_angle, "ROTATE_PML_ANGLE");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.k_min_PML, "K_MIN_PML");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.k_max_PML, "K_MAX_PML");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.damping_change_factor_acoustic,
                    "damping_change_factor_acoustic");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.damping_change_factor_elastic,
                    "damping_change_factor_elastic");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.pml_parameter_adjustment, "PML_PARAMETER_ADJUSTMENT");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.stacey_absorbing_conditions,
                    "STACEY_ABSORBING_CONDITIONS");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.add_periodic_conditions, "ADD_PERIODIC_CONDITIONS");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.periodic_horiz_dist, "PERIODIC_HORIZ_DIST");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  /*

      Velocity and Density Models

  */

  try {
    param_file.read(param.nbmodels, "nbmodels");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.tomography_file, "TOMOGRAPHY_FILE");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  try {
    param_file.read(param.read_external_mesh, "read_external_mesh");
  } catch std::runtime_error &e {
    std::cout << e << std::endl;
    some_parameters_are_missing = true;
  }

  if (param.read_external_mesh) {
    try {
      param_file.read(param.mesh_file, "mesh_file");
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }

    try {
      param_file.read(param.nodes_coords_file, "nodes_coords_file");
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }

    try {
      param_file.read(param.materials_file, "materials_file");
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }

    try {
      param_file.read(param.free_surface_file, "free_surface_file");
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }

    try {
      param_file.read(param.axial_elements_file, "axial_elements_file");
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }

    try {
      param_file.read(param.absorbing_surface_file, "absorbing_surface_file");
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }

    try {
      param_file.read(param.acoustic_forcing_surface_file,
                      "acoustic_forcing_surface_file");
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }

    try {
      param_file.read(param.absorbing_cpml_file, "absorbing_cpml_file");
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }

    try {
      param_file.read(param.tangential_detection_curve_file,
                      "tangential_detection_curve_file");
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }
  } else {
    try {
      param_file.read(param.interfacesfile, "interfacesfile");
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }

    try {
      param_file.read(param.xmin_param, "xmin");
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }

    try {
      param_file.read(param.xmax_param, "xmax");
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }

    try {
      param_file.read(param.nx_param, "nx");
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }

    try {
      param_file.read(param.absorbbottom, "absorbbottom");
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }

    try {
      param_file.read(param.absorbtop, "absorbtop");
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }

    try {
      param_file.read(param.absorbright, "absorbright");
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }

    try {
      param_file.read(param.absorbleft, "absorbleft");
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }

    try {
      param_file.read(param.nbregions, "nbregions");
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }
  }

  param_file.close();
  return;
}

void IO::read_sources_file(std::string source_file,
                           std::vector<specfem::sources::source> &sources,
                           int nsources) {

  IO::param_file::param_file param_file(param_file);
  param_file.open();

  // Have to add checks to figure out if sources file is in correct format.

  for (int i = 0; i < nsources, nsources) {
    specfem::sources::source source;

    try {
      param_file.read(source.source_surf, "source_surf", false);
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }

    try {
      param_file.read(source.x_source, "x_source", false);
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }

    try {
      param_file.read(source.z_source, "z_source", false);
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }

    try {
      param_file.read(source.source_type, "source_type", false);
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }

    try {
      param_file.read(source.time_function_type, "time_function_type", false);
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }

    try {
      param_file.read(source.name_of_source_file, "name_of_source_file", false);
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }

    try {
      param_file.read(source.burst_band_width, "burst_band_width", false);
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }

    try {
      param_file.read(source.f0_source, "f0_source", false);
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }

    try {
      param_file.read(source.tshift_src, "tshift_src", false);
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }

    try {
      param_file.read(source.anglesource, "anglesource", false);
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }

    try {
      param_file.read(source.Mxx, "Mxx", false);
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }

    try {
      param_file.read(source.Mxz, "Mxz", false);
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }

    try {
      param_file.read(source.Mzz, "Mzz", false);
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }

    try {
      param_file.read(source.factor, "factor", false);
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }

    try {
      param_file.read(source.vx_source, "vx_source", false);
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }

    try {
      param_file.read(source.vz_source, "vz_source", false);
    } catch std::runtime_error &e {
      std::cout << e << std::endl;
      some_parameters_are_missing = true;
    }

    sources.push_back(source);
  }

  return;
}
