#include "test_source_solutions.hpp"
#include "enumerations/wavefield.hpp"
#include "source_time_function/interface.hpp"

// Local constants since these would be set by the simulation.
int nsteps = 100;
type_real dt = 0.01;
int tshift = 0;            // for the single sources we are reading!
type_real user_t0 = -10.0; // user defined t0

// Internal t0 is being fixed using the halfduration of the source
specfem::wavefield::simulation_field wavefield_type =
    specfem::wavefield::simulation_field::forward;

// 2D source test solutions
const SourceVector2DType single_moment_tensor_2d = { std::make_shared<
    specfem::sources::moment_tensor<specfem::dimension::type::dim2> >(
    2000.0, 3000.0, 1.0, 1.0, 0.0,
    std::make_unique<specfem::forcing_function::Ricker>(nsteps, dt, 1.0, 30.0,
                                                        1.0e10, false),
    wavefield_type) };

const SourceVector2DType single_force_2d = {
  std::make_shared<specfem::sources::force<specfem::dimension::type::dim2> >(
      2500.0, 2500.0, 0.0,
      std::make_unique<specfem::forcing_function::Ricker>(nsteps, dt, 10.0, 5.0,
                                                          1.0e10, false),
      wavefield_type)
};

const SourceVector2DType single_cosserat_force_2d = { std::make_shared<
    specfem::sources::cosserat_force<specfem::dimension::type::dim2> >(
    2500.0, 2500.0, 0.0, 1.0, 0.0,
    std::make_unique<specfem::forcing_function::Ricker>(nsteps, dt, 10.0, 0.0,
                                                        1e10, false),
    wavefield_type) };

const SourceVector2DType multiple_sources_2d = {
  std::make_shared<
      specfem::sources::moment_tensor<specfem::dimension::type::dim2> >(
      2000.0, 3000.0, 1.0, 1.0, 0.0,
      std::make_unique<specfem::forcing_function::Ricker>(nsteps, dt, 1.0, 30.0,
                                                          1.0e10, false),
      wavefield_type),
  std::make_shared<specfem::sources::force<specfem::dimension::type::dim2> >(
      2500.0, 2500.0, 0.0,
      std::make_unique<specfem::forcing_function::Ricker>(nsteps, dt, 10.0, 5.0,
                                                          1.0e10, false),
      wavefield_type)
};

// 3D source test solutions
const SourceVector3DType single_force_3d = {
  std::make_shared<specfem::sources::force<specfem::dimension::type::dim3> >(
      2500.0, 2500.0, 2500.0, 0.0, 0.0, 0.0,
      std::make_unique<specfem::forcing_function::Ricker>(nsteps, dt, 10.0, 5.0,
                                                          1.0e10, false),
      wavefield_type)
};

const SourceVector3DType single_moment_tensor_3d = { std::make_shared<
    specfem::sources::moment_tensor<specfem::dimension::type::dim3> >(
    2000.0, 3000.0, 2000.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
    std::make_unique<specfem::forcing_function::Ricker>(nsteps, dt, 1.0, 30.0,
                                                        1.0e10, false),
    wavefield_type) };

const SourceVector3DType multiple_sources_3d = {
  std::make_shared<specfem::sources::force<specfem::dimension::type::dim3> >(
      2500.0, 2500.0, 2500.0, 0.0, 0.0, 0.0,
      std::make_unique<specfem::forcing_function::Ricker>(nsteps, dt, 10.0, 5.0,
                                                          1.0e10, false),
      wavefield_type),
  std::make_shared<
      specfem::sources::moment_tensor<specfem::dimension::type::dim3> >(
      2000.0, 3000.0, 2000.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
      std::make_unique<specfem::forcing_function::Ricker>(nsteps, dt, 1.0, 30.0,
                                                          1.0e10, false),
      wavefield_type)
};
