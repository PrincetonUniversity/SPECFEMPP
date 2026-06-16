#include "test_source_solutions.hpp"
#include "specfem/coordinate_systems/cartesian.hpp"
#include "specfem/coordinate_systems/geographic.hpp"
#include "specfem/datetime.hpp"
#include "specfem/enums.hpp"
#include "specfem/io/sources/impl/timing.hpp"
#include "specfem/source_time_functions.hpp"
#include "specfem/source_time_functions/gaussianhdur.hpp"

// Local constants since these would be set by the simulation.
int nsteps = 100;
type_real dt = 0.01;
int tshift = 0;            // for the single sources we are reading!
type_real user_t0 = -10.0; // user defined t0

// Internal t0 is being fixed using the halfduration of the source
specfem::simulation::field_type wavefield_type =
    specfem::simulation::field_type::forward;

// 2D source test solutions
const SourceVector2DType single_moment_tensor_2d = { std::make_shared<
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim2>>(
    2000.0, 3000.0, 1.0, 1.0, 0.0,
    std::make_unique<specfem::source_time_functions::Ricker>(
        nsteps, dt, 1.0, 30.0, 1.0e10, false),
    wavefield_type) };

const SourceVector2DType single_force_2d = { std::make_shared<
    specfem::sources::force<specfem::element::dimension_tag::dim2>>(
    2500.0, 2500.0, 0.0,
    std::make_unique<specfem::source_time_functions::Ricker>(
        nsteps, dt, 10.0, 5.0, 1.0e10, false),
    wavefield_type) };

const SourceVector2DType single_cosserat_force_2d = { std::make_shared<
    specfem::sources::cosserat_force<specfem::element::dimension_tag::dim2>>(
    2500.0, 2500.0, 0.0, 1.0, 0.0,
    std::make_unique<specfem::source_time_functions::Ricker>(nsteps, dt, 10.0,
                                                             0.0, 1e10, false),
    wavefield_type) };

const SourceVector2DType multiple_sources_2d = {
  std::make_shared<
      specfem::sources::moment_tensor<specfem::element::dimension_tag::dim2>>(
      2000.0, 3000.0, 1.0, 1.0, 0.0,
      std::make_unique<specfem::source_time_functions::Ricker>(
          nsteps, dt, 1.0, 30.0, 1.0e10, false),
      wavefield_type),
  std::make_shared<
      specfem::sources::force<specfem::element::dimension_tag::dim2>>(
      2500.0, 2500.0, 0.0,
      std::make_unique<specfem::source_time_functions::Ricker>(
          nsteps, dt, 10.0, 5.0, 1.0e10, false),
      wavefield_type)
};

// 3D source test solutions
const SourceVector3DType single_force_3d = { std::make_shared<
    specfem::sources::force<specfem::element::dimension_tag::dim3>>(
    2500.0, 2500.0, 2500.0, 0.0, 0.0, 0.0,
    std::make_unique<specfem::source_time_functions::Ricker>(
        nsteps, dt, 10.0, 5.0, 1.0e10, false),
    wavefield_type) };

const SourceVector3DType single_moment_tensor_3d = { std::make_shared<
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>>(
    2000.0, 3000.0, 2000.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
    std::make_unique<specfem::source_time_functions::Ricker>(
        nsteps, dt, 1.0, 30.0, 1.0e10, false),
    wavefield_type) };

// Same as single_moment_tensor_3d but from geographic latitude/longitude/depth
const SourceVector3DType single_moment_tensor_geographic_yaml_3d = {
  std::make_shared<
      specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>>(
      std::make_unique<specfem::coordinate_systems::geographic_coordinates>(
          2.674, 51.561, 2000.0),
      1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
      std::make_unique<specfem::source_time_functions::Ricker>(
          nsteps, dt, 1.0, 30.0, 1.0e10, false),
      wavefield_type)
};

const SourceVector3DType multiple_sources_3d = {
  std::make_shared<
      specfem::sources::force<specfem::element::dimension_tag::dim3>>(
      2500.0, 2500.0, 2500.0, 0.0, 0.0, 0.0,
      std::make_unique<specfem::source_time_functions::Ricker>(
          nsteps, dt, 10.0, 5.0, 1.0e10, false),
      wavefield_type),
  std::make_shared<
      specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>>(
      2000.0, 3000.0, 2000.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
      std::make_unique<specfem::source_time_functions::Ricker>(
          nsteps, dt, 1.0, 30.0, 1.0e10, false),
      wavefield_type)
};

// 3D CMTSOLUTION expected sources (GaussianHdur STF, factor=1.0)
// Test data: Mxx=1e7, Myy=1e7, Mzz=0, Mxy=1e7, Mxz=0, Myz=0 in dyne-cm
// After *1e-7: Mxx=1.0, Myy=1.0, Mzz=0.0, Mxy=1.0, Mxz=0.0, Myz=0.0 in N-m
const SourceVector3DType single_moment_tensor_cmt_3d = []() {
  auto src = std::make_shared<
      specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>>(
      std::make_unique<specfem::coordinate_systems::cartesian_coordinates<
          specfem::element::dimension_tag::dim3>>(2000.0, 3000.0, 2000.0),
      1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
      std::make_unique<specfem::source_time_functions::GaussianHdur>(
          nsteps, dt, 1.0, 30.0, 1.0, false),
      wavefield_type);
  src->set_starttime(specfem::datetime::make(2000, 1, 1, 0, 0, 0.0));
  return SourceVector3DType{ std::move(src) };
}();

// Same physical source but parsed from Mrr/Mtt/Mpp labels with conversion
const SourceVector3DType spherical_moment_tensor_cmt_3d = []() {
  auto src = std::make_shared<
      specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>>(
      std::make_unique<specfem::coordinate_systems::cartesian_coordinates<
          specfem::element::dimension_tag::dim3>>(2000.0, 3000.0, 2000.0),
      1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
      std::make_unique<specfem::source_time_functions::GaussianHdur>(
          nsteps, dt, 1.0, 30.0, 1.0, false),
      wavefield_type);
  src->set_starttime(specfem::datetime::make(2000, 1, 1, 0, 0, 0.0));
  return SourceVector3DType{ std::move(src) };
}();

// Same physical source from latitude/longitude/depth (depth 2.0 km -> 2000 m)
const SourceVector3DType single_moment_tensor_geographic_cmt_3d = []() {
  auto src = std::make_shared<
      specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>>(
      std::make_unique<specfem::coordinate_systems::geographic_coordinates>(
          2.674, 51.561, 2000.0),
      1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
      std::make_unique<specfem::source_time_functions::GaussianHdur>(
          nsteps, dt, 1.0, 30.0, 1.0, false),
      wavefield_type);
  src->set_starttime(specfem::datetime::make(2000, 1, 1, 0, 0, 0.0));
  return SourceVector3DType{ std::move(src) };
}();

// 3D FORCESOLUTION expected source (Ricker STF, factor=1.0)
// Test data: factor=0, comp_X=1, comp_Y=0, comp_Z=0 -> fx=0, fy=0, fz=0
const SourceVector3DType single_force_forcesolution_3d = { std::make_shared<
    specfem::sources::force<specfem::element::dimension_tag::dim3>>(
    std::make_unique<specfem::coordinate_systems::cartesian_coordinates<
        specfem::element::dimension_tag::dim3>>(2500.0, 2500.0, 2500.0),
    0.0, 0.0, 0.0,
    std::make_unique<specfem::source_time_functions::Ricker>(nsteps, dt, 10.0,
                                                             5.0, 1.0, false),
    wavefield_type) };

// 3D CMTSOLUTION multiple sources: one Cartesian, one spherical
// Source 1: PDE 2000-01-01 00:00:00, Mxx=1e7... dyne-cm -> 1.0... N-m
// Source 2: PDE 2001-06-15 12:30:45.50, Mrr/Mtt/Mpp -> Cartesian
// Both carry starttimes; run through adjust_source_timing so expected tshifts
// match what read_sources produces.
const SourceVector3DType multiple_sources_cmt_3d = []() {
  auto src1 = std::make_shared<
      specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>>(
      std::make_unique<specfem::coordinate_systems::cartesian_coordinates<
          specfem::element::dimension_tag::dim3>>(2000.0, 3000.0, 2000.0),
      1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
      std::make_unique<specfem::source_time_functions::GaussianHdur>(
          nsteps, dt, 1.0, 30.0, 1.0, false),
      wavefield_type);
  src1->set_starttime(specfem::datetime::make(2000, 1, 1, 0, 0, 0.0));

  auto src2 = std::make_shared<
      specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>>(
      std::make_unique<specfem::coordinate_systems::cartesian_coordinates<
          specfem::element::dimension_tag::dim3>>(5000.0, 6000.0, 1000.0),
      2.0, 2.0, 0.0, 2.0, 0.0, 0.0,
      std::make_unique<specfem::source_time_functions::GaussianHdur>(
          nsteps, dt, 2.0, 10.0, 1.0, false),
      wavefield_type);
  src2->set_starttime(specfem::datetime::make(2001, 6, 15, 12, 30, 45.50));

  SourceVector3DType sources{ std::move(src1), std::move(src2) };
  specfem::io::sources_impl::adjust_source_timing<
      specfem::element::dimension_tag::dim3>(sources, user_t0);
  return sources;
}();

// 3D FORCESOLUTION multiple forces:
// Force 1: Ricker, factor=0, comp=(1,0,0) -> fx=0, fy=0, fz=0
// Force 2: GaussianHdur, factor=1e10, comp=(0,1,0) -> fx=0, fy=1e10, fz=0
const SourceVector3DType multiple_forces_forcesolution_3d = {
  std::make_shared<
      specfem::sources::force<specfem::element::dimension_tag::dim3>>(
      std::make_unique<specfem::coordinate_systems::cartesian_coordinates<
          specfem::element::dimension_tag::dim3>>(2500.0, 2500.0, 2500.0),
      0.0, 0.0, 0.0,
      std::make_unique<specfem::source_time_functions::Ricker>(nsteps, dt, 10.0,
                                                               5.0, 1.0, false),
      wavefield_type),
  std::make_shared<
      specfem::sources::force<specfem::element::dimension_tag::dim3>>(
      std::make_unique<specfem::coordinate_systems::cartesian_coordinates<
          specfem::element::dimension_tag::dim3>>(4000.0, 5000.0, 3000.0),
      0.0, 1.0e10, 0.0,
      std::make_unique<specfem::source_time_functions::GaussianHdur>(
          nsteps, dt, 5.0, 2.0, 1.0, false),
      wavefield_type)
};
