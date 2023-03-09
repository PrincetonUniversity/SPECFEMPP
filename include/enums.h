#ifndef ENUMS_H
#define ENUMS_H

namespace specfem {

namespace elements {
enum type {
  elastic,    ///< elastic element
  acoustic,   ///< acoustic element
  poroelastic ///< poroelastic element
};
} // namespace elements

namespace wave {
enum type {
  p_sv, ///< P-SV wave
  sh    ///< SH wave
};
} // namespace wave

namespace seismogram {
enum type {
  displacement, ///< Displacement seismogram
  velocity,     ///< Velocity Seismogram
  acceleration  ///< Acceleration seismogram
};

namespace format {
enum type {
  seismic_unix, ///< Seismic unix output format
  ascii         ///< ASCII output format
};
}
} // namespace seismogram

} // namespace specfem

#endif
